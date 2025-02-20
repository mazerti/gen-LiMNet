import glob
import ipaddress
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from util import TimedStep, pandas2torch, numpy2torch


def computeDeltas(data, srcCol, dstCol, timestampCol):
    df = data[[timestampCol, srcCol, dstCol]].sort_values(by=[srcCol, timestampCol])
    df["deltaSrc"] = (
        df[timestampCol]
        .diff()
        .apply(lambda x: 0 if pd.isna(x) or x < 0 else x)
        .rename("deltaSrc")
    )
    df.sort_values(by=[dstCol, timestampCol])
    df["deltaDst"] = (
        df[timestampCol]
        .diff()
        .apply(lambda x: 0 if pd.isna(x) or x < 0 else x)
        .rename("deltaDst")
    )
    return df.sort_index()[["deltaSrc", "deltaDst"]]


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, edgeData, size, starts, Xv):
        super().__init__()
        self.starts = starts
        self.edgeData = edgeData
        self.size = size
        self.Xv = Xv

    def __getitem__(self, id):
        start = self.starts[id]
        stop = start + self.size

        E, Xe, labels = self.edgeData
        E = E[start:stop]
        Xe = Xe[start:stop]
        Xv = self.Xv[E]
        labels = tuple(lbls[start:stop] for lbls in labels)

        V = torch.unique(E)
        I = torch.arange(V.shape[0])
        IM = torch.full(
            (self.Xv.shape[0],), torch.iinfo(torch.int64).max, dtype=torch.int64
        )
        IM[V] = I

        return (IM[E], Xe, Xv, labels)

    def __len__(self):
        return len(self.starts)


class DataLoader:
    def __init__(
        self,
        paths,
        sequenceLength,
        sourceColumn="SOURCEID",
        targetColumn="TARGETID",
        timestampColumn="TIMESTAMP",
        sequenceStride=None,
        edgeFeatures=set(),
        nodeFeatures=set(),
        align=1,
        useDeltas=False,
    ):
        self.paths = paths
        self.sequenceLength = sequenceLength
        self.sequenceStride = sequenceStride or sequenceLength
        self.sourceColumn = sourceColumn
        self.targetColumn = targetColumn
        self.timestampColumn = timestampColumn
        self.edgeFeatures = edgeFeatures
        self.nodeFeatures = nodeFeatures
        self.align = align
        self.useDeltas = useDeltas

    def loadData(self, tasks):
        columns = set([self.sourceColumn, self.targetColumn, self.timestampColumn])
        columns.update(self.edgeFeatures)
        columns.update(*(task.required_features for task in tasks))

        files = tqdm(
            [file for path in self.paths for file in glob.iglob(path)],
            desc="loading data",
        )
        data = pd.concat(
            (pd.read_csv(file, usecols=columns) for file in files),
            ignore_index=True,
            copy=False,
        )

        with TimedStep("Sorting"):
            data.sort_values(
                self.timestampColumn, kind="quicksort", inplace=True, ignore_index=True
            )

        with TimedStep("Generating node IDs"):
            id2ip = pd.unique(
                data[[self.sourceColumn, self.targetColumn]].values.ravel("K")
            )
            ip2id = {ip: id for id, ip in enumerate(id2ip)}
            data.loc[:, (self.sourceColumn, self.targetColumn)] = data[
                [self.sourceColumn, self.targetColumn]
            ].map(ip2id.get)
            E = pandas2torch(
                data[[self.sourceColumn, self.targetColumn]], dtype=np.int64
            )
            ip2id = {str(ip): id for ip, id in ip2id.items()}

        with TimedStep("Computing labels"):
            num_outputs = max(task.output for task in tasks) + 1
            labels = tuple([] for _ in range(num_outputs))
            for task in tasks:
                labels[task.output].append(task.compute_labels(data, id2ip, ip2id))
            labels = tuple(torch.stack(lbls, dim=-1) for lbls in labels)

        with TimedStep("Computing node features"):
            if len(self.nodeFeatures) > 0:
                features = []
                # Add specific feature logic here
                Xv = torch.stack(features, dim=1)
            else:
                Xv = torch.ones(len(id2ip), 1)

        with TimedStep("Computing edge features"):
            Xnum = data[list(self.edgeFeatures)].select_dtypes(include=np.number)
            if not Xnum.empty:
                vMin, med, vMax = map(
                    lambda t: t[1], Xnum.quantile([0, 0.5, 1]).iterrows()
                )
                Xnum = (Xnum - med) / (vMax - vMin)

            Xcat = data[list(self.edgeFeatures)].select_dtypes(exclude=np.number)
            if not Xcat.empty:
                Xcat = pd.get_dummies(Xcat)

            if self.useDeltas:
                deltas = computeDeltas(
                    data,
                    self.sourceColumn,
                    self.targetColumn,
                    self.timestampColumn,
                )

            match (self.useDeltas, Xnum.empty, Xcat.empty):
                case (True, False, False):
                    Xe = deltas.join(Xnum).join(Xcat)
                case (True, False, True):
                    Xe = deltas.join(Xnum)
                case (True, True, False):
                    Xe = deltas.join(Xcat)
                case (True, True, True):
                    Xe = deltas
                case (False, False, False):
                    Xe = Xnum.join(Xcat)
                case (False, False, True):
                    Xe = Xnum
                case (False, True, False):
                    Xe = Xcat
                case (False, True, True):
                    Xe = np.zeros(data.shape[0], 1)

            totalFeatures = 2 * Xv.shape[1] + Xe.shape[1]
            if totalFeatures % self.align != 0:
                for i in range(self.align - (totalFeatures % self.align)):
                    Xe[f"_pad_{i}"] = 0
            Xe = Xe.dropna(axis=1)
            Xe = pandas2torch(Xe)

        numSequences = int((E.shape[0] - self.sequenceLength) / self.sequenceStride)
        starts = np.arange(numSequences) * self.sequenceStride
        dataset = SequenceDataset((E, Xe, labels), self.sequenceLength, starts, Xv)

        info = {
            "N": Xv.shape[0],
            "E": Xe.shape[0],
            "Fv": Xv.shape[1],
            "Fe": Xe.shape[1],
            "Nseqs": len(dataset),
            "id2ip": id2ip.tolist(),
            "ip2id": ip2id,
        }

        return dataset, info
