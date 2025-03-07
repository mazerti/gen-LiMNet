import glob
import ipaddress
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from util import TimedStep, pandas2torch, numpy2torch
from sklearn.model_selection import train_test_split


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


def user_labels(data, label, positiveTag=1):
    return (
        pd.concat(
            (data["source"], data[label] == positiveTag),
            axis="columns",
        )
        .groupby(by="source", as_index=False)
        .any()
    )


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

        E, Xe, labels, masks = self.edgeData
        E = E[start:stop]
        Xe = Xe[start:stop]
        Xv = self.Xv[E]
        labels = tuple(lbls[start:stop] for lbls in labels)
        trainMask, valMask = masks
        trainMask = trainMask[start:stop]
        valMask = valMask[start:stop]

        V = torch.unique(E)
        I = torch.arange(V.shape[0])
        IM = torch.full(
            (self.Xv.shape[0],), torch.iinfo(torch.int64).max, dtype=torch.int64
        )
        IM[V] = I

        return (IM[E], Xe, Xv, labels, (trainMask, valMask))

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
        balanceOn="LABEL",
        sequenceStride=None,
        edgeFeatures=set(),
        nodeFeatures=set(),
        align=1,
        useDeltas=False,
        bipartite=False,
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
        self.bipartite = bipartite
        self.balanceOn = balanceOn

    def loadData(self, tasks, baseFolder, resume, trainRatio):
        """
        Load the dataset from the listed files.
        Also generates the features and labels.

        Returns a pytorch Dataset instance along with a summary of the data.
        """
        columns = self.aggregate_columns(tasks)

        files = tqdm(self.list_data_files(), desc="loading data")
        data = self.read_files(files, columns)

        self.sort_data(data)

        id2ip, ip2id, E = self.generate_node_ids(data)
        labels = self.generate_labels(tasks, data)
        Xe, Xv = self.generate_features(data, id2ip)
        masks = self.generate_masks(
            data, self.balanceOn, baseFolder, resume, trainRatio
        )

        numSequences = int((E.shape[0] - self.sequenceLength) / self.sequenceStride)
        starts = np.arange(numSequences) * self.sequenceStride

        dataset = SequenceDataset(
            (
                pandas2torch(E, dtype=np.int64),
                pandas2torch(Xe),
                tuple(pandas2torch(label) for label in labels),
                tuple(pandas2torch(mask, dtype=np.bool) for mask in masks),
            ),
            self.sequenceLength,
            starts,
            pandas2torch(Xv, ),
        )
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

    def generate_masks(
        self, data, label, baseFolder, resume, trainRatio, positiveTag=1
    ):
        with TimedStep("Computing training/validation masks"):
            nodes = user_labels(data, label, positiveTag)
            trainNodes, valNodes = train_test_split(
                nodes["source"],
                stratify=nodes[label],
                test_size=trainRatio,
                random_state=self.set_seed(baseFolder, resume),
            )
            trainMask = data["source"].isin(trainNodes)
            valMask = data["source"].isin(valNodes)
        return trainMask, valMask

    def set_seed(self, baseFolder, resume):
        if resume:
            with open(f"{baseFolder}/seed.txt", "r") as f:
                seed = int(f.read())
        else:
            seed = np.random.randint(2**31)
            with open(f"{baseFolder}/seed.txt", "w") as f:
                f.write(str(seed))
        torch.manual_seed(seed)
        return seed

    def generate_features(self, data, id2ip):
        """Generate and clean features from raw data."""
        Xv = self.generate_node_features(id2ip)
        Xe = self.generate_edge_features(data, Xv)
        Xe = self.generate_feature_padding(Xv, Xe)
        Xe = Xe.dropna(axis=1)
        return Xe, Xv

    def generate_edge_features(self, data, Xv):
        """
        Generates the edge features from the raw data.
        """
        with TimedStep("Computing edge features"):
            Xnum = self.generate_numerical_features(data)
            Xcat = self.generate_categorical_features(data)
            deltas = (
                computeDeltas(data, "source", "target", "timestamp")
                if self.useDeltas
                else None
            )

            Xe = self.merge_features(data, Xnum, Xcat, deltas)

        return Xe

    def generate_feature_padding(self, Xv, Xe):
        totalFeatures = 2 * Xv.shape[1] + Xe.shape[1]
        if totalFeatures % self.align != 0:
            for i in range(self.align - (totalFeatures % self.align)):
                Xe[f"_pad_{i}"] = 0
        return Xe

    def generate_categorical_features(self, data):
        Xcat = data[list(self.edgeFeatures)].select_dtypes(exclude=np.number)
        if not Xcat.empty:
            Xcat = pd.get_dummies(Xcat)
        return Xcat

    def generate_numerical_features(self, data):
        Xnum = data[list(self.edgeFeatures)].select_dtypes(include=np.number)
        if not Xnum.empty:
            vMin, med, vMax = map(lambda t: t[1], Xnum.quantile([0, 0.5, 1]).iterrows())
            Xnum = (Xnum - med) / (vMax - vMin)
        return Xnum

    def merge_features(self, data, Xnum, Xcat, deltas):
        features = []
        if self.useDeltas:
            features.append(deltas)
        if not Xnum.empty:
            features.append(Xnum)
        if not Xcat.empty:
            features.append(Xcat)
        if len(features) == 0:
            features.append(pd.DataFrame(np.zeros((data.shape[0], 1))))
        return pd.concat(features, axis=1)

    def generate_node_features(self, id2ip):
        with TimedStep("Computing node features"):
            if len(self.nodeFeatures) > 0:
                features = []
                # Add specific feature logic here
                Xv = pd.concat(features, axis=1)
            else:
                Xv = pd.DataFrame(np.ones((len(id2ip), 1)))
        return Xv

    def generate_labels(self, tasks, data):
        with TimedStep("Computing labels"):
            num_outputs = max(task.output for task in tasks) + 1
            labels = tuple([] for _ in range(num_outputs))
            for task in tasks:
                labels[task.output].append(task.compute_labels(data))
            labels = tuple(pd.concat(lbls, axis=1) for lbls in labels)
        return labels

    def generate_node_ids(self, data):
        with TimedStep("Generating node IDs"):
            id2ip = pd.unique(data[["source", "target"]].values.ravel("K"))
            ip2id = {ip: id for id, ip in enumerate(id2ip)}
            data.loc[:, ("source", "target")] = data[["source", "target"]].map(
                ip2id.get
            )
            E = data[["source", "target"]]
            ip2id = {str(ip): id for ip, id in ip2id.items()}
        return id2ip, ip2id, E

    def sort_data(self, data):
        with TimedStep("Sorting"):
            data.sort_values(
                "timestamp", kind="quicksort", inplace=True, ignore_index=True
            )

    def read_files(self, files, columns):
        data = pd.concat(
            (pd.read_csv(file, usecols=columns) for file in files),
            ignore_index=True,
            copy=False,
        ).rename(
            columns={
                self.sourceColumn: "source",
                self.targetColumn: "target",
                self.timestampColumn: "timestamp",
            }
        )
        if self.bipartite:
            data["source"] = data["source"].apply(lambda x: f"s-{x}")
            data["target"] = data["target"].apply(lambda x: f"t-{x}")
        return data

    def list_data_files(self):
        return [file for path in self.paths for file in glob.iglob(path)]

    def aggregate_columns(self, tasks):
        columns = set([self.sourceColumn, self.targetColumn, self.timestampColumn])
        columns.update(self.edgeFeatures)
        columns.update(*(task.required_features for task in tasks))
        return columns
