import glob
import importlib
import numpy as np
import os
import re
import sys
import time
import torch
from ignite.engine import Events
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split


class TimedStep:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        tqdm.write(f"{self.name}...")
        self.start_time = time.perf_counter()

    def __exit__(self, *exc):
        t = time.perf_counter()
        tqdm.write("  Done {:.2f} s".format(t - self.start_time))


def numpy2torch(data, dtype=np.float32):
    return torch.from_numpy(dtype(data))


def pandas2torch(data, dtype=np.float32):
    return numpy2torch(data.to_numpy(), dtype)


def loadConfigurationFile(configFile):
    confDir, confModule = os.path.split(configFile)
    if confDir != "":
        sys.path.append(confDir)
    if confModule.endswith(".py"):
        confModule = confModule[:-3]
    return importlib.import_module(confModule)


def latestCheckpoint(baseFolder):
    extractLastNumber = lambda s: int(re.findall("\d+", s)[-1])
    return sorted(glob.glob(f"{baseFolder}/checkpoint*"), key=extractLastNumber)[-1]


def tqdmHandler(engine, epochs=False, **kwargs):
    startEvent = Events.STARTED if epochs else Events.EPOCH_STARTED
    stepEvent = Events.EPOCH_COMPLETED if epochs else Events.ITERATION_COMPLETED
    closeEvent = Events.COMPLETED if epochs else Events.EPOCH_COMPLETED

    pbar = None

    @engine.on(startEvent)
    def create_pbar(engine):
        nonlocal pbar
        total = engine.state.max_epochs if epochs else engine.state.epoch_length
        initial = engine.state.epoch if epochs else 0
        pbar = tqdm(total=total, initial=initial, **kwargs)

    @engine.on(stepEvent)
    def update_pbar(engine):
        pbar.update(1)

    @engine.on(closeEvent)
    def close_pbar(engine):
        nonlocal pbar
        pbar.close()
        pbar = None


def node_labels(interactions):
    uniqueNodes = interactions[:, 0].unique()
    nodeLabels = torch.zeros_like(uniqueNodes, dtype=torch.float)
    for index, node in enumerate(uniqueNodes):
        nodeInteractions = interactions[interactions[:, 0] == node]
        if (nodeInteractions[:, 1] == 1).any():
            nodeLabels[index] = 1
    return nodeLabels


def recursive_shape(things):
    def __recursive_shape(things):
        if hasattr(things, "shape"):
            return [f"{type(things)}:{things.shape}"]
        if type(things) is str:
            return []
        try:
            iter(things)
            res = [f"{type(things)}:{len(things)}"]
            for thing in things:
                for row in __recursive_shape(thing):
                    res.append(f"|- {row}")
            return res
        except TypeError:
            return [f"{type(things)}"]

    return "\n".join(__recursive_shape(things))


def split_dataset(dataset, baseFolder, resume, trainRatio, device):
    if resume:
        with open(f"{baseFolder}/seed.txt", "r") as f:
            seed = int(f.read())
    else:
        seed = np.random.randint(2**31)
        with open(f"{baseFolder}/seed.txt", "w") as f:
            f.write(str(seed))
    torch.manual_seed(seed)

    allNodes = dataset.edgeData[0][:, 0]
    interactionLabels = dataset.edgeData[2][0][
        :, 0
    ]  # Assuming only one label TODO: extend to more
    interactions = torch.stack((allNodes, interactionLabels)).transpose(0, 1)

    uniqueNodes = allNodes.unique()
    nodeLabels = torch.zeros_like(uniqueNodes, dtype=torch.float)
    for index, node in enumerate(uniqueNodes):
        nodeInteractions = interactions[allNodes == node]
        if (nodeInteractions[:, 1] == 1).any():
            nodeLabels[index] = 1
    nodeLabels = node_labels(interactions)

    trainNodes, valNodes = train_test_split(
        uniqueNodes.numpy(),
        stratify=nodeLabels.numpy(),
        test_size=trainRatio,
        random_state=42,
    )

    trainMask = torch.isin(allNodes, torch.tensor(trainNodes))
    valMask = torch.isin(allNodes, torch.tensor(valNodes))
    dataset.set_masks(trainMask, valMask)
