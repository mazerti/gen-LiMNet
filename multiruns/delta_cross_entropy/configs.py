import cells
from sequence_loader import DataLoader

# from packet_header_loader import DataLoader
from metrics import accuracy, auc
import memory_model as mem
import language_model as lang
import outputs as o
import torch

trials = 1

datasets = {
    "mooc": (
        "/data/gen-limnet/mooc_actions_merged.csv",
        "USERID",
        "TARGETID",
        "TIMESTAMP",
        {"FEATURE0", "FEATURE1", "FEATURE2", "FEATURE3"},
        "LABEL",
        192,
        5,
    ),
    "wikipedia": (
        "/data/gen-limnet/wikipedia.csv",
        "user_id",
        "item_id",
        "timestamp",
        {f"f{i}" for i in range(172)},
        "state_label",
        192,
        5,
    ),
    "reddit": (
        "/data/gen-limnet/reddit.csv",
        "user_id",
        "item_id",
        "timestamp",
        {f"f{i}" for i in range(172)},
        "state_label",
        192,
        5,
    ),
}

cells = {
    # 'fastgrnn': (cells.FastGRNN   , lang.SimpleRecurrentLayer, False, mem.SimpleMemoryUpdater, False),
    "gru": (
        torch.nn.GRUCell,
        lang.SimpleRecurrentLayer,
        True,
        mem.SimpleMemoryUpdater,
        False,
    ),
    # 'lstm':     (torch.nn.LSTMCell, lang.LSTMRecurrentLayer  , True , mem.LSTMMemoryUpdater  , False),
}

embeddingSizes = [32, 64]

configs = {}

for datasetName, (
    datasetLocation,
    sourceColumn,
    targetColumn,
    timestampColumn,
    edgeFeatures,
    edgeLabels,
    trainBatchSize,
    epochs,
) in datasets.items():
    for cellName, (
        cellType,
        recLayerType,
        recMixedPrecision,
        memLayerType,
        memMixedPrecision,
    ) in cells.items():
        configName = f"{datasetName}-mem-{cellName}-64"
        configs[configName] = {
            "dataLoader": DataLoader(
                paths=[datasetLocation],
                sourceColumn=sourceColumn,
                targetColumn=targetColumn,
                timestampColumn=timestampColumn,
                edgeFeatures=edgeFeatures,
                nodeFeatures={},
                sequenceLength=5_000,
                sequenceStride=1_000,
                useDeltas=True,
            ),
            "trainRatio": 0.8,
            "epochs": epochs,
            "trainBatchSize": trainBatchSize,
            "validBatchSize": 4096,
            "model": mem.MemoryNetwork(mem.SimpleMemoryUpdater(64, torch.nn.GRUCell)),
            "optimizer": torch.optim.Adam,
            "mixedPrecision": False,
            "outputs": [
                (
                    mem.EdgeDecoder,
                    [
                        o.EdgeBinaryClassification(
                            edgeLabels,
                            1,
                            metrics=[accuracy, auc],
                            loss=torch.nn.CrossEntropyLoss(),
                        ),
                    ],
                ),
            ],
        }
