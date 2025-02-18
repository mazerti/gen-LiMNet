import cells
from sequence_loader import DataLoader
from metrics import accuracy, auc
import memory_model as m
import outputs as o
import torch

conf = {
    "dataLoader": DataLoader(
        paths=["/data/gen-limnet/reddit.csv"],
        sourceColumn="user_id",
        targetColumn="item_id",
        timestampColumn="timestamp",
        edgeFeatures={f"f{i}" for i in range(172)},
        nodeFeatures={},
        sequenceLength=5_000,
        sequenceStride=1_000,
    ),
    "trainRatio": 0.8,
    "epochs": 5,
    "trainBatchSize": 192,
    "validBatchSize": 4096,
    "model": m.MemoryNetwork(m.SimpleMemoryUpdater(64, torch.nn.GRUCell)),
    "optimizer": torch.optim.Adam,
    "mixedPrecision": False,
    "outputs": [
        (
            m.EdgeDecoder,
            [
                o.EdgeBinaryClassification("state_label", 1, metrics=[accuracy, auc]),
            ],
        ),
    ],
}
