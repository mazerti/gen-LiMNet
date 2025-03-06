import cells
from sequence_loader import DataLoader
from metrics import (
    accuracy,
    auc,
    precision,
    recall,
    drawPrecisionRecallCurve,
    dynamicWeightedBCE,
    loggedLoss,
)
import memory_model as m
import outputs as o
import torch

conf = {
    "dataLoader": DataLoader(
        paths=["/data/gen-limnet/tooSmall.csv"],
        sourceColumn="source",
        targetColumn="target",
        timestampColumn="timestamp",
        edgeFeatures={"feature"},
        nodeFeatures={},
        sequenceLength=49,
        sequenceStride=1,
        bipartite=True,
    ),
    "trainRatio": 0.8,
    "epochs": 20,
    "batchSize": 1,
    "trainBatchSize": 1,
    "validBatchSize": 1,
    "model": m.MemoryNetwork(m.SimpleMemoryUpdater(64, torch.nn.GRUCell)),
    "optimizer": lambda x: torch.optim.Adam(x, lr=0.2),
    "mixedPrecision": False,
    "outputs": [
        (
            m.UserDecoder,
            [
                o.UserClassification(
                    "label",
                    1,
                    metrics=[
                        precision,
                        recall,
                        accuracy,
                        auc,
                        drawPrecisionRecallCurve,
                    ],
                    loss=dynamicWeightedBCE(),
                )
            ],
        ),
    ],
}
