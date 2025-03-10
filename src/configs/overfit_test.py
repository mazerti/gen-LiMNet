import cells
from sequence_loader import DataLoader
from metrics import (
    accuracy,
    auc,
    precision,
    recall,
    drawPrecisionRecallCurve,
    confusion_matrix,
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
        balanceOn="label",
        nodeFeatures={},
        sequenceLength=500,
        sequenceStride=100,
        bipartite=True,
    ),
    "trainRatio": 0.8,
    "epochs": 100,
    "batchSize": 192,
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
