from sequence_loader import DataLoader
from metrics import (
    accuracy,
    auc,
    precision,
    recall,
    drawPrecisionRecallCurve,
    loggedLoss,
    dynamicWeightedBCE,
    sigmoidFocalLoss,
)
import memory_model as m
import outputs as o
import torch

conf = {
    "dataLoader": DataLoader(
        paths=["/data/gen-limnet/mooc_actions_merged.csv"],
        sourceColumn="USERID",
        targetColumn="TARGETID",
        timestampColumn="TIMESTAMP",
        edgeFeatures={"FEATURE0", "FEATURE1", "FEATURE2", "FEATURE3"},
        nodeFeatures={},
        sequenceLength=5_000,
        sequenceStride=1_000,
    ),
    "trainRatio": 0.8,
    "epochs": 10,
    "trainBatchSize": 196,
    "validBatchSize": 4096,
    "model": m.MemoryNetwork(m.SimpleMemoryUpdater(64, torch.nn.GRUCell)),
    "optimizer": lambda x: torch.optim.Adam(x, lr=0.1),
    "mixedPrecision": False,
    "outputs": [
        (
            m.EdgeDecoder,
            [
                o.EdgeBinaryClassification(
                    "LABEL",
                    1,
                    metrics=[
                        precision,
                        recall,
                        accuracy,
                        auc,
                        drawPrecisionRecallCurve,
                    ],
                    loss=dynamicWeightedBCE(),
                ),
            ],
        ),
    ],
}
