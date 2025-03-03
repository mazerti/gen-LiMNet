import cells
from sequence_loader import DataLoader
from metrics import (
    accuracy,
    auc,
    precision,
    recall,
    drawPrecisionRecallCurve,
    dynamicWeightedBCE,
)
import memory_model as m
import outputs as o
import torch

conf = {
    "dataLoader": DataLoader(
        paths=["/data/gen-limnet/wikipedia.csv"],
        sourceColumn="user_id",
        targetColumn="item_id",
        timestampColumn="timestamp",
        edgeFeatures={f"f{i}" for i in range(172)},
        nodeFeatures={},
        sequenceLength=5_000,
        sequenceStride=1_000,
        bipartite=True,
    ),
    "trainRatio": 0.8,
    "epochs": 5,
    "trainBatchSize": 192,
    "validBatchSize": 4096,
    "model": m.MemoryNetwork(m.SimpleMemoryUpdater(64, torch.nn.GRUCell)),
    "optimizer": lambda x: torch.optim.Adam(x, lr=0.1),
    "mixedPrecision": False,
    "outputs": [
        (
            # m.EdgeDecoder,
            # [
            #     o.EdgeBinaryClassification(
            #         "state_label",
            #         1,
            #         metrics=[
            #             precision,
            #             recall,
            #             accuracy,
            #             auc,
            #             drawPrecisionRecallCurve,
            #         ],
            #         loss=dynamicWeightedBCE(),
            #     ),
            # ],
            m.UserDecoder,
            [
                o.UserClassification(
                    "state_label",
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
