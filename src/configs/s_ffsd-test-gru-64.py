import cells
from sequence_loader import DataLoader
import metrics
import memory_model as m
import outputs as o
import torch


def auc(prepare_pred, device):
    def filter_unlabeled(output):
        y_pred, y_true = output
        mask = y_true != 2
        return y_pred[mask], y_true[mask]

    return metrics.auc(prepare_pred, device, task_transform=filter_unlabeled)


conf = {
    "dataLoader": DataLoader(
        paths=["/data/gen-limnet/S-FFSD.csv"],
        sourceColumn="Source",
        targetColumn="Target",
        timestampColumn="Time",
        edgeFeatures={"Amount", "Location", "Type"},
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
    "mixedPrecision": True,
    "outputs": [
        (
            m.EdgeDecoder,
            [
                o.EdgeMulticlassification(
                    labels={"Labels"},
                    taskName="dropout-prediction",
                    metrics=[auc],
                ),
            ],
        ),
    ],
}
