import torch
import pandas as pd
from util import pandas2torch

########## UTILITY FUNCTIONS ##########


def get_decoders(outputs):
    decoders = []
    for i, (decoder, tasks) in enumerate(outputs):
        pred_size = 0
        true_size = 0
        for task in tasks:
            task.pred_pos = pred_size
            task.pred_end = task.pred_pos + task.pred_size
            pred_size += task.pred_size
            task.true_pos = true_size
            task.true_end = task.true_pos + task.true_size
            true_size += task.true_size
        decoders.append(decoder(pred_size))
    return tuple(decoders)


def get_tasks(outputs, device):
    all_tasks = []
    for i, (_, tasks) in enumerate(outputs):
        for task in tasks:
            task.initialize(i, device)
            all_tasks.append(task)
    return all_tasks


########## BASE CLASSES ##########


class Task:
    def __init__(self, key, loss, activation, taskWeight=1, metrics=[]):
        self.key = key
        self.loss = loss
        self.activation = activation
        self.taskWeight = taskWeight
        self._metrics = metrics

    def initialize(self, output, device):
        self.output = output
        self.metrics = {
            f"{self.key}-{metric.__name__}": metric(self.prepare_predictions, device)
            for metric in self._metrics
        }
        self._metrics = None

    def extract(self, state_output):
        Ypred, Ytrue = state_output
        return (
            Ypred[self.output][..., self.pred_pos : self.pred_end],
            Ytrue[self.output][..., self.pred_pos : self.pred_end],
        )

    def compute_loss(self, state_output):
        Ypred, Ytrue = self.extract(state_output)
        Ypred = torch.squeeze(Ypred)
        Ytrue = torch.squeeze(Ytrue)
        return self.taskWeight * self.loss(Ypred, Ytrue)

    def prepare_predictions(self, state_output):
        Ypred, Ytrue = self.extract(state_output)
        return self.activation(Ypred), Ytrue

    def compute_labels(self, data):
        raise NotImplementedError("compute_labels should be implemented by subclasses")

    @property
    def required_features(self):
        return set()


class BinaryClassification(Task):
    def __init__(
        self, key, loss=torch.nn.BCEWithLogitsLoss(), activation=torch.sigmoid, **kwargs
    ):
        super().__init__(key, loss, activation, **kwargs)
        self.pred_size = 1
        self.true_size = 1


class MultiClassification(Task):
    def __init__(
        self,
        key,
        loss=torch.nn.CrossEntropyLoss(),
        activation=torch.nn.Softmax(),
        **kwargs,
    ):
        super().__init__(key, loss, activation, **kwargs)
        self.pred_size = 1
        self.true_size = 1


########## ACTUAL OUTPUTS ##########


class NodeIsMalicious(BinaryClassification):
    def __init__(self, **kwargs):
        super().__init__("node-is-malicious", **kwargs)

    @property
    def required_features(self):
        return {"traffic_type", "port_src", "port_dst"}

    def compute_labels(self, data):
        Ye = data["traffic_type"] == "mal"
        Ysrc = pandas2torch(Ye & (data["port_src"] > 10000))
        Ydst = pandas2torch(Ye & (data["port_dst"] > 10000))
        return torch.stack((Ysrc, Ydst), dim=1)


class NodeIsAttacked(BinaryClassification):
    def __init__(self, **kwargs):
        super().__init__("node-is-attacked", **kwargs)

    @property
    def required_features(self):
        return {"traffic_type", "port_src", "port_dst"}

    def compute_labels(self, data):
        Ye = data["traffic_type"] == "mal"
        Ysrc = pandas2torch(Ye & (data["port_src"] < 10000))
        Ydst = pandas2torch(Ye & (data["port_dst"] < 10000))
        return torch.stack((Ysrc, Ydst), dim=1)


class EdgeIsMalicious(BinaryClassification):
    def __init__(self, **kwargs):
        super().__init__("edge-is-malicious", **kwargs)

    @property
    def required_features(self):
        return {"traffic_type"}

    def compute_labels(self, data):
        return pandas2torch(data["traffic_type"] == "mal")


class EdgeBinaryClassification(BinaryClassification):
    def __init__(
        self, label: str, positiveTag, taskName="edge_label_classification", **kwargs
    ):
        self.label = label
        self.positiveTag = positiveTag
        super().__init__(taskName, **kwargs)

    @property
    def required_features(self):
        return {self.label}

    def compute_labels(self, data):
        return pandas2torch(data[self.label] == self.positiveTag)


class EdgeMulticlassification(MultiClassification):
    def __init__(self, labels, taskName="edge_label_classification", **kwargs):
        self.labels = labels
        super().__init__(taskName, **kwargs)

    @property
    def required_features(self):
        return self.labels

    def compute_labels(self, data):
        return pandas2torch(data[list(self.labels)])
