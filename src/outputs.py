import torch
import pandas as pd
from util import pandas2torch
import numpy as np

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
        return(Ypred[self.output], Ytrue[self.output])
        # return (
        #     Ypred[self.output][..., self.pred_pos : self.pred_end],
        #     Ytrue[self.output][..., self.pred_pos : self.pred_end],
        # )

    def compute_loss(self, state_output):
        Ypred, Ytrue = self.extract(state_output)
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


class Regression(Task):
    def __init__(
        self, key, loss=torch.nn.MSELoss(), activation=torch.nn.Identity(), **kwargs
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
        self, label: str, positiveTag, taskName="edge-label-classification", **kwargs
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
    def __init__(self, labels, taskName="edge-label-classification", **kwargs):
        self.labels = labels
        super().__init__(taskName, **kwargs)

    @property
    def required_features(self):
        return self.labels

    def compute_labels(self, data):
        return pandas2torch(data[list(self.labels)])


class UserClassification(EdgeBinaryClassification):
    """User classification after observing an interaction perform by that user."""

    def __init__(self, label, positiveTag, taskName="user-classification", **kwargs):
        super().__init__(label, positiveTag, taskName, **kwargs)

    def compute_labels(self, data):
        node_label = (
            pd.concat(
                (data["source"], data[self.label] == self.positiveTag),
                axis="columns",
            )
            .groupby(by="source", as_index=False)
            .any()
        )
        labels = node_label.loc[data["source"]]
        return pandas2torch(labels[self.label])


class NodeProbToPositive(Regression):
    def __init__(
        self,
        edge_label,
        positive_tag=1,
        taskName="incomming-positive-estimation",
        **kwargs,
    ):
        self.edge_label = edge_label
        self.positive_tag = positive_tag
        super().__init__(taskName, activation=lambda x: torch.clamp(x, 0, 1), **kwargs)

    @property
    def required_features(self):
        return {self.edge_label}

    def compute_labels(self, data, predictive_rate=0.001):
        df = data[["source", "timestamp", self.edge_label]]
        posDf = df.query(f"{self.edge_label} == {self.positive_tag}")
        posDf = posDf.rename(columns={"timestamp": "time_positive"})
        df = df.merge(posDf[["source", "time_positive"]], on="source", how="outer")
        df["will_positive"] = np.exp(
            predictive_rate * (df["timestamp"] - df["time_positive"])
        )
        df = df.fillna(0)
        Ysrc = pandas2torch(df["will_positive"])
        Ydst = torch.zeros_like(Ysrc)
        return torch.stack((Ysrc, Ydst), dim=1)
