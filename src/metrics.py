import ignite
import ignite.contrib.metrics
import ignite.metrics
import torch
from focal_loss.focal_loss import FocalLoss
from sklearn import metrics as m
import matplotlib.pyplot as plt
from functools import reduce
from operator import mul

from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


def accuracy(prepare_predictions, device):
    def output_transform(state_output):
        ypred, ytrue = prepare_predictions(state_output)
        return torch.round(torch.squeeze(ypred)), torch.squeeze(ytrue)

    return ignite.metrics.Accuracy(output_transform, device=device)


def auc(prepare_predictions, device, task_transform=None, **kwargs):
    def output_transform(state_output):
        ypred, ytrue = prepare_predictions(state_output)
        output = (torch.flatten(ypred), torch.flatten(ytrue))
        if task_transform is None:
            return output
        output = task_transform(output)
        return output

    return ignite.contrib.metrics.ROC_AUC(output_transform, kwargs)


def precision(prepare_predictions, device):
    def output_transform(state_output):
        ypred, ytrue = prepare_predictions(state_output)
        return torch.round(torch.squeeze(ypred)), torch.squeeze(ytrue)

    return ignite.metrics.Precision(output_transform, device=device)


def recall(prepare_predictions, device):
    def output_transform(state_output):
        ypred, ytrue = prepare_predictions(state_output)
        return torch.round(torch.squeeze(ypred)), torch.squeeze(ytrue)

    return ignite.metrics.Recall(output_transform, device=device)


def balancedAccuracy(prepare_predictions, device):
    pass


class DrawPrecisionRecallCurve(ignite.metrics.Metric):
    def __init__(self, output_transform=..., device=..., skip_unrolling=False):
        super(DrawPrecisionRecallCurve, self).__init__(
            output_transform, device, skip_unrolling
        )

    @reinit__is_reduced
    def reset(self):
        self.precisions, self.recall, self.thresholds = [], [], []
        super(DrawPrecisionRecallCurve, self).reset()

    @reinit__is_reduced
    def update(self, output):
        ypred, ytrue = output[0].detach(), output[1].detach()
        self.precisions, self.recall, self.thresholds = m.precision_recall_curve(
            y_true=ytrue.flatten().cpu().numpy(),
            y_score=ypred.flatten().cpu().numpy(),
        )
        print(
            f"\nprec: {self.precisions}\nrec: {self.recall}\nthres: {self.thresholds}\n"
        )

    @sync_all_reduce("_num_examples", "_num_correct:SUM")
    def compute(self):
        fig, ax = plt.subplots()
        ax.plot(self.recall, self.precisions, color="purple")
        ax.set_title("Precision-Recall Curve")
        ax.set_ylabel("Precision")
        ax.set_xlabel("Recall")
        fig.savefig("test-prec-recall-curve.png")
        plt.close(fig)
        return m.auc(self.recall, self.precisions)


def drawPrecisionRecallCurve(prepare_predictions, device):
    def output_transform(state_output):
        ypred, ytrue = prepare_predictions(state_output)
        return torch.squeeze(ypred), torch.squeeze(ytrue)

    return DrawPrecisionRecallCurve(output_transform, device=device)


def fbeta(beta):
    def _fbeta(prepare_predictions, device):
        def output_transform(state_output):
            ypred, ytrue = prepare_predictions(state_output)
            return torch.round(torch.squeeze(ypred)), torch.squeeze(ytrue)

        return ignite.metrics.Fbeta(
            beta, output_transform=output_transform, device=device
        )

    return _fbeta


f1 = fbeta(1)


# --- Losses ---


def loggedLoss(lossFn, threshold=0.5):
    def loss(output, target):
        shape = list(output.shape)
        size = reduce(mul, shape, 1)
        nb_classes = target.unique(return_counts=True)[1].tolist()
        nb_pred = int((output > threshold).sum())
        print(f"\nBatch size: {size}({'x'.join([str(x) for x in shape])})")
        print(
            f"YTrue: {",".join([f"{nb}({round(100*nb/size, 4)}%)" for nb in nb_classes])}"
        )
        print(
            f"Predicted (ypred > {threshold}): {nb_pred}({round(100*nb_pred/size, 4)}%)"
        )
        loss_value = lossFn(output, target)
        print(f"Computed Loss: {loss_value}")
        return loss_value

    return loss


def dynamicWeightedBCE():
    def loss(output, target):
        nb_classes = target.unique(return_counts=True)[1]
        if len(nb_classes) == 2:
            weight = nb_classes[0] / nb_classes[1]
        else:
            weight = None
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
        return loss_fn(output, target)

    return loss


def sigmoidFocalLoss(gamma=0.7):
    # m = torch.nn.Softmax(dim=-1)
    m = torch.nn.Sigmoid()
    focalLoss = FocalLoss(gamma)

    def loss(output, target):
        pred = m(output)
        return focalLoss(pred.flatten(), target.long())

    return loss
