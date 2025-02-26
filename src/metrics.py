import ignite
import ignite.contrib.metrics
import ignite.metrics
import torch
from sklearn import metrics as m
import matplotlib.pyplot as plt

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
