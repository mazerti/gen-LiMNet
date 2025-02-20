import ignite
import ignite.contrib.metrics
import torch


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
