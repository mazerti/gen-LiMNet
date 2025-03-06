import torch
from ignite.engine.deterministic import DeterministicEngine
from ignite.engine.engine import Engine
from ignite.utils import convert_tensor


def _prepare_batch(batch, device=None, non_blocking: bool = False):
    """Prepare batch for training or evaluation: pass to a device with options."""
    x, y = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )


def supervised_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device=None,
    non_blocking: bool = False,
    prepare_batch=_prepare_batch,
    model_transform=lambda output: output,
    output_transform=lambda x, y, y_pred, loss: loss.item(),
    gradient_accumulation_steps: int = 1,
    model_fn=lambda model, x: model(x),
):
    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "Gradient_accumulation_steps must be strictly positive. "
            "No gradient accumulation if the value set to one (default)."
        )

    def update(engine: Engine, batch):
        if (engine.state.iteration - 1) % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        model.train()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        mask, _ = convert_tensor(batch[2], device=device, non_blocking=non_blocking)
        output = model_fn(model, x)
        y_pred = model_transform(output)
        y_pred = tuple((torch.masked_select(torch.squeeze(pred), mask) for pred in y_pred))
        y = tuple((torch.masked_select(torch.squeeze(lab), mask) for lab in y))
        loss = loss_fn(y_pred, y)
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()
        if engine.state.iteration % gradient_accumulation_steps == 0:
            optimizer.step()
        return output_transform(x, y, y_pred, loss * gradient_accumulation_steps)

    return update


def create_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device=None,
    non_blocking: bool = False,
    prepare_batch=_prepare_batch,
    model_transform=lambda output: output,
    output_transform=lambda x, y, y_pred, loss: loss.item(),
    deterministic: bool = False,
    amp_mode=None,
    scaler=False,
    gradient_accumulation_steps: int = 1,
    model_fn=lambda model, x: model(x),
):
    """No mixed precision for now."""
    device_type = device.type if isinstance(device, torch.device) else device
    on_tpu = "xla" in device_type if device_type is not None else False
    on_mps = "mps" in device_type if device_type is not None else False

    _update = supervised_training_step(
        model,
        optimizer,
        loss_fn,
        device,
        non_blocking,
        prepare_batch,
        model_transform,
        output_transform,
        gradient_accumulation_steps,
        model_fn,
    )

    trainer = Engine(_update) if not deterministic else DeterministicEngine(_update)

    return trainer


def supervised_evaluation_step(
    model: torch.nn.Module,
    device=None,
    non_blocking: bool = False,
    prepare_batch=_prepare_batch,
    model_transform=lambda output: output,
    output_transform=lambda x, y, y_pred: (y_pred, y),
    model_fn=lambda model, x: model(x),
):
    def evaluate_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            _, mask = convert_tensor(batch[2], device=device, non_blocking=non_blocking)
            output = model_fn(model, x)
            y_pred = model_transform(output)
            y_pred = tuple((torch.masked_select(torch.squeeze(pred), mask) for pred in y_pred))
            y = tuple((torch.masked_select(torch.squeeze(lab), mask) for lab in y))
            # It can happen that y is empty here. That is an issue.
            return output_transform(x, y, y_pred)

    return evaluate_step


def create_evaluator(
    model: torch.nn.Module,
    metrics=None,
    device=None,
    non_blocking: bool = False,
    prepare_batch=_prepare_batch,
    model_transform=lambda output: output,
    output_transform=lambda x, y, y_pred: (y_pred, y),
    amp_mode=None,
    model_fn=lambda model, x: model(x),
):
    device_type = device.type if isinstance(device, torch.device) else device
    on_tpu = "xla" in device_type if device_type is not None else False
    on_mps = "mps" in device_type if device_type is not None else False

    metrics = metrics or {}
    evaluate_step = supervised_evaluation_step(
        model,
        device,
        non_blocking=non_blocking,
        prepare_batch=prepare_batch,
        model_transform=model_transform,
        output_transform=output_transform,
        model_fn=model_fn,
    )

    evaluator = Engine(evaluate_step)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator
