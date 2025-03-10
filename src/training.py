import argparse
import ignite
import json
import os
import ignite.engine
from ignite.utils import convert_tensor
import torch.utils.data.dataloader
import outputs as o
import pickle
import torch
import util
from custom_engine import create_trainer, create_evaluator


def lossFunction(tasks):
    def compute_loss(state_output):
        return sum(task.compute_loss(state_output) for task in tasks)

    return compute_loss


def prepare_batch(batch, device, non_blocking):
    features, labels, masks = batch
    return (
        convert_tensor(features, device=device, non_blocking=non_blocking),
        convert_tensor(labels, device=device, non_blocking=non_blocking),
    )


def run_training(conf, baseFolder, device, resume):
    model = conf["model"]
    outputs = conf["outputs"]
    mixedPrecision = conf.get("mixedPrecision", False)
    tasks = o.get_tasks(outputs, device)

    info, data = load_data(conf, baseFolder, resume, model, tasks)

    build_model(device, model, outputs, mixedPrecision, info)

    loss = lossFunction(tasks)
    scaler = torch.amp.GradScaler("cuda")
    optimizer = conf["optimizer"](model.parameters())

    metrics = gather_metrics(tasks, loss)

    trainer = create_trainer(
        model,
        optimizer,
        lambda Ypred, Ytrue: loss((Ypred, Ytrue)),
        device=device,
        prepare_batch=prepare_batch,
        amp_mode="amp" if mixedPrecision else None,
        scaler=scaler if mixedPrecision else False,
    )
    evaluator = create_evaluator(
        model,
        metrics,
        device=device,
        amp_mode="amp" if mixedPrecision else None,
        prepare_batch=prepare_batch,
    )

    # Define tracking events/statistics (checkpoints, losses, etc.)
    batchLosses = []

    @trainer.on(ignite.engine.Events.ITERATION_COMPLETED)
    def collect_batch_losses(trainer):
        batchLosses.append(trainer.state.output)

    trainer.state_dict_user_keys.append("metrics_history")

    @trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
    def do_validation(trainer):
        nonlocal batchLosses
        evaluator.run(data)
        metrics = dict(evaluator.state.metrics)
        metrics["epoch_time"] = trainer.state.times["EPOCH_COMPLETED"]
        if len(batchLosses) > 0:
            metrics["training-loss"] = sum(batchLosses)
            batchLosses = []
        else:
            metrics["training-loss"] = float("nan")
        try:
            trainer.state.metrics_history.append(metrics)
        except AttributeError:
            trainer.state.metrics_history = [metrics]

        # can do better
        with open(f"{baseFolder}/metrics.json", "w") as f:
            json.dump(trainer.state.metrics_history, f, indent=4)

    util.tqdmHandler(trainer, desc="epochs", epochs=True)
    util.tqdmHandler(trainer, desc="training", leave=False)
    util.tqdmHandler(evaluator, desc="validation", leave=False)

    to_save = {
        "model": model,
        "optimizer": optimizer,
        "trainer": trainer,
        "scaler": scaler,
    }
    checkpointer = ignite.handlers.Checkpoint(
        to_save,
        save_handler=ignite.handlers.DiskSaver(f"{baseFolder}", require_empty=False),
        n_saved=2,
        global_step_transform=lambda *_: trainer.state.epoch,
        include_self=True,
    )
    trainer.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED, checkpointer)

    if resume:
        to_load = dict(to_save)
        to_load["checkpointer"] = checkpointer
        latest_checkpoint = util.latestCheckpoint(baseFolder)
        ignite.handlers.Checkpoint.load_objects(to_load, torch.load(latest_checkpoint))
        assert (
            conf["epochs"] > trainer.state.epoch
        ), f'Training already reached the expected number of epochs! {conf["epochs"]} {trainer.state.epoch}'
    else:
        do_validation(trainer)

    # Run training
    trainer.run(data, max_epochs=conf["epochs"])

    # Store results
    with open(f"{baseFolder}/metrics.json", "w") as f:
        json.dump(trainer.state.metrics_history, f, indent=4)
    torch.save(
        model.state_dict(),
        f"{baseFolder}/model.pt",
        pickle_protocol=pickle.HIGHEST_PROTOCOL,
    )


def gather_metrics(tasks, loss):
    metrics = {
        metricKey: metricFunc
        for task in tasks
        for metricKey, metricFunc in task.metrics.items()
    }
    metrics["validation-loss"] = ignite.metrics.Average(loss)
    return metrics


def build_model(device, model, outputs, mixedPrecision, info):
    model.build(
        info,
        o.get_decoders(outputs),
        torch.float16 if mixedPrecision else torch.float32,
    )
    model.to(device)
    model.optimize()


def load_data(conf, baseFolder, resume, model, tasks):
    dataset, info = conf["dataLoader"].loadData(
        tasks, baseFolder, resume, conf["trainRatio"]
    )
    store_data_info(baseFolder, info)
    data = to_dataloader(dataset, batchSize=conf["batchSize"], model=model)
    return info, data


def to_dataloader(dataset, batchSize, model):
    data_loader_args = dict(collate_fn=model.makeBatch, num_workers=12, pin_memory=True)
    data = torch.utils.data.DataLoader(
        dataset, batch_size=batchSize, **data_loader_args
    )
    return data


def store_data_info(baseFolder, info):
    with open(f"{baseFolder}/info.json", "w") as f:
        json.dump(info, f, indent=4)


def _parseArgs():
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument(
        "run_name",
        type=str,
        help="the results will be stored in folder runs/<run_name>",
    )
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        help="path of the config file with the training parameters; defaults to runs/<run_name>/conf.py",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        nargs="?",
        const=0,
        help="whether to use a GPU; a 0-based index can optionally specify which GPU to use",
    )
    parser.add_argument(
        "--cpus", type=int, help="max number of cores to use when training on CPU"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume training from the latest checkpoint",
    )
    return parser.parse_args()


def create_clean_base_folder(args, baseFolder, isConfigFileGiven):
    if not os.path.exists(baseFolder):
        os.makedirs(baseFolder)
    else:
        for filename in os.listdir(baseFolder):
            if filename == "conf.py" and isConfigFileGiven:
                continue
            if filename.startswith("checkpoint") and args.resume:
                continue
            if filename == "seed.txt" and args.resume:
                continue
            if filename == "__pycache__":
                continue
            os.remove(f"{baseFolder}/{filename}")


def copy_config_file(targetConfFile, inputConfFile, isConfigFileGiven):
    if not isConfigFileGiven:
        with open(targetConfFile, "w") as target, open(inputConfFile, "r") as input:
            target.write(input.read())


def main():
    args = _parseArgs()

    baseFolder = f"runs/{args.run_name}"
    targetConfFile = f"{baseFolder}/conf.py"
    inputConfFile = args.config or targetConfFile

    if args.cpus is not None:
        torch.set_num_threads(args.cpus)

    create_clean_base_folder(args, baseFolder, isConfigFileGiven=args.config)
    copy_config_file(targetConfFile, inputConfFile, isConfigFileGiven=args.config)

    run_training(
        conf=util.loadConfigurationFile(inputConfFile).conf,
        baseFolder=baseFolder,
        device=torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cpu"),
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
