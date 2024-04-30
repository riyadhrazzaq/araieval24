import logging
import os
from pathlib import Path
from statistics import mean

import numpy as np
import optuna
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, get_scheduler

import config as cfg
from datautil import *
from modelutil import *

sys.path.append(".")

import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
from transformers.modeling_outputs import TokenClassifierOutput

from config import labels as LABELS

from task1.scorer.task1 import FLC_score_to_string

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using device: {}".format(device))


def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    path = Path(checkpoint_dir) / "model_best.pt"

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def validation(model, valid_dl, max_step):
    print('\tValidating...')
    model.eval()
    loss_across_batches = []
    metric_across_batches = dict(
        zip(
            [
                "f1",
            ]
            + LABELS,
            [
                0.0,
            ]
            * 24,
        )
    )

    with torch.no_grad():
        for step_no, batch in enumerate(valid_dl):
            for key in batch["tensors"].keys():
                batch["tensors"][key] = batch["tensors"][key].to(device)

            # calculate loss on valid
            loss, logits = step(model, batch["tensors"])
            loss_across_batches.append(loss.item())

            metrics = compute_metrics(batch, logits)
            # sum up
            for k, v in metrics.items():
                metric_across_batches[k] += v

            if step_no == max_step:
                break

            if step_no % cfg.valid_step_interval == 0:
                print(f"\tStep: {step_no}/{len(valid_dl)}, Loss: {loss.item()}")

        # we need the mean
        for k, v in metric_across_batches.items():
            metric_across_batches[k] /= step_no + 1

    return {"loss": mean(loss_across_batches), **metric_across_batches}


def step(model, batch):
    output: TokenClassifierOutput = model(
        input_ids=batch["input_ids"],
        token_type_ids=batch["token_type_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"] if "labels" in batch else None,
    )

    return output["loss"] if "loss" in output else None, output["logits"]


def fit(
        model: nn.Module,
        optimizer,
        train_dl: torch.utils.data.DataLoader,
        valid_dl: torch.utils.data.DataLoader,
        config: dict,
        checkpoint_dir="./checkpoint",
        max_step=-1,
        epoch=0,
        lr_scheduler=None,
        trial=None,
        disable_tqdm=False
):
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    model.to(device)
    best_f1 = float("-inf")

    _training_history = {"train/loss": [], "valid/loss": [], "valid/f1": []}
    _validation_history = {f"valid/{k}": [] for k in LABELS}
    history = {**_training_history, **_validation_history}

    for epoch in range(epoch + 1, epoch + config["max_epoch"] + 1):
        model.train()
        loss_across_batches = []

        for step_no, batch in enumerate(train_dl):
            # move to gpu
            for key in batch["tensors"].keys():
                batch["tensors"][key] = batch["tensors"][key].to(device)

            # reset grads
            optimizer.zero_grad()

            # step forward
            loss, logits = step(model, batch["tensors"])

            # step backward
            loss.backward()

            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            loss_across_batches.append(loss.item())

            # skip training on the entire training dataset
            # useful during debugging
            if step_no == max_step:
                break

            if step_no % cfg.train_step_interval == 0:
                print(f"\tStep: {step_no}/{len(train_dl)}, Loss: {loss.item()}")

        validation_metrics = validation(model, valid_dl, max_step)

        history["train/loss"].append(mean(loss_across_batches))
        for k, v in validation_metrics.items():
            history[f"valid/{k}"].append(v)

        if validation_metrics["f1"] > best_f1:
            best_f1 = validation_metrics["f1"]
            save_checkpoint(model, optimizer, epoch, checkpoint_dir)
            print("\n🎉 best f1 reached, saved a checkpoint.")

        if trial:
            trial.report(validation_metrics["f1"], epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        log(epoch, history)

    return history


def log(epoch, history):
    print(
        f"\nEpoch: {epoch},\tTrain Loss: {history['train/loss'][-1]},\tVal Loss: {history['valid/loss'][-1]}\tVal F1: {history['valid/f1'][-1]}"
    )


def load_checkpoint(model, checkpoint_path, optimizer=None, lr_scheduler=None):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        epoch = checkpoint["epoch"]

        logger.info(f"🎉 Loaded existing model. Epoch: {checkpoint['epoch']}")
        return model, optimizer, lr_scheduler, epoch

    else:
        raise Exception("No checkpoint found in the provided path")


def save_history(history, history_dir, save_graph=True):
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)

    with open(f"{history_dir}/history.json", "w") as f:
        json.dump(history, f)

    if save_graph:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(history["train/loss"], label="train/loss")
        ax.plot(history["valid/loss"], label="valid/loss")
        ax.legend()
        plt.savefig(f"{history_dir}/history_loss.png")

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(history["valid/f1"], label="valid/f1")
        ax.legend()
        plt.savefig(f"{history_dir}/history_f1.png")


def compute_metrics(batch, logits):
    gold = format_for_output(batch["raws"])
    hypotheses: List[Dict] = []
    logits = logits.transpose(1, 2)
    assert logits.size(1) == len(LABELS), "expects the label in dim=1"

    for i in range(logits.size(0)):
        hypothesis = parse_label_encoding(
            None, batch["encodings"][i], logits[i], LABELS
        )
        hypotheses.append({"id": batch["raws"][i]["id"], "labels": hypothesis})

    hypotheses = format_for_output(hypotheses)

    res_for_screen, f1, f1_per_label = FLC_score_to_string(
        gold, hypotheses, per_label=True
    )

    metrics = {"f1": f1, **f1_per_label}

    return metrics


def train(parameters, trial=None, disable_tqdm=False):
    logger.info(f"Training Parameters: {parameters}")

    tokenizer = BertTokenizerFast.from_pretrained(cfg.model_name)
    train_ds = DatasetFromJson(parameters["training_file"], tokenizer, cfg.max_length)
    val_ds = DatasetFromJson(parameters["validation_file"], tokenizer, cfg.max_length)
    train_dl = DataLoader(
        train_ds,
        batch_size=parameters["batch_size"],
        collate_fn=CollateFn(tokenizer, return_raw=False),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=parameters["batch_size"],
        collate_fn=CollateFn(tokenizer=tokenizer, return_raw=True),
    )

    model = model_init(parameters["model_name"], not parameters["no_pretrain"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=parameters["lr"], weight_decay=parameters["weight_decay"]
    )
    num_training_steps = parameters['max_epoch'] * len(train_dl)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=parameters['warmup_steps'],
        num_training_steps=num_training_steps
    )
    checkpoint_dir = f"{cfg.checkpoint_dir}/{parameters['experiment_name']}"
    history_dir = f"{checkpoint_dir}/history"
    history = fit(
        model,
        optimizer,
        train_dl,
        val_dl,
        parameters,
        checkpoint_dir,
        max_step=parameters["max_step"],
        epoch=0,
        lr_scheduler=lr_scheduler,
        trial=trial,
        disable_tqdm=disable_tqdm
    )
    save_history(history, history_dir, save_graph=True)
    return history
