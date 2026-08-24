import numpy as np
import torch
from tqdm import tqdm
from dataclasses import asdict

import importlib
import sys
import os
import time

from sklearn.cluster import KMeans

from src.model.INN.INN_losses import (
    get_loss,
    round_loss,
    DimensionSampler,
    append_losses,
    build_regularization_loss_string,
)
from src.utils.logger import get_logger

# imported lazily-safe: analysis imports model code, so keep this after the model imports
from src.analysis.ood_probe import format_probe


def train_INN(
    model,
    optimizer,
    losses,
    kwargs_data,
    kwargs_loss,
    save_model_path,
    model_config,
    val_metrics_loss=None,
    start_epoch=0,
    num_epochs=1,
    print_info=True,
    log_dir=None,
    continue_log_path=None,
    conditions=None,
    use_counts=False,
    save_on_validation=False,
    ood_probe=None,
    ood_probe_every=50,
):
    """`ood_probe`: an analysis.ood_probe.OODProbe for a *validation* combo held out
    alongside the reported test combo. Scored every `ood_probe_every` epochs and logged;
    the best-MMD state is written next to `save_model_path` as `*_best_ood.pt`, in
    addition to (not instead of) the usual checkpoint - the NLL keeps improving long
    after OOD prediction stops, so the last checkpoint is not the one you want for OOD,
    and both endpoints are worth keeping."""

    condition_type = model_config.condition_type
    train_means = model_config.trainable_means
    supervise_latent_meaning = model_config.supervise_latent_meaning
    lam_supervise = model_config.lam_supervise

    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    if log_dir is None:
        logging = False
        log_path = None
    else:
        print_info = False  # avoid duplicate prints if logging is enabled
        logging = True
        logger, log_path = get_logger(log_dir=log_dir, log_file=continue_log_path)

    device = kwargs_data["device"]
    dtype = kwargs_data["dtype"]

    train_dataloader = kwargs_data["train_dataloader"]
    N_batches = len(train_dataloader)

    data_mean = (
        kwargs_data.get("data_mean", torch.zeros(kwargs_data["N_dim"]))
        .to(device=device)
        .to(dtype=dtype)
        .unsqueeze(0)
    )
    data_std = (
        kwargs_data.get("data_std", torch.ones(kwargs_data["N_dim"]))
        .to(device=device)
        .to(dtype=dtype)
        .unsqueeze(0)
    )

    patience = 10
    patience_counter = 0

    test_dataloader = (
        kwargs_data["test_dataloader"] if kwargs_data.get("test_dataloader") is not None else None
    )
    N_batches_val = len(kwargs_data["test_dataloader"]) if test_dataloader is not None else 0
    val_losses = losses.copy() if val_metrics_loss is None else val_metrics_loss

    best_val_loss = float("inf")
    batch_val_losses = []
    mean_val_loss = float("inf")

    best_ood_mmd = float("inf")
    ood_history = []
    best_ood_path = save_model_path.replace(".pt", "_best_ood.pt")

    print(f"Sigma noise: {kwargs_loss['sigma_noise']}")
    sigma_noise = kwargs_loss["sigma_noise"]
    logger.info(model_config)
    if logging:
        if start_epoch > 0:
            logger.info("")
            logger.info(f"Resuming training from epoch {start_epoch} for {num_epochs} epochs.")
            logger.info(f"Using raw counts: {use_counts}")
        else:
            logger.info(
                f"Starting training for {num_epochs} epochs with batch size {train_dataloader.batch_size}."
            )
            logger.info("")
            logger.info(f"Setup:")
            logger.info("---------------------------------------------------------")
            logger.info(f"Conditioning on: {conditions}")
            logger.info(f"Condition type: {condition_type}")
            logger.info(f"Trainable means: {train_means}")
            logger.info(f"Learning rate means: {model_config.lr_means}")
            logger.info(f"Supervise latent meaning: {supervise_latent_meaning}")
            logger.info(f"Latents per condition: {model_config.latent_per_condition}")
            # resolved value, not the config's None-means-auto (see
            # INNs_model.default_means_seperation)
            logger.info(
                f"Means separation: {getattr(model, 'means_seperation', None)} "
                f"(config: {model_config.means_seperation})"
            )
            logger.info(f"trainable sigma: {model_config.trainable_sigma}")
            logger.info("---------------------------------------------------------")
            logger.info("")
            logger.info(f"Model Setup:")
            logger.info("---------------------------------------------------------")
            logger.info(f"N_dim: {model_config.N_dim}")
            logger.info(f"N_blocks: {model_config.N_blocks}")
            logger.info(f"Hidden channels: {model_config.ch_hidden}")
            logger.info(f"Coupling block type: {model_config.coupling_block_type}")
            logger.info(f"N_hidden_layers: {model_config.n_hidden_layers}")
            logger.info(f"Act func: {model_config.act_func}")
            logger.info(f"Pre-normalize: {model_config.pre_normalize}")
            logger.info(f"Pre-rotate: {model_config.pre_rotate}")
            logger.info(f"Normalize: {model_config.normalize}")
            logger.info("---------------------------------------------------------")
            logger.info("")
            logger.info(f"Data Kwargs:")
            logger.info("---------------------------------------------------------")
            logger.info(f"N_dim: {kwargs_data['N_dim']}")
            logger.info(f"D_dim: {kwargs_data['D_dim']}")
            logger.info(f"Sigma noise: {sigma_noise}")
            logger.info("---------------------------------------------------------")
            logger.info("")
            logger.info(f"Loss Kwargs:")
            logger.info("---------------------------------------------------------")
            logger.info(f"Use NLL loss: {kwargs_loss['use_NLL']}")
            logger.info(f"Use MER: {kwargs_loss['use_MER']}")
            logger.info(f"Mode MER: {kwargs_loss['mode_MER']}")
            logger.info(f"lam_MTC: {kwargs_loss['lam_MTC']}")
            logger.info(f"use_rec: {kwargs_loss['use_rec']}")
            if kwargs_loss["use_rec"]:
                logger.info(f"lam_rec: {kwargs_loss['lam_rec']}")
                logger.info(f"dims_rec: {kwargs_loss['dims_rec']}")
            logger.info("---------------------------------------------------------")
            logger.info("")
            logger.info("")

    info_func = lambda x: x
    if print_info:
        info_func = lambda x: tqdm(x)

    sampler = DimensionSampler(N_dim=kwargs_data["N_dim"], device=device)

    times = []
    for epoch in info_func(range(start_epoch, start_epoch + num_epochs)):
        for i_batch, (x, c) in enumerate(train_dataloader):
            start_ = time.time()
            optimizer.zero_grad()

            if conditions is not None and c != -1:
                c = [cond.to(device=device, dtype=dtype) for cond in c]
            else:
                c = None

            # To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
            # check if x is already a tensor
            if not isinstance(x, torch.Tensor) and not isinstance(x, torch.tensor):
                x = torch.tensor(x, device=device, dtype=dtype)
            else:
                x = x.clone().detach().to(device=device, dtype=dtype)

            # Normalizing data
            # x = x - data_mean / data_std

            # Noise data
            x = x + torch.randn_like(x) * sigma_noise
            any_all_zeros = any(not torch.any(t) for t in x)
            if any_all_zeros:
                print("Warning: No zero values in the data after adding noise.")
            x.requires_grad = True

            t1 = time.time() - start_
            start = time.time()

            # insert metrics from last update
            metrics_last = {}
            if len(losses["H_i"]) < kwargs_data["N_dim"]:
                metrics_last["H_i"] = (torch.zeros(kwargs_data["N_dim"]) * torch.inf).to(
                    device=device, dtype=dtype
                )
                # initialize with infinite values for masking
            else:
                metrics_last["H_i"] = torch.tensor(
                    losses["H_i"][-kwargs_data["N_dim"] :], device=device, dtype=dtype
                )

            t2 = time.time() - start
            start = time.time()

            loss, metrics = get_loss(
                model,
                x,
                kwargs_data,
                kwargs_loss,
                sampler,
                model_config,
                c=c,
                metrics_last=metrics_last,
            )
            t3 = time.time() - start
            start = time.time()

            # assert error if loss is nan
            if torch.isnan(loss):
                raise ValueError("Loss is NaN. Stopping training.")

            loss.backward()
            optimizer.step()

            t4 = time.time() - start
            start = time.time()

            # torch.nn.utils.clip_grad_norm_(f.parameters(), 10)
            losses = append_losses(losses, epoch, metrics, loss)

            end = time.time()
            times.append(end - start_)

        if logging:
            losses_regularization = build_regularization_loss_string(
                losses, kwargs_loss, kwargs_data, N_batches
            )

            logger.info(
                "Epoch "
                + str(epoch)
                + ": Loss: "
                + str(round_loss(losses["loss"][-N_batches:].mean()))
                + " | z2: "
                + str(round_loss(losses["z2"][-N_batches:].mean()))
                + " | NLL: "
                + str(round_loss(losses["NLL"][-N_batches:].mean()))
                + " | "
                + losses_regularization
            )

            t5 = time.time() - start
            full_time = time.time() - start_

            # print(
            #     f"Time for epoch {epoch}: {full_time:.4f} seconds (T1: {t1/full_time:.2f}, T2: {t2/full_time:.2f}, T3: {t3/full_time:.2f}, T4: {t4/full_time:.2f}, T5: {t5/full_time:.2f})"
            # )

        if epoch % 10 == 0 and test_dataloader is not None:
            model.eval()
            with torch.no_grad():
                for i_batch, (x_val, c_val) in enumerate(test_dataloader):
                    if conditions is not None and c_val != -1:
                        c_val = [cond.to(device=device, dtype=dtype) for cond in c_val]
                    else:
                        c_val = None

                    if not isinstance(x_val, torch.Tensor) and not isinstance(x_val, torch.tensor):
                        x_val = torch.tensor(x_val, device=device, dtype=dtype)
                    else:
                        x_val = x_val.clone().detach().to(device=device, dtype=dtype)

                    x_val = x_val + torch.randn_like(x_val) * sigma_noise
                    val_loss, val_metrics = get_loss(
                        model,
                        x_val,
                        kwargs_data,
                        kwargs_loss,
                        sampler,
                        model_config,
                        c=c_val,
                        metrics_last=metrics_last,
                    )

                    if torch.isnan(val_loss):
                        raise ValueError("Validation Loss is NaN. Stopping training.")

                    batch_val_losses.append(val_loss.cpu().detach().numpy())
                    val_losses = append_losses(val_losses, epoch, val_metrics, val_loss)

                if logging:
                    val_losses_regularization = build_regularization_loss_string(
                        val_losses, kwargs_loss, kwargs_data, N_batches_val
                    )

                mean_val_loss = np.mean(batch_val_losses)
                batch_val_losses = []

                logger.info(
                    "Validation: "
                    + ": Loss: "
                    + str(round_loss(val_losses["loss"][-N_batches_val:].mean()))
                    + " (Best: "
                    + str(np.round(best_val_loss, 4))
                    + ")"
                    + " | z2: "
                    + str(round_loss(val_losses["z2"][-N_batches_val:].mean()))
                    + " | NLL: "
                    + str(round_loss(val_losses["NLL"][-N_batches_val:].mean()))
                    + " | "
                    + val_losses_regularization
                )

        if condition_type == "mixture" and train_means == False:
            update_means_epoch(model, train_dataloader, device, momentum=1.0)

        if ood_probe is not None and epoch % ood_probe_every == 0:
            # schedule-free keeps the parameters at the train (y) iterate; the weights a
            # later `load` would see are the x iterate, so probing without this swap
            # scores a different model than the one being selected.
            schedule_free = hasattr(optimizer, "eval") and hasattr(optimizer, "train")
            if schedule_free:
                optimizer.eval()
            ood_metrics = ood_probe(model)
            if schedule_free:
                optimizer.train()

            ood_metrics["epoch"] = epoch
            ood_metrics["combo"] = ood_probe.label
            ood_history.append(ood_metrics)
            logger.info(format_probe(ood_metrics))

            if ood_metrics["mmd"] < best_ood_mmd:
                best_ood_mmd = ood_metrics["mmd"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "ood_metrics": ood_metrics,
                        "ood_history": ood_history,
                        "log_path": log_path,
                        "counts": use_counts,
                        "conditions": conditions,
                        "model_config": asdict(model_config),
                    },
                    best_ood_path,
                )
                logger.info(f"New best OOD MMD ({best_ood_mmd:.4f}); saved {best_ood_path}")

        if save_on_validation and test_dataloader is not None:
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                patience_counter = 0
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics_loss": losses,
                    "val_metrics_loss": val_losses,
                    "log_path": log_path,
                    "counts": use_counts,
                    "conditions": conditions,
                    "model_config": asdict(model_config),
                }
                torch.save(checkpoint, save_model_path)
                logger.info("Saving checkpoint to " + save_model_path)
            else:
                patience_counter += 1
                logger.info(
                    f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}"
                )
                if patience_counter >= patience:
                    logger.info("Early stopping due to no improvement in validation loss.")
                    break
        else:
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
            if epoch % 50 == 0:
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics_loss": losses,
                    "val_metrics_loss": val_losses,
                    "log_path": log_path,
                    "counts": use_counts,
                    "conditions": conditions,
                    "model_config": asdict(model_config),
                }
                torch.save(checkpoint, save_model_path)
                logger.info("Saving checkpoint to " + save_model_path)

        model.train()

    if not save_on_validation:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics_loss": losses,
            "val_metrics_loss": val_losses,
            "log_path": log_path,
            "counts": use_counts,
            "conditions": conditions,
            "model_config": asdict(model_config),
        }
        torch.save(checkpoint, save_model_path)
        logger.info("Finished training. Saving model to " + save_model_path)

    print(f"Average time per update: {np.mean(times):.4f} seconds")

    return model, optimizer, losses, val_losses, log_path


def update_means_epoch(model, dataloader, device, momentum=0.9):
    """EMA-update the mixture-prior means towards the per-combo empirical latent means.

    Writes into `model.means_active` - the actual Parameter - and not `model.means`.
    `means` is a property that returns `torch.cat([means_active, means_zero])` whenever
    `latent_per_condition` is set (i.e. means_dim < N_dim), so the old
    `model.means[mask] = ...` assigned into that freshly built temporary and the update
    was silently discarded: with `latent_per_condition` set this function was a no-op,
    and the means stayed at their initialization for the whole run.

    Only the first `means_dim` latent dimensions are parameters; the remaining dims are
    the structurally-zero `means_zero` buffer (shared N(0,1) prior across conditions by
    design), so the empirical mean is taken over the active dims only.
    """
    if getattr(model, "factorized", False):
        raise RuntimeError(
            "update_means_epoch does not apply to a factorized prior: its means are "
            "composed from per-condition factors, so there is no per-combo parameter to "
            "set from an empirical mean (and writing one would break the factorization "
            "that makes held-out combos predictable). Use trainable_means=True."
        )

    means = model.means_free  # (K, means_dim), the parameter itself
    means_dim = means.shape[1]

    cluster_sums = torch.zeros_like(means)  # (K, means_dim)
    counts = torch.zeros(means.shape[0], 1, device=device)  # (K, 1)

    model.eval()
    with torch.no_grad():
        for x, c in dataloader:
            x = x.to(device)
            c = [cond.to(device=device, dtype=torch.float32) for cond in c]

            z, _ = model(x, c=None, rev=False)

            counts += c[0].sum(dim=0, keepdim=True).T
            cluster_sums += c[0].T @ z[:, :means_dim]

        mask = counts.squeeze() > 0
        epoch_means = cluster_sums[mask] / counts[mask]
        means.data[mask] = (1 - momentum) * means.data[mask] + momentum * epoch_means
    model.train()
