import numpy as np
import torch
from tqdm import tqdm

import importlib
import sys
import os

from src.model.loss_utils import get_loss, round_loss
from src.utils.logger import get_logger

try:
    importlib.reload(sys.modules["src.model.loss_utils"])
except KeyError:
    pass


def train_INN(
    model,
    optimizer,
    losses,
    kwargs_data,
    kwargs_loss,
    save_model_path,
    start_epoch=0,
    num_epochs=1,
    print_info=True,
    log_dir=None,
    continue_log_path=None,
    conditions=None,
    use_counts=False,
):

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

    dataloader = kwargs_data["dataloader"]
    N_batches = len(dataloader)
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

    print(f"Sigma noise: {kwargs_loss['sigma_noise']}")
    sigma_noise = kwargs_loss["sigma_noise"]

    if logging:
        if start_epoch > 0:
            logger.info("")
            logger.info(f"Resuming training from epoch {start_epoch} for {num_epochs} epochs.")
            logger.info(f"Using raw counts: {use_counts}")
        else:
            logger.info(
                f"Starting training for {num_epochs} epochs with batch size {dataloader.batch_size}."
            )
            logger.info("")
            logger.info(f"Setup:")
            logger.info("---------------------------------------------------------")
            logger.info(f"Conditioning on: {conditions}")
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

    for epoch in info_func(range(start_epoch, start_epoch + num_epochs)):
        for i_batch, (x, c) in enumerate(dataloader):
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
            # x = x - data_mean  # / data_std

            # Noise data
            x = x + torch.randn_like(x) * sigma_noise
            any_all_zeros = any(not torch.any(t) for t in x)
            if any_all_zeros:
                print("Warning: No zero values in the data after adding noise.")
            x.requires_grad = True

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
            loss, metrics = get_loss(
                model, x, kwargs_data, kwargs_loss, c=c, metrics_last=metrics_last
            )

            # assert error if loss is nan
            if torch.isnan(loss):
                pass
                raise ValueError("Loss is NaN. Stopping training.")

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(f.parameters(), 10)
            losses["epoch"] = np.append(losses["epoch"], epoch)
            losses["loss"] = np.append(losses["loss"], loss.cpu().detach().numpy())
            losses["z2"] = np.append(losses["z2"], metrics["z2"].mean().cpu().detach().numpy())
            losses["NLL"] = np.append(losses["NLL"], metrics["NLL"].cpu().detach().numpy())
            losses["MTC"] = np.append(losses["MTC"], metrics["MTC"].cpu().detach().numpy())
            # H_i is a vector not scalar
            losses["H_i"] = np.append(losses["H_i"], metrics["H_i"].cpu().detach().numpy())
            losses["H_core"] = np.append(losses["H_core"], metrics["H_core"].cpu().detach().numpy())
            losses["H_detail"] = np.append(
                losses["H_detail"], metrics["H_detail"].cpu().detach().numpy()
            )
            losses["MI_core_detail"] = np.append(
                losses["MI_core_detail"],
                metrics["MI_core_detail"].cpu().detach().numpy(),
            )
            losses["L2_rec"] = np.append(losses["L2_rec"], metrics["L2_rec"].cpu().detach().numpy())
            if "sing_vals" in metrics:
                if "sing_vals" not in losses:
                    losses["sing_vals"] = np.array([])
                losses["sing_vals"] = np.append(
                    losses["sing_vals"], metrics["sing_vals"].cpu().detach().numpy()
                )
            optimizer.step()

        if logging:

            losses_regularization = ""
            # 'MTC:', round_loss(losses['MTC'][-N_batches:].mean()),
            #'L2_rec:', round_loss(losses['L2_rec'][-N_batches:].mean())
            # add regularization losses if they are used
            if kwargs_loss["use_MER"]:
                if kwargs_loss.get("use_align", False):
                    losses_regularization += (
                        "MI_core_detail: "
                        + str(round_loss(losses["MI_core_detail"][-N_batches:].mean()))
                        + " "
                    )
                    losses_regularization += (
                        "H_core: " + str(round_loss(losses["H_core"][-N_batches:].mean())) + " "
                    )
                    losses_regularization += (
                        "H_detail: " + str(round_loss(losses["H_detail"][-N_batches:].mean())) + " "
                    )
                else:
                    losses_regularization += (
                        "MTC: " + str(round_loss(losses["MTC"][-N_batches:].mean())) + " "
                    )
                    if kwargs_data["N_dim"] <= 3:
                        # plot H_i per dimension
                        for dim in range(kwargs_data["N_dim"]):
                            losses_regularization += (
                                "H_"
                                + str(dim)
                                + ": "
                                + str(
                                    round_loss(
                                        losses["H_i"]
                                        .reshape(-1, kwargs_data["N_dim"])[-N_batches:, dim]
                                        .mean()
                                    )
                                )
                                + " "
                            )
            if kwargs_loss.get("use_rec", False):
                losses_regularization += (
                    "L2_rec: " + str(round_loss(losses["L2_rec"][-N_batches:].mean())) + " "
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

        if epoch % 50 == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics_loss": losses,
                "log_path": log_path,
                "counts": use_counts,
                "conditions": conditions,
            }
            torch.save(checkpoint, save_model_path)
            logger.info("Saving checkpoint to " + save_model_path)

    logger.info("Finished training. Saving model to " + save_model_path)

    return model, optimizer, losses, log_path
