import os
import numpy as np
import torch
import argparse

from src.model.data_utils import (
    load_data,
    prepare_data,
    get_condition_shapes,
)
from src.model.training_utils import train_INN
from src.model.build_model import get_INN, ModelConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train INN model")

    parser.add_argument("--dataset", type=str, default="kang")
    parser.add_argument("--top_genes", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigma_noise", type=float, default=1.1)
    parser.add_argument("--lam_MTC", type=float, default=0.1)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--model_path", type=str, default="/home/jhoefer/sandbox/models/EOFlow")
    parser.add_argument("--log_root", type=str, default="/home/jhoefer/sandbox/results/logs")
    parser.add_argument("--model_prefix", type=str, default=None)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--use_counts", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # Data loading and preprocessing
    adata, _, _ = load_data(
        args.dataset, args.top_genes, log_transform=True, cell_types=["CD4 Memory"]
    )
    dataset, dataloader = prepare_data(
        adata, args.batch_size, device, dtype, counts=args.use_counts, label_key=args.conditions
    )

    D_dim = dataset.X.shape[1]
    N_dim = D_dim

    # Get condition shapes for model initialization
    condition_shapes = None
    if args.conditions is not None:
        condition_shapes = get_condition_shapes(adata, args.conditions)

    top_genes = min(args.top_genes, D_dim)

    # Paths
    # Define default name for model and output dir based on hyperparams
    default_name = "MTC_" + str(args.lam_MTC) + "_sigma_" + str(args.sigma_noise)

    # Model will be saved to group share
    model_path = os.path.join(
        "/g/stegle/jhoefer/models/EOFlow", args.dataset, "top_" + str(top_genes)
    )

    # Output (logs, plots) will be saved to local sandbox
    output_dir_path = os.path.join(
        "/home/jhoefer/sandbox/results", args.dataset, "top_" + str(top_genes)
    )

    # If model_prefix is provided, append it to the model name
    if args.model_prefix is not None:
        model_name = args.model_prefix + "_" + default_name + "_model.pt"
        output_dir_name = args.model_prefix + "_" + default_name
    else:
        model_name = default_name + "_model.pt"
        output_dir_name = default_name

    # If use_counts is True, append "_counts" to model name and output dir name
    if args.use_counts:
        model_path += "_counts"
        output_dir_path += "_counts"

    if args.conditions is not None:
        model_path += "_cond"
        output_dir_path += "_cond"

    model_path += "_small_batch_ablation"
    output_dir_path += "_small_batch_ablation"

    log_dir = os.path.join(output_dir_path, "logs", output_dir_name)

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Training setup
    kwargs_data = {
        "device": device,
        "dtype": dtype,
        "N_dim": N_dim,
        "D_dim": D_dim,
        "dataloader": dataloader,
        "data_mean": dataset.X.mean(dim=0),
        "data_std": dataset.X.std(dim=0),
        "sigma_noise": args.sigma_noise,
        "sigma_inflate": args.sigma_noise,
    }

    kwargs_loss = {
        "use_NLL": True,
        "use_MER": False,
        "mode_MER": "unbiased",  #'full', #'unbiased'
        "lam_MTC": args.lam_MTC,
        "lam_ME_i": list(np.zeros(D_dim)),
        # 'use_align': False,
        # 'dims_align': D_dim - 1,
        # 'lam_align': 1, #0.1,
        # 'lam_disent': 0, #0.1,
        "use_rec": False,
        "dims_rec": D_dim - 1,
        "lam_rec": 100,  #
        "sigma_noise": args.sigma_noise,
    }

    metrics_loss = {
        "epoch": [],
        "loss": [],
        "z2": [],
        "NLL": [],
        "MTC": [],
        "H_i": [],
        "H_core": [],
        "H_detail": [],
        "MI_core_detail": [],
        "L2_rec": [],
    }

    start_epoch = 0
    log_path = None
    if os.path.exists(os.path.join(model_path, model_name)) and args.checkpoint:
        checkpoint = torch.load(
            os.path.join(model_path, model_name),
            map_location=device,
        )

        model_config = checkpoint["model_config"]
        flow, optimizer_flow = get_INN(config=model_config)
        flow = flow.to(device)
        flow.load_state_dict(checkpoint["model_state_dict"])
        optimizer_flow.load_state_dict(checkpoint["optimizer_state_dict"])

        metrics_loss = checkpoint["metrics_loss"]
        log_path = checkpoint["log_path"]
        start_epoch = checkpoint["epoch"] + 1

    else:
        model_config = ModelConfig(
            N_dim=N_dim,
            condition_shapes=condition_shapes,
            ch_hidden=2048,
            N_blocks=12,
            lr=args.lr,
            warmup_steps=2 * len(dataloader),
            pre_normalize=False,
            normalize=True,
        )

        # Model initialization
        flow, optimizer_flow = get_INN(config=model_config)
        flow = flow.to(device)
        print(model_config)

    NUM_EPOCHS = max(1, np.ceil(args.epochs * args.batch_size / adata.X.shape[0]).astype(int))
    flow, optimizer_flow, metrics_loss, log_path = train_INN(
        flow,
        optimizer_flow,
        metrics_loss,
        kwargs_data,
        kwargs_loss,
        model_config=model_config,
        num_epochs=NUM_EPOCHS,
        start_epoch=start_epoch,
        print_info=True,
        log_dir=log_dir,
        continue_log_path=log_path,
        save_model_path=os.path.join(model_path, model_name),
        conditions=args.conditions,
        use_counts=args.use_counts,
    )


if __name__ == "__main__":
    main()
