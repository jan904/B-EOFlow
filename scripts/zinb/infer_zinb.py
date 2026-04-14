import os
import numpy as np
import torch
import argparse


from src.model.data_utils import (
    load_data,
    prepare_data,
)
from src.utils.zinb import inferr_zinb_params


def parse_args():
    parser = argparse.ArgumentParser(description="Infer ZINB parameters")

    parser.add_argument("--N_genes", type=int, default=500)
    parser.add_argument("--gene", type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    # Data loading
    adata, _, _ = load_data("kang", args.N_genes)

    inferr_zinb_params(adata, n=args.N_genes, gene=args.gene)


if __name__ == "__main__":
    main()
