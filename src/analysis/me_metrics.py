import os
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
from tqdm import tqdm

from src.utils.utils import print_mem
from src.model.loss_utils import get_NLL_z


def get_jacobian(
    kwargs_data,
    model,
    condition_shapes=None,
    compute_inverse=False,
    N_samples=1000,
    print_info=True,
    z_input=None,
    x_input=None,
    subtract_mean=False,
):
    N_dim = kwargs_data["N_dim"]
    device = kwargs_data["device"]

    data_mean = kwargs_data.get("data_mean", torch.zeros(N_dim)).unsqueeze(0).to(device)
    data_std = kwargs_data.get("data_std", torch.ones(N_dim)).unsqueeze(0).to(device)

    if condition_shapes is not None:
        shape = condition_shapes[0]
        cond = torch.randint(0, shape, (N_samples,)).to(device)
        cond = [nn.functional.one_hot(cond.to(device=device, dtype=int), num_classes=shape)]
    else:
        cond = None

    if z_input is not None:
        z = z_input
        N_samples = z.shape[0]
    elif x_input is not None:
        x = x_input
        N_samples = x.shape[0]
        if subtract_mean:
            x = x - data_mean
        # x = x - data_mean / data_std
        z, _ = model(x, c=cond, rev=False)
        z = z.detach().requires_grad_(True)
    else:
        z = torch.randn(N_samples, N_dim).to(device)
        z.requires_grad = True

    if print_info:  # use tqdm
        print_update_function = lambda x: tqdm(x)
    else:
        print_update_function = lambda x: x

    if print_info:
        print("Computing Jacobian with backward autodiff")
        if compute_inverse:
            print("Computing Jacobian of encoder instead of decoder and then invert")

    if not compute_inverse:
        print_mem("before forward")
        x, ljd_dec = model(z, c=cond, rev=True)
        print_mem("after forward")
    else:
        x, _ = model(z, c=cond, rev=True)
        x = x.detach()
        x.requires_grad = True
        z, ljd_enc = model(x, c=cond, rev=False)

        x, z = z, x

    jac_dec = torch.zeros((N_samples, N_dim, N_dim)).to(device=device)

    x = x.reshape(-1, N_dim)

    for i in print_update_function(range(N_dim)):  # loop through all dims of x
        x_grad = grad((x).reshape(N_samples, -1)[:, i].sum(), z, create_graph=True)[0]  #
        jac_dec[:, :, i] = x_grad.detach().to(device=device)
        del x_grad

    print_mem(f"after grad")
    if compute_inverse:
        x, z = z, x
        ljd_dec = -ljd_enc
        jac_dec = torch.linalg.inv(jac_dec)

    return jac_dec.detach(), ljd_dec.detach(), z.detach(), x.detach()


def get_MPMI(
    kwargs_data,
    jac_dec_1,
    jac_dec_2,
    max_dim=None,
    dtype=torch.float64,
    device="cpu",
    take_mean=False,
    print_info=False,
):
    N_samples = jac_dec_1.shape[0]
    N_dim = kwargs_data["N_dim"]

    # conditional = kwargs_data['conditional']
    if max_dim == None:
        max_dim = [N_dim, N_dim]
    elif isinstance(max_dim, int):
        max_dim = [max_dim, max_dim]
    else:
        assert (
            isinstance(max_dim, (list, tuple)) and len(max_dim) == 2
        ), "max_dim must be a list or tuple of two integers"

    # both jacobians must have the same number of samples and data dimensions, latent dimensions can differ!
    assert jac_dec_1.shape[0] == jac_dec_2.shape[0]
    assert jac_dec_1.shape[2] == jac_dec_2.shape[2]

    jac_dec_1 = jac_dec_1.to(device=device).to(dtype=dtype)
    jac_dec_2 = jac_dec_2.to(device=device).to(dtype=dtype)

    symmetric = False
    if jac_dec_1.shape == jac_dec_2.shape:
        if (jac_dec_1 == jac_dec_2).all():
            symmetric = True
            assert (
                max_dim[0] == max_dim[1]
            ), "For symmetric MPMI, max_dim must be the same for both variables."

    MPMI = torch.zeros((N_samples, max_dim[0], max_dim[1]), dtype=dtype).to(device=device)

    # precompute jac_det_i
    ljd_1_i = torch.log(torch.sum(jac_dec_1[:, :, :] ** 2, -1))
    ljd_2_i = torch.log(torch.sum(jac_dec_2[:, :, :] ** 2, -1))

    update_func = lambda x: x
    if print_info:
        update_func = lambda x: tqdm(x)
        print(f"Computing MPMI matrix")

    for i in update_func(range(max_dim[0])):
        second_range = range(max_dim[1]) if not symmetric else range(i + 1, max_dim[0])
        idx = torch.tensor([i, 0])
        for j in second_range:
            idx = torch.tensor([i, j])  # .to(device='cuda')

            submatrix = torch.cat(
                (
                    jac_dec_1[:, idx[0], :].unsqueeze(1),
                    jac_dec_2[:, idx[1], :].unsqueeze(1),
                ),
                1,
            )

            GTG = torch.einsum("xij, xkj -> xik", submatrix, submatrix)

            ljd_12_ij_temp = torch.log(GTG[:, 0, 0] * GTG[:, 1, 1] - GTG[:, 0, 1] * GTG[:, 1, 0])

            ljd_1_i_temp = ljd_1_i[:, idx[0]]
            ljd_2_i_temp = ljd_2_i[:, idx[1]]

            MPMI[:, i, j] = (
                1
                / 2
                * (ljd_1_i_temp + ljd_2_i_temp - ljd_12_ij_temp).to(device=device).to(dtype=dtype)
            )
    if take_mean:
        MPMI = MPMI.mean(0)
    return MPMI.to(device=device).to(dtype=dtype)


def get_manifold_entropy(
    kwargs_data, jac_dec, ljd_dec, z_dim=None, z=None, NLL_prior=None, print_info=True
):
    N_dim = kwargs_data["N_dim"]
    device = kwargs_data["device"]

    if z_dim is not None:
        N_dim = z_dim

    assert not (z is None and NLL_prior is None), "Either z or NLL_prior must be provided"
    assert not (
        z is not None and NLL_prior is not None
    ), "Only one of z or NLL_prior must be provided"

    data_std = kwargs_data.get("data_std", torch.ones(N_dim)).unsqueeze(0).to(device)

    # ensure data_std has N_dim dimensions
    if data_std.shape[-1] == 1:
        data_std = data_std.repeat(1, N_dim)  # [1 x N_dim]
    if z is not None:
        NLL_z = get_NLL_z(kwargs_data, z=z)
    else:
        NLL_z = NLL_prior

    H = (
        NLL_z.sum(-1) + ljd_dec + torch.log(data_std).sum(-1)
    ).mean()  # we add log(data_std) to the entropy as we compute the entropy in the original data space
    H_i = torch.zeros((N_dim), dtype=torch.float32)

    update_func = lambda x: x
    if print_info:
        update_func = lambda x: tqdm(x)

    for i in update_func(range(N_dim)):
        H_i[i] = (
            torch.mean(1 / 2 * torch.log(torch.sum(jac_dec[:, i, :] ** 2, -1)), -1)
            + NLL_z[:, i].mean()
            + torch.log(data_std[0, i])
        )

    latent_sort = torch.argsort(H_i, descending=True)

    return H_i, H, latent_sort
