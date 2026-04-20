import torch
import torch.nn as nn
from torch.autograd import grad
from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual
import numpy as np


def round_loss(loss, round=4):
    # check if loss is a list
    if isinstance(loss, list):
        return [round_loss(l, round) for l in loss]
    else:
        return np.round(loss.item(), round)


def get_NLL_z(kwargs_data, z=None):
    # Assume standard normal prior p(z)
    # Compute the negative log-likelihood (NLL) using samples z per dimension. Alternatively this is the entropy of p_i(z_i) if z is None
    N_dim = kwargs_data["N_dim"]

    if z is None:
        z = torch.ones(N_dim)
    return 1 / 2 * (z**2)  # + torch.tensor(1 / 2 * np.log(2 * np.pi))


def get_loss(model, x, kwargs_data, kwargs_loss, c=None, metrics_last=None):
    use_NLL = (
        kwargs_loss.get("use_NLL", True) * 1
    )  # use Negative Log-Likelihood-loss (Maximum Likelihood)
    use_MER = kwargs_loss["use_MER"]  # use Manifold Entropic Regularization
    mode_MER = kwargs_loss.get("mode_MER", "full")  #'unbiased'
    assert mode_MER in ["full", "unbiased"], "mode_MER not recognized!"

    use_align = kwargs_loss.get("use_align", False)
    dims_align = kwargs_loss.get("dims_align", 0)
    lam_align = kwargs_loss.get("lam_align", 0.0)
    lam_disent = kwargs_loss.get("lam_disent", 0.0)
    # use_disent = kwargs_loss.get('use_disent', False)
    if use_align:
        assert (
            mode_MER == "full"
        ), "Alignment loss can only be used with computing the full Jacobian!"

    use_rec = kwargs_loss.get("use_rec", False)
    dims_rec = kwargs_loss.get("dims_rec", None)
    lam_rec = kwargs_loss.get("lam_rec", 0.0)

    device = x.device
    batchsize = x.shape[0]
    N_dim = x.shape[1]

    if use_MER and mode_MER == "unbiased" and batchsize < N_dim:
        assert (
            metrics_last is not None
        ), "Metrics array from last update must be provided when using unbiased MER with batchsize smaller than N_dim!"
        assert metrics_last["H_i"] is not None
        H_i = metrics_last["H_i"]

    elif mode_MER == "unbiased":
        # assert that there are more or equal samples than dimensions
        assert (
            batchsize >= N_dim
        ), "For unbiased MER, number of samples must be larger than or equal to number of dimensions!"
    lam_MTC = kwargs_loss["lam_MTC"]  # weight of Manifold Total Correlation term
    lam_ME_i = kwargs_loss["lam_ME_i"]  # weights of Manifold Entropy terms per dimension

    eval_MEMetrics = kwargs_loss.get("eval_MEMetrics", False)

    get_jac_sing_vals = kwargs_loss.get("get_jac_sing_vals", False)

    z, ljd_enc = model(x, c=c, rev=False)  # pass through encoder
    NLL_z_i = get_NLL_z(kwargs_data, z=z)  # NLL per dimension

    NLL = NLL_z_i.sum(-1) - ljd_enc  # NLL per sample [B]
    if use_MER or eval_MEMetrics:
        if mode_MER == "full":
            jac_dec = []
            x_rec, ljd_dec = model(z, c=c, rev=True)  # pass through decoder
            for j in range(N_dim):
                jac_dec.append(grad(x_rec[:, j].sum(), z, create_graph=True)[0])
            jac_dec = torch.stack(jac_dec, axis=2)  # Jacobian of the decoder
            ljd_dec_i = 1 / 2 * torch.log(torch.sum(jac_dec**2, axis=-1))

            # ML           = torch.sum(NLL_z_i.sum(-1) - ljd, 0) #instead of '-ljd' we could also use '+ljd_inv' as the encoder and decoder are the inverse of each other
            # PF_objective = torch.sum(NLL_z_i + ljd_dec_i,   0) #we use the decoder Jacobian columns as in the brute force implementation
            NLL_i = NLL_z_i + ljd_dec_i  # [B x D]

            with torch.no_grad():
                H_i = NLL_i.mean(dim=0).detach()

            # loss = ((1-lam_MTC)*ML + torch.sum(lam_MTC*PF_objective)) / (batchsize * N_dim)
            lam_ME_i = torch.tensor(lam_ME_i).to(device)[None, :]  # [1 x D]
            loss = (
                torch.sum((1 - lam_MTC) * (use_NLL * NLL), dim=(0))
                + torch.sum((lam_MTC + lam_ME_i) * NLL_i, dim=(0, 1))
            ) / (batchsize * N_dim)

            if use_align:
                N_dim_core = dims_align
                N_dim_detail = N_dim - dims_align

                jac_dec_core = jac_dec[:, 0:dims_align]
                jac_dec_detail = jac_dec[:, dims_align:]

                NLL_z_core = NLL_z_i[:, 0:dims_align].sum(-1)
                NLL_z_detail = NLL_z_i[:, dims_align:].sum(-1)

                ljd_dec_core = (
                    1
                    / 2
                    * torch.logdet(torch.einsum("bik, bjk -> bij", jac_dec_core, jac_dec_core))
                )
                ljd_dec_detail = (
                    1
                    / 2
                    * torch.logdet(torch.einsum("bik, bjk -> bij", jac_dec_detail, jac_dec_detail))
                )

                NLL_core = NLL_z_core + ljd_dec_core
                NLL_detail = NLL_z_detail + ljd_dec_detail

                with torch.no_grad():
                    H_core = NLL_core.mean().detach()
                    H_detail = NLL_detail.mean().detach()
                    MI_core_detail = (NLL_core + NLL_detail - NLL).mean().detach()

                loss = loss + (
                    lam_align / N_dim_detail * NLL_detail.sum()
                    + lam_disent * (NLL_core + NLL_detail - NLL).sum()
                ) / (batchsize * N_dim)
            else:
                H_core = torch.tensor(0.0).to(device)
                H_detail = torch.tensor(0.0).to(device)
                MI_core_detail = torch.tensor(0.0).to(device)

            with torch.no_grad():
                MTC = (torch.sum(NLL_i, dim=(-1)) - NLL).mean().detach()
        elif mode_MER == "unbiased":
            # sample random indices
            p = [1 / N_dim] * N_dim

            if batchsize >= N_dim:
                e = torch.multinomial(torch.tensor(p), batchsize, replacement=True).to(device)
                # if N_dim <= batchsize: #ensure each dimension is sampled at least once
                rand_perm = torch.randperm(N_dim)
                e[rand_perm[0:N_dim]] = torch.arange(0, N_dim).to(device)
            else:
                e = torch.multinomial(torch.tensor(p), batchsize, replacement=True).to(device)
                rand_perm = torch.randperm(N_dim)
                e[0:batchsize] = rand_perm[0:batchsize].to(device)
            e = nn.functional.one_hot(e, num_classes=N_dim).to(x.dtype)  # create one-hot-vector

            with dual_level():  # compute jvp
                dual_z = make_dual(z, e)
                dual_x, _ = model(dual_z, c=c, rev=True)
                _, x_jvp = unpack_dual(dual_x)

            x_jvp = x_jvp.reshape(batchsize, -1)  # reshape jpv to [B x D]
            ljd_dec_i = 1 / 2 * torch.log(torch.sum(x_jvp**2, axis=-1))[:, None]  # [B x D]

            M = torch.sum(e, dim=0)  # [D]
            m = e * batchsize / (M[None, :] + 1e-8)  # [B x D]
            M = M.to(torch.bool)  # [D]

            lam_ME_i = torch.tensor(lam_ME_i).to(device)[None, :]  # [1 x D]

            if batchsize >= N_dim:
                NLL_i = m[:, M] * ljd_dec_i + M[None, :] * NLL_z_i  # [B x D]

                with torch.no_grad():
                    H_i = NLL_i.mean(dim=0).detach()
                    MTC = (torch.sum(NLL_i, dim=(-1)) - NLL).mean().detach()

                loss = (
                    torch.sum((1 - lam_MTC) * (use_NLL * NLL), dim=(0))
                    + torch.sum((lam_MTC + lam_ME_i) * NLL_i, dim=(0, -1))
                ) / (batchsize * N_dim)
            else:
                NLL_i = torch.sum((lam_MTC + lam_ME_i) * M * NLL_z_i, -1).sum()
                NLL_i += torch.sum((m[:, M] * (lam_MTC + lam_ME_i[:, M]) * ljd_dec_i), 0).sum()
                NLL_i = NLL_i * (N_dim / M.sum())

                with torch.no_grad():
                    H_i[M] = (
                        torch.sum(NLL_z_i[:, M], 0) + torch.sum((m[:, M] * ljd_dec_i), 0)
                    ).detach() / x.shape[0]
                    # check that H_i has no inf or nan values
                    if torch.any(torch.isnan(H_i)) or torch.any(torch.isinf(H_i)):
                        MTC = torch.tensor(0.0).to(device)
                    else:
                        MTC = H_i.sum(-1) - NLL.mean().detach()

                loss = (torch.sum((1 - lam_MTC) * (use_NLL * NLL), dim=(0)) + NLL_i) / (
                    batchsize * N_dim
                )

            print(f"NLL: {NLL}, NLL_i: {NLL_i}")
            H_core = torch.tensor(0.0).to(device)
            H_detail = torch.tensor(0.0).to(device)
            MI_core_detail = torch.tensor(0.0).to(device)
        if not use_MER:
            loss = (use_NLL * NLL).sum() / (batchsize * N_dim)
    else:
        loss = (use_NLL * NLL).sum() / (batchsize * N_dim)
        MTC = torch.tensor(0.0).to(device)
        H_i = torch.zeros(N_dim).to(device)
        H_core = torch.tensor(0.0).to(device)
        H_detail = torch.tensor(0.0).to(device)
        MI_core_detail = torch.tensor(0.0).to(device)

    if use_rec:
        assert dims_rec is not None, "dims_rec must be specified when use_rec is True!"
        z_rec = z.clone()
        z_rec[:, dims_rec:] = 0.0
        x_rec, _ = model(z_rec, c=c, rev=True)
        L2_rec = 1 / 2 * ((x_rec - x) ** 2).mean()
        loss = loss + lam_rec * L2_rec
    else:
        L2_rec = torch.tensor(0.0).to(device)

    # loss is used for backpropagation
    # ML, MTC and MSE in losses_eval are used for evaluation during training

    metrics = {
        "z2": (z**2).mean(dim=0).detach(),
        "NLL": NLL.mean().detach(),
        "MTC": MTC.mean().detach(),
        "H_i": H_i.detach(),
        "H_core": H_core.detach(),
        "H_detail": H_detail.detach(),
        "MI_core_detail": MI_core_detail.detach(),
        "L2_rec": L2_rec.detach(),
    }

    if get_jac_sing_vals:
        assert model[0].M is not None, "Linear transform M not found in first layer of flow!"

        jac_dec = model[0].M.detach()  # Linear transform matrix
        # S = 1/torch.linalg.svdvals(jac_dec)
        # log of inverse singular values
        S = torch.linalg.svdvals(jac_dec)
        S = -torch.log(S)
        # save
        metrics["sing_vals"] = S
    return loss, metrics
