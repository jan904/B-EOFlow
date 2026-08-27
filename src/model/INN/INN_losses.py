import torch
import torch.nn as nn
from torch.autograd import grad
from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual
import numpy as np


class DimensionSampler:
    def __init__(self, N_dim, device):
        self.N_dim = N_dim
        self.counts = torch.zeros(N_dim)
        self.device = device

    def _get_adjusted_p(self, p):
        counts = self.counts

        sample_weights = 1.0 / (counts + 1.0)
        sample_weights = sample_weights / sample_weights.sum()

        adjusted_p = torch.tensor(p) * sample_weights
        adjusted_p = adjusted_p / adjusted_p.sum()
        return adjusted_p

    def sample(self, batchsize):
        p = [1 / self.N_dim] * self.N_dim
        adjusted_p = self._get_adjusted_p(p)

        if batchsize >= self.N_dim:
            e = torch.multinomial(torch.tensor(adjusted_p), batchsize, replacement=True).to(
                self.device
            )
            # if N_dim <= batchsize: #ensure each dimension is sampled at least once
            rand_perm = torch.randperm(self.N_dim)
            e[rand_perm[0 : self.N_dim]] = torch.arange(0, self.N_dim).to(self.device)

        else:
            e = torch.multinomial(torch.tensor(adjusted_p), batchsize, replacement=True).to(
                self.device
            )
            rand_perm = torch.randperm(self.N_dim)
            e[0:batchsize] = rand_perm[0:batchsize].to(self.device)

        for dim in e:
            self.counts[dim.item()] += 1

        e = nn.functional.one_hot(e, num_classes=self.N_dim).to(torch.float32)
        return e


def round_loss(loss, round=4):
    # check if loss is a list
    if isinstance(loss, list):
        return [round_loss(l, round) for l in loss]
    else:
        return np.round(loss.item(), round)


def get_NLL_z(kwargs_data, z=None, c=None, means=None, log_sigma=None):
    # Assume standard normal prior p(z)
    # Compute the negative log-likelihood (NLL) using samples z per dimension. Alternatively this is the entropy of p_i(z_i) if z is None
    N_dim = kwargs_data["N_dim"]

    if c is None:
        if z is None:
            z = torch.ones(N_dim)
        return 1 / 2 * (z**2)  # + torch.tensor(1 / 2 * np.log(2 * np.pi))
    else:
        if c.dim() == 1:
            c = c.unsqueeze(0)
        c = c.float()
        mu = c @ means
        if z is None:
            z = torch.ones_like(mu)

        if log_sigma is None:
            return 1 / 2 * ((z - mu) ** 2)  # + torch.tensor(1 / 2 * np.log(2 * np.pi))

        log_sigma = c @ log_sigma.unsqueeze(1)
        sigma = torch.exp(log_sigma)

        return (
            1 / 2 * ((z - mu) / sigma) ** 2 + log_sigma
        )  # + torch.tensor(1 / 2 * np.log(2 * np.pi))


import time


def get_loss(
    model,
    x,
    kwargs_data,
    kwargs_loss,
    sampler,
    model_config,
    c=None,
    metrics_last=None,
):

    # If condition_type is 'normal', c is directly used as condition for the flow
    # If condition_type is 'mixture', only the loss depends on c but not the flow itself
    if c is not None:
        if model_config.condition_type == "normal":
            c_model = c
            c_prior = None
        elif model_config.condition_type == "mixture":
            c_model = None
            c_prior = c[0]

    start_ = time.time()
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

    # elif mode_MER == "unbiased":
    #     # assert that there are more or equal samples than dimensions
    #     assert (
    #         batchsize >= N_dim
    #     ), "For unbiased MER, number of samples must be larger than or equal to number of dimensions!"
    lam_MTC = kwargs_loss["lam_MTC"]  # weight of Manifold Total Correlation term
    lam_ME_i = kwargs_loss["lam_ME_i"]  # weights of Manifold Entropy terms per dimension

    eval_MEMetrics = kwargs_loss.get("eval_MEMetrics", False)

    get_jac_sing_vals = kwargs_loss.get("get_jac_sing_vals", False)

    t1 = time.time() - start_
    start = time.time()

    z, ljd_enc = model(x, c=c_model, rev=False)  # pass through encoder

    t2 = time.time() - start
    start = time.time()

    t3 = time.time() - start
    start = time.time()

    NLL_z_i = get_NLL_z(
        kwargs_data, z=z, c=c_prior, means=model.means, log_sigma=model.log_sigma
    )  # NLL per dimension
    NLL = NLL_z_i.sum(-1) - ljd_enc  # NLL per sample [B]

    if model_config.supervise_latent_meaning == "partition":
        NLL_z_i = get_NLL_z(
            kwargs_data, z=z, c=c_prior, means=model.means, log_sigma=model.log_sigma
        )  # NLL per dimension

        # The prior is partitioned: the first `means_dim` latent dims carry the
        # condition mean, the remaining N_dim - means_dim dims share a N(0,1) prior
        # across all conditions. `factor` rebalances the two blocks so the condition
        # dims aren't drowned out by the far more numerous shared dims -
        # (N_dim - means_dim) / means_dim equalises their total weight, and
        # `partition_divisor` tempers that from there (same convention as the
        # `supervise_latent_meaning == "both"` branch below).
        #
        # This used to read `means_active.shape[0]` / `len(means_active)` - both the
        # number of components (K), not the number of condition-carrying dims - while
        # indexing `dim_weight` by `shape[1]`. With K=198, means_dim=396, N_dim=1937,
        # divisor=8 that produced 1.098 instead of 0.486: a dimension-vs-component mixup
        # that also happened to be silently correct only when partition_divisor equalled
        # latent_per_condition.
        #
        # `partition_weight`, when set, replaces the derived value outright. The formula
        # above scales as 1/means_dim, so widening the block silently weakens the pull on
        # exactly the dims the prior means occupy - measured across a width sweep, the
        # empirical-vs-learned mean gap grew from 6.8% of ||mu|| at means_dim=29 to 51.8%
        # at 1450, and the share of that gap which corrupts a counterfactual (rather than
        # cancelling as a per-cell-type offset) went from 23% to 52%. Pinning the weight
        # makes `latent_per_condition` a capacity knob only.
        means_dim = model.means_active.shape[1]
        factor = getattr(model_config, "partition_weight", None)
        if factor is None:
            factor = (N_dim - means_dim) / means_dim / model_config.partition_divisor
        dim_weight = torch.ones(N_dim, device=device)
        dim_weight[:means_dim] = factor  # upweight or downweight the condition dims

        NLL_z_i = NLL_z_i * dim_weight[None, :]  # broadcast across batch

        if model_config.balance_classes:
            class_idx = c_prior.argmax(dim=1)  # [B]
            class_count = torch.bincount(
                class_idx, minlength=len(model.means_active)
            ).float()  # [C]

            # Normalize over the samples, not over the C combo slots. `class_count` is
            # per batch, and with C=198 slots at batch 1024 roughly 72 of them are empty
            # in any given batch - under the old `1/(count + 1e-8)` those absent slots
            # took weight 1e8 and, being a third of the vector, dominated the mean the
            # weights were then divided by. Since `sample_weights` only ever indexes
            # classes that ARE present, every sample came out at ~3e-9: the prior term
            # was scaled away to nothing while `- ljd_enc` below stayed at full size, so
            # the flow maximised the encoder log-det unopposed and z diverged from epoch
            # one (z2 ~ 1e14, then NaN).
            #
            # Indexing first and normalizing after keeps the mean weight at exactly 1, so
            # balancing changes the NLL's *distribution* across combos without touching
            # its scale. No epsilon is needed: a class reached through `class_idx` has at
            # least one sample in the batch by construction.
            sample_weights = 1.0 / class_count[class_idx]  # [B]
            sample_weights = sample_weights / sample_weights.mean()

            NLL_z_i = NLL_z_i * sample_weights[:, None]  # broadcast across dimensions

        NLL = NLL_z_i.sum(-1) - ljd_enc  # NLL per sample [B]

    if use_MER or eval_MEMetrics:
        if mode_MER == "full":
            jac_dec = []
            x_rec, ljd_dec = model(z, c=c_model, rev=True)  # pass through decoder
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
            e = sampler.sample(batchsize)

            with dual_level():  # compute jvp
                dual_z = make_dual(z, e)
                dual_x, _ = model(dual_z, c=c_model, rev=True)
                _, x_jvp = unpack_dual(dual_x)

            x_jvp = x_jvp.reshape(batchsize, -1)  # reshape jpv to [B x D]
            ljd_dec_i = 1 / 2 * torch.log(torch.sum(x_jvp**2, axis=-1))[:, None]  # [B x D]

            M = torch.sum(e, dim=0)  # [D]
            m = e * batchsize / (M[None, :] + 1e-8)  # [B x D]
            M = M.to(torch.bool)  # [D]

            lam_ME_i = torch.tensor(lam_ME_i).to(device)[None, :]  # [1 x D]

            t4 = time.time() - start
            start = time.time()

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
                t5 = time.time() - start
                start = time.time()

                loss = (torch.sum((1 - lam_MTC) * (use_NLL * NLL), dim=(0)) + NLL_i) / (
                    batchsize * N_dim
                )
                t6 = time.time() - start
                start = time.time()

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
        x_rec, _ = model(z_rec, c=c_model, rev=True)
        L2_rec = 1 / 2 * ((x_rec - x) ** 2).mean()
        loss = loss + lam_rec * L2_rec
    else:
        L2_rec = torch.tensor(0.0).to(device)

    # loss is used for sbackpropagation
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

    t7 = time.time() - start
    full_time = time.time() - start_

    # print(f"Time for forward pass: {full_time:.4f} seconds")
    # print(
    #     f"T1: {t1/full_time:.2f}, T2: {t2/full_time:.2f}, T3: {t3/full_time:.2f}, T4: {t4/full_time:.2f}, T5: {t5/full_time:.2f}, T6: {t6/full_time:.2f}",
    #     f"T7: {t7/full_time:.2f}",
    # )

    return loss, metrics


def append_losses(losses, epoch, metrics, loss):
    # torch.nn.utils.clip_grad_norm_(f.parameters(), 10)
    losses["epoch"] = np.append(losses["epoch"], epoch)
    losses["loss"] = np.append(losses["loss"], loss.cpu().detach().numpy())
    losses["z2"] = np.append(losses["z2"], metrics["z2"].mean().cpu().detach().numpy())
    losses["NLL"] = np.append(losses["NLL"], metrics["NLL"].cpu().detach().numpy())
    losses["MTC"] = np.append(losses["MTC"], metrics["MTC"].cpu().detach().numpy())
    # H_i is a vector (one entry per dimension), not a scalar: keep only the latest one
    # rather than appending it. The training loop reads it back as `losses["H_i"][-N_dim:]`
    # to seed the next step's per-dimension entropy estimate, so storing just the most
    # recent vector is equivalent to appending and slicing - while keeping the array at
    # N_dim floats instead of growing by N_dim every batch (~200 KB/epoch, which is what
    # pushed checkpoints into the GBs).
    #
    # Dropping the assignment altogether leaves the array empty, and `len(losses["H_i"])
    # < N_dim` then sends INN_training down its "initialize with infinite values" branch
    # on *every* step: `torch.zeros(N_dim) * torch.inf` is all-NaN, which trips the isnan
    # guard in get_loss and reports MTC as exactly 0.0 for the whole run - a dead metric
    # that reads like the regularizer having driven total correlation to zero.
    losses["H_i"] = metrics["H_i"].cpu().detach().numpy()
    losses["H_core"] = np.append(losses["H_core"], metrics["H_core"].cpu().detach().numpy())
    losses["H_detail"] = np.append(losses["H_detail"], metrics["H_detail"].cpu().detach().numpy())
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

    return losses


def build_regularization_loss_string(losses, kwargs_loss, kwargs_data, N_batches):
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

    return losses_regularization
