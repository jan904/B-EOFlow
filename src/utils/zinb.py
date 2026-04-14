import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def sample_zinb(psi, mu, alpha, size):
    # Sample from the zero-inflated negative binomial distribution
    with pm.Model() as zinb_model:
        obs = pm.ZeroInflatedNegativeBinomial(
            "obs",
            psi=psi,
            mu=mu,
            alpha=alpha,
            shape=size,
        )
        samples = pm.sample_prior_predictive(samples=1)
    return samples.prior["obs"][0][0]


def fit_zinb(data):
    with pm.Model() as model:
        mu = pm.HalfNormal("mu", sigma=10.0)
        alpha = pm.HalfNormal("alpha", sigma=10.0)
        pi = pm.Beta("pi", alpha=1.0, beta=1.0)

        obs = pm.ZeroInflatedNegativeBinomial(
            "obs",
            psi=pi,
            mu=mu,
            alpha=alpha,
            observed=data,
        )
        idata = pm.sample()

    mu_mean = idata.posterior["mu"][0].median().item()
    alpha_mean = idata.posterior["alpha"][0].median().item()
    pi_mean = idata.posterior["pi"][0].median().item()

    return mu_mean, alpha_mean, pi_mean


def inferr_zinb_params(adata, n=500, gene=None):
    plot_data_counts = adata.layers["counts"].toarray()

    if gene is None:
        rand_genes = np.random.permutation(plot_data_counts.shape[1])[:100]
    else:
        rand_genes = [gene]

    for gene in rand_genes:
        samples = plot_data_counts[:, gene]
        mu_mean, alpha_mean, pi_mean = fit_zinb(samples)
        samples_inferred = sample_zinb(psi=pi_mean, mu=mu_mean, alpha=alpha_mean, size=17913)
        max_value = np.where(samples > 100)[0].min() if np.any(samples > 100) else samples.max()

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(
            samples,
            bins=30,
            kde=False,
            ax=ax[0],
            color="blue",
            label="Original",
            binwidth=1,
        )
        sns.histplot(
            samples_inferred,
            bins=30,
            kde=False,
            ax=ax[1],
            color="orange",
            label="Inferred",
            binwidth=1,
        )
        ax[0].set_title("Sample Distributions")
        ax[0].set_xlim(0, max_value + 1)
        ax[0].legend()
        ax[1].set_title("Inferred Distributions")
        ax[1].set_xlim(0, max_value + 1)
        ax[1].legend()
        plt.suptitle(f"mu={mu_mean:.2f}, alpha={alpha_mean:.2f}, pi={pi_mean:.2f}")
        plt.tight_layout()
        plt.savefig(f"/home/jhoefer/sandbox/results/zinb_fits/gene_{gene}.png")
        plt.show()
