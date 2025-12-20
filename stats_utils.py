import numpy as np


def binned_templates(df_sig, df_bkg, bins, x_range):
    """
    Build binned signal and background templates in m4l.
    """
    s, edges = np.histogram(
        df_sig["m4l_GeV"],
        bins=bins,
        range=x_range,
        weights=df_sig["event_weight"],
    )

    b, _ = np.histogram(
        df_bkg["m4l_GeV"],
        bins=bins,
        range=x_range,
        weights=df_bkg["event_weight"],
    )

    return s.astype(float), b.astype(float), edges


def binned_counts(df_data, bins, x_range):
    """
    Build binned data counts in m4l.
    """
    k, edges = np.histogram(
        df_data["m4l_GeV"],
        bins=bins,
        range=x_range,
    )

    return k.astype(float), edges


def _poisson_nll(k, lam):
    """
    Poisson negative log-likelihood (up to additive constant).
    k and lam can be scalars or arrays.
    """
    k = np.asarray(k, dtype=float)
    lam = np.asarray(lam, dtype=float)
    lam = np.clip(lam, 1e-12, None)
    return float(np.sum(lam - k * np.log(lam)))


def fit_mu_hat(k, s, b, mu_grid=None):
    """
    Find best-fit signal strength mu >= 0 via grid scan.
    """
    k = np.asarray(k, dtype=float)
    s = np.asarray(s, dtype=float)
    b = np.asarray(b, dtype=float)

    if mu_grid is None:
        mu_grid = np.linspace(0.0, 10.0, 4001)

    nll_vals = np.array([
        _poisson_nll(k, mu * s + b)
        for mu in mu_grid
    ])

    idx = int(np.argmin(nll_vals))
    return float(mu_grid[idx]), float(nll_vals[idx])


def profile_likelihood_significance(k, s, b):
    """
    Compute observed profile-likelihood significance Z for a binned model.
    """
    mu_hat, nll_hat = fit_mu_hat(k, s, b)
    nll_0 = _poisson_nll(k, b)

    q0 = max(0.0, 2.0 * (nll_0 - nll_hat))
    Z = float(np.sqrt(q0))

    return {
        "mu_hat": mu_hat,
        "q0": q0,
        "Z_obs": Z,
    }


def expected_significance_asimov(s, b):
    """
    Compute expected (Asimov) significance for a binned model.
    """
    s = np.asarray(s, dtype=float)
    b = np.asarray(b, dtype=float)
    kA = s + b

    mu_hat, nll_hat = fit_mu_hat(kA, s, b)
    nll_0 = _poisson_nll(kA, b)

    q0 = max(0.0, 2.0 * (nll_0 - nll_hat))
    Z = float(np.sqrt(q0))

    return {
        "mu_hat_asimov": mu_hat,
        "q0_asimov": q0,
        "Z_exp": Z,
    }


def counting_q0_Z(k, s, b, mu_max=10.0, n_grid=4001):
    """
    Profile-likelihood q0 and Z for a *single-bin* counting experiment.

    Model: Poisson(k | mu*s + b), with mu >= 0.
    """
    k = float(k)
    s = float(s)
    b = float(b)

    mu_grid = np.linspace(0.0, float(mu_max), int(n_grid))
    nll_vals = np.array([_poisson_nll(k, mu * s + b) for mu in mu_grid])

    idx = int(np.argmin(nll_vals))
    mu_hat = float(mu_grid[idx])
    nll_hat = float(nll_vals[idx])

    nll_0 = _poisson_nll(k, b)
    q0 = max(0.0, 2.0 * (nll_0 - nll_hat))
    Z = float(np.sqrt(q0))

    return {"mu_hat": mu_hat, "q0": q0, "Z": Z}


def counting_asimov_Z(s, b):
    """
    Expected (Asimov) Z for a single-bin counting experiment.
    Uses the standard closed form (Cowan et al.) for discovery with known b.
    """
    s = float(s)
    b = float(b)
    if b <= 0:
        return {"Z_exp": float("nan")}

    # Z_A = sqrt(2[(s+b)ln(1+s/b)-s])
    Z = float(np.sqrt(2.0 * ((s + b) * np.log(1.0 + s / b) - s)))
    return {"Z_exp": Z}
