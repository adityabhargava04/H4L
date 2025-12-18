import numpy as np
import matplotlib.pyplot as plt


def plot_m4l_density(df_bkg, df_sig, df_data, fig=None, mass_range=(80, 250), n_bins=80):
    """
    Plot normalized m4l density distributions for background, signal, and data.
    """
    bins = np.linspace(mass_range[0], mass_range[1], n_bins)
    if fig is None:
        fig = plt.figure(figsize=(7, 5))
    ax = fig.gca()

    # Background
    ax.hist(df_bkg["m4l_GeV"], bins=bins, weights=df_bkg["event_weight"], density=True, 
            histtype="stepfilled", alpha=0.4, label="Background",)
    # Signal
    ax.hist(df_sig["m4l_GeV"], bins=bins, weights=df_sig["event_weight"], density=True,
            histtype="step", linewidth=2.0, label="H → 4ℓ (MC)")
    # Data
    ax.hist(df_data["m4l_GeV"], bins=bins, density=True, histtype="step", linewidth=1.8, label="Data")

    ax.set_xlim(*mass_range)
    ax.set_xlabel(r"$m_{4\ell}$ [GeV]"); ax.set_ylabel("Probability density")
    ax.legend(frameon=False)
    
    fig.tight_layout()
    return fig
