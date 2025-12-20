import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "legend.frameon": False,
})

def plot_m4l_stage(data, sig, bkg, title, bins=40, xlim=(80, 250)):
    """
    Publication-quality m4l distribution:
    data points + signal line + background fill
    """

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    # Bin definition
    bin_edges = np.linspace(*xlim, bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]

    # --- Background ---
    ax.hist(
        bkg["m4l_GeV"],
        bins=bin_edges,
        weights=bkg["event_weight"],
        histtype="stepfilled",
        color="#9ecae1",
        edgecolor="#08519c",
        alpha=0.8,
        label="Background (ZZ)",
    )

    # --- Signal ---
    ax.hist(
        sig["m4l_GeV"],
        bins=bin_edges,
        weights=sig["event_weight"],
        histtype="step",
        color="#cb181d",
        linewidth=2.2,
        label=r"Signal ($H \to 4\ell$)",
    )

    # --- Data ---
    counts, _ = np.histogram(data["m4l_GeV"], bins=bin_edges)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    ax.errorbar(
        centers,
        counts,
        yerr=np.sqrt(counts),
        fmt="o",
        color="black",
        markersize=4.5,
        capsize=2,
        label="Data",
        zorder=10,
    )

    # Labels
    ax.set_xlabel(r"$m_{4\ell}$ [GeV]")
    ax.set_ylabel(f"Events / {bin_width:.1f} GeV")
    ax.set_xlim(xlim)
    ax.set_title(title)

    # Minor ticks
    ax.minorticks_on()

    # Legend (data first)
    handles, labels = ax.get_legend_handles_labels()
    order = [2, 1, 0]
    ax.legend([handles[i] for i in order], [labels[i] for i in order])

    fig.tight_layout()
    plt.show()


def plot_Z_masses(df, title):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    bins_z1 = np.linspace(40, 120, 41)
    bins_z2 = np.linspace(0, 120, 41)

    ax.hist(
        df["mZ1_GeV"],
        bins=bins_z1,
        histtype="step",
        linewidth=2,
        color="#08519c",
        label=r"$Z_1$",
    )

    ax.hist(
        df["mZ2_GeV"],
        bins=bins_z2,
        histtype="step",
        linewidth=2,
        linestyle="--",
        color="#cb181d",
        label=r"$Z_2$",
    )

    ax.axvline(
        91.1876,
        linestyle=":",
        linewidth=1.8,
        color="black",
        label=r"$m_Z$",
    )

    ax.set_xlabel("Invariant mass [GeV]")
    ax.set_ylabel("Events")
    ax.set_title(title)
    ax.set_xlim(0, 120)
    ax.minorticks_on()
    ax.legend()

    fig.tight_layout()
    plt.show()


