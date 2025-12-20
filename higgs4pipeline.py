import pandas as pd
import numpy as np

from preprocessing_utils import *
from analysis_utils import *
from stats_utils import *
from plotting_utils import *


data_files = [
    "../datasets/data_A.4lep.root",
    "../datasets/data_B.4lep.root",
    "../datasets/data_C.4lep.root",
    "../datasets/data_D.4lep.root",
]

df_data = load_many_root_files(
    data_files,
    label="data",
    verbose=True,
)

signal_files = {
    "ggH": ["../datasets/mc_345060.ggH125_ZZ4lep.4lep.root"],
    "VBF": ["../datasets/mc_344235.VBFH125_ZZ4lep.4lep.root"],
    "WH":  ["../datasets/mc_341964.WH125_ZZ4lep.4lep.root"],
    "ZH":  ["../datasets/mc_341947.ZH125_ZZ4lep.4lep.root"],
}

df_signal = pd.concat(
    [
        load_many_root_files(files, label=proc)
        for proc, files in signal_files.items()
    ],
    ignore_index=True,
)

df_bkg = load_many_root_files(
    ["../datasets/mc_363490.llll.4lep.root"],
    label="ZZ",
)

# Preprocessing

pre_data = preprocess_events(df_data, is_mc=False)
pre_sig  = preprocess_events(df_signal, is_mc=True)
pre_bkg  = preprocess_events(df_bkg, is_mc=True)

# --- sigma × L normalization ---
pre_sig, pre_bkg, sig_scales, bkg_scale = (
    renormalize_signal_and_background(pre_sig, pre_bkg)
)

print("Signal normalization scales:", sig_scales)
print("Background normalization scale:", bkg_scale)

# ZZ reconstructon

pre_data = add_zz_columns(pre_data)
pre_sig  = add_zz_columns(pre_sig)
pre_bkg  = add_zz_columns(pre_bkg)

# Selections and cutflow

selections = define_selections()

mask_data = apply_selections(pre_data, selections)
mask_sig  = apply_selections(pre_sig, selections)
mask_bkg  = apply_selections(pre_bkg, selections)

data_sel = pre_data.loc[mask_data].copy()
sig_sel  = pre_sig.loc[mask_sig].copy()
bkg_sel  = pre_bkg.loc[mask_bkg].copy()

cutflow = build_cutflow(
    pre_data,
    pre_bkg,
    pre_sig,
    selections,
)

print("\nCutflow:")
print(cutflow)

# Higgs window
m_sig_lo = 115.0
m_sig_hi = 130.0

# Full range
m_min = 80.0
m_max = 250.0

sb_mask_data = (
    ((data_sel["m4l_GeV"] > m_min) & (data_sel["m4l_GeV"] < m_sig_lo)) |
    ((data_sel["m4l_GeV"] > m_sig_hi) & (data_sel["m4l_GeV"] < m_max))
)

sb_mask_bkg = (
    ((bkg_sel["m4l_GeV"] > m_min) & (bkg_sel["m4l_GeV"] < m_sig_lo)) |
    ((bkg_sel["m4l_GeV"] > m_sig_hi) & (bkg_sel["m4l_GeV"] < m_max))
)

k_sb = int(sb_mask_data.sum())
b_sb_mc = float(bkg_sel.loc[sb_mask_bkg, "event_weight"].sum())

if b_sb_mc <= 0:
    raise RuntimeError("Background MC has zero yield in sidebands")

scale_b = k_sb / b_sb_mc

print("\nBackground sideband normalization:")
print("  k_sb (data) =", k_sb)
print("  b_sb (MC)   =", b_sb_mc)
print("  scale_b     =", scale_b)


win_data = (data_sel["m4l_GeV"] > m_sig_lo) & (data_sel["m4l_GeV"] < m_sig_hi)
win_sig  = (sig_sel["m4l_GeV"]  > m_sig_lo) & (sig_sel["m4l_GeV"]  < m_sig_hi)
win_bkg  = (bkg_sel["m4l_GeV"]  > m_sig_lo) & (bkg_sel["m4l_GeV"]  < m_sig_hi)

k_win = int(win_data.sum())
s_win = float(sig_sel.loc[win_sig, "event_weight"].sum())
b_win = float(bkg_sel.loc[win_bkg, "event_weight"].sum()) * scale_b

print(f"\nHiggs window: {m_sig_lo:.1f} < m4l < {m_sig_hi:.1f} GeV")
print("  k (data) =", k_win)
print("  s (signal exp.) =", s_win)
print("  b (bkg exp.) =", b_win)

obs = counting_q0_Z(k_win, s_win, b_win)
exp = counting_asimov_Z(s_win, b_win)


# 1) Z-mass diagnostic (data only) is always fine
plot_Z_masses(pre_data, "Z candidate masses before selections")

# 2) Preselection shapes (normalize shapes so overlays are meaningful)
pre_sig_shape = pre_sig.copy()
pre_bkg_shape = pre_bkg.copy()

# Avoid divide-by-zero just in case
if pre_sig_shape["event_weight"].sum() > 0:
    pre_sig_shape["event_weight"] /= pre_sig_shape["event_weight"].sum()
if pre_bkg_shape["event_weight"].sum() > 0:
    pre_bkg_shape["event_weight"] /= pre_bkg_shape["event_weight"].sum()

plot_m4l_stage(pre_data, pre_sig_shape, pre_bkg_shape, "m4l before selections (shape-normalized MC)")

# 3) Post-selection: scale background to sidebands and signal to mu_hat
mu_hat = float(obs.get("mu_hat", 0.0))

sig_postfit = sig_sel.copy()
bkg_sideband = bkg_sel.copy()

sig_postfit["event_weight"] *= mu_hat
bkg_sideband["event_weight"] *= scale_b

plot_m4l_stage(data_sel, sig_postfit, bkg_sideband, f"m4l after selections (bkg sideband-scaled, sig × mu_hat={mu_hat:.3f})")


print("\nCounting-experiment significance:")
print("Observed:", obs)
print("Expected (Asimov):", exp)

bins = 34
x_range = (m_min, m_max)

s, b, _ = binned_templates(
    sig_sel,
    bkg_sel,
    bins=bins,
    x_range=x_range,
)

k, _ = binned_counts(
    data_sel,
    bins=bins,
    x_range=x_range,
)

print("\n[Diagnostic only] Shape likelihood:")
print(profile_likelihood_significance(k, s, b))
