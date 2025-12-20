import uproot
import numpy as np
import pandas as pd


LUMI_FBINV = 10.0
LUMI_PBINV = LUMI_FBINV * 1000.0


# Effective cross sections to the 4l final state (pb)
# 13 TeV, SM Higgs, decay-filtered samples
SIGMA_PB_SIGNAL = {
    "ggH": 0.0126,
    "VBF": 0.0010,
    "WH":  0.00023,
    "ZH":  0.00019,
}

SIGMA_PB_BKG = {
    "ZZ": 0.016,
}


def pad_lepton_columns(df, n_lep=4):
    """
    Unroll jagged lepton arrays into fixed columns with NaN padding.
    """
    for i in range(n_lep):
        df[f"lep_pt_{i}"] = df["lep_pt"].apply(lambda x: x[i] if len(x) > i else np.nan)
        df[f"lep_eta_{i}"] = df["lep_eta"].apply(lambda x: x[i] if len(x) > i else np.nan)
        df[f"lep_phi_{i}"] = df["lep_phi"].apply(lambda x: x[i] if len(x) > i else np.nan)
        df[f"lep_E_{i}"] = df["lep_E"].apply(lambda x: x[i] if len(x) > i else np.nan)
        df[f"lep_type_{i}"] = df["lep_type"].apply(lambda x: x[i] if len(x) > i else np.nan)
        df[f"lep_charge_{i}"] = df["lep_charge"].apply(lambda x: x[i] if len(x) > i else np.nan)

    return df.drop(
        columns=["lep_pt", "lep_eta", "lep_phi", "lep_E", "lep_type", "lep_charge"]
    )


def root_to_pandas(filename, branches=None, verbose=False):
    """
    Load the first TTree in a ROOT file into a pandas DataFrame.
    """
    if branches is None:
        branches = [
            "eventNumber", "lep_n",
            "lep_pt", "lep_eta", "lep_phi", "lep_E",
            "lep_type", "lep_charge",
            "mcWeight",
        ]

    with uproot.open(filename) as file:
        trees = [
            (key, obj) for key, obj in file.items()
            if isinstance(obj, uproot.behaviors.TTree.TTree)
        ]

        if not trees:
            raise RuntimeError("No TTree found in ROOT file")

        if verbose:
            print("Found TTrees:")
            for key, _ in trees:
                print(f"  {key}")

        tree = trees[0][1]
        df = tree.arrays(branches, library="pd")

    return pad_lepton_columns(df, n_lep=4)


def load_many_root_files(file_list, label=None, **kwargs):
    """
    Load and concatenate multiple ROOT files.
    """
    dfs = []
    for fname in file_list:
        df = root_to_pandas(fname, **kwargs)
        df = df.assign(source_file=fname, sample=label)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# Event-level preprocessing

def classify_channel_padded(row):
    """
    Classify a 4-lepton event by flavor composition.
    """
    types = [row[f"lep_type_{i}"] for i in range(4)]
    n_e = sum(abs(t) == 11 for t in types)
    n_mu = sum(abs(t) == 13 for t in types)

    if n_e == 4:
        return "4e"
    elif n_mu == 4:
        return "4mu"
    elif n_e == 2 and n_mu == 2:
        return "2e2mu"
    else:
        return "other"


def compute_m4l_padded(row):
    """
    Compute four-lepton invariant mass (MeV).
    """
    pt  = np.array([row[f"lep_pt_{i}"]  for i in range(4)])
    eta = np.array([row[f"lep_eta_{i}"] for i in range(4)])
    phi = np.array([row[f"lep_phi_{i}"] for i in range(4)])
    E   = np.array([row[f"lep_E_{i}"]   for i in range(4)])

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    E_tot  = E.sum()
    px_tot = px.sum()
    py_tot = py.sum()
    pz_tot = pz.sum()

    m2 = E_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2)
    return np.sqrt(max(m2, 0.0))


def preprocess_events(df, is_mc):
    """
    Add event-level quantities:
      - channel
      - m4l (MeV, GeV)
      - raw event_weight
    """
    df = df.copy()

    df["channel"] = df.apply(classify_channel_padded, axis=1)
    df["m4l_MeV"] = df.apply(compute_m4l_padded, axis=1)
    df["m4l_GeV"] = df["m4l_MeV"] / 1000.0

    if is_mc:
        df["event_weight"] = df["mcWeight"].astype(float)
    else:
        df["event_weight"] = 1.0

    return df


sig_scale = 0.07


def renormalize_mc(df, sigma_pb, lumi_pb=LUMI_PBINV, extra_scale=1.0):
    """
    Rescale MC event_weight so that sum(weights) = sigma × lumi × extra_scale.
    """
    current = df["event_weight"].sum()
    if current <= 0:
        raise ValueError("MC sample has non-positive total weight")

    scale = (sigma_pb * lumi_pb / current) * float(extra_scale)
    df = df.copy()
    df["event_weight"] *= scale
    return df, scale


def renormalize_signal_and_background(pre_sig, pre_bkg):
    """
    Apply sigma × L normalization to signal and background MC samples.

    - Signal gets an additional sig_scale to match a small dataset sensitivity.
    - Background uses plain sigma×L (no toy scaling) and is later sideband-normalized.
    """
    # Signal: normalize per production mode (with toy scale)
    sig_parts = []
    sig_scales = {}

    for proc, sigma in SIGMA_PB_SIGNAL.items():
        part = pre_sig[pre_sig["sample"] == proc].copy()
        part, scale = renormalize_mc(part, sigma, extra_scale=sig_scale)
        sig_scales[proc] = scale
        sig_parts.append(part)

    pre_sig = pd.concat(sig_parts, ignore_index=True)

    # Background: normalize with sigma×L only (no toy scaling)
    pre_bkg, bkg_scale = renormalize_mc(pre_bkg, SIGMA_PB_BKG["ZZ"], extra_scale=1.0)

    return pre_sig, pre_bkg, sig_scales, bkg_scale