import uproot
import numpy as np
import pandas as pd

# Data Loading

def pad_lepton_columns(df, n_lep=4):
    """
    Unroll jagged lepton arrays into fixed columns with NaN padding.
    """
    for i in range(n_lep):
        df[f"lep_pt_{i}"] = df["lep_pt"].apply(
            lambda x: x[i] if len(x) > i else np.nan
        )
        df[f"lep_eta_{i}"] = df["lep_eta"].apply(
            lambda x: x[i] if len(x) > i else np.nan
        )
        df[f"lep_phi_{i}"] = df["lep_phi"].apply(
            lambda x: x[i] if len(x) > i else np.nan
        )
        df[f"lep_E_{i}"] = df["lep_E"].apply(
            lambda x: x[i] if len(x) > i else np.nan
        )
        df[f"lep_type_{i}"] = df["lep_type"].apply(
            lambda x: x[i] if len(x) > i else np.nan
        )

    return df.drop(columns=["lep_pt", "lep_eta", "lep_phi", "lep_E", "lep_type"])


def root_to_pandas(filename, branches=None, verbose=False):
    """
    Load the first TTree found in a ROOT file into a pandas DataFrame,
    with lepton branches NaN-padded into flat columns.
    """
    if branches is None:
        branches = ["eventNumber", "lep_n", "lep_pt", "lep_eta", "lep_phi", "lep_E", "lep_type", "mcWeight"]

    with uproot.open(filename) as file:
        trees = [(key, obj) for key, obj in file.items() if isinstance(obj, uproot.behaviors.TTree.TTree)]

        if not trees:
            raise ValueError("No TTree found")

        if verbose:
            print("Found TTrees:")
            for key, _ in trees:
                print(f"  {key}")

        tree = trees[0][1]
        df = tree.arrays(branches, library="pd")

    df = pad_lepton_columns(df, n_lep=4)
    return df


def load_many_root_files(file_list, label=None, **kwargs):
    dfs = []
    for fname in file_list:
        df = root_to_pandas(fname, **kwargs)
        df = df.assign(source_file=fname,sample=label)
        dfs.append(df)

    return pd.concat(dfs, axis=0, copy=False, ignore_index=True)


# Data Processing

def classify_channel_padded(row):
    """
    Classify a 4-lepton event by lepton flavor (padded columns).
    """
    types = [row["lep_type_0"], row["lep_type_1"], row["lep_type_2"], row["lep_type_3"]]

    n_e  = sum(abs(x) == 11 for x in types)
    n_mu = sum(abs(x) == 13 for x in types)

    if n_e == 4: return "4e"
    elif n_mu == 4: return "4mu"
    elif n_e == 2 and n_mu == 2: return "2e2mu"
    else: return "other"


def compute_m4l_padded(row):
    """
    Compute the four-lepton invariant mass (MeV) from padded columns.
    """
    pt  = np.array([row[f"lep_pt_{i}"]  for i in range(4)])
    eta = np.array([row[f"lep_eta_{i}"] for i in range(4)])
    phi = np.array([row[f"lep_phi_{i}"] for i in range(4)])
    E   = np.array([row[f"lep_E_{i}"]   for i in range(4)])

    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    px_tot, py_tot, pz_tot = px.sum(), py.sum(), pz.sum()
    E_tot  = E.sum()
    m2 = E_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2)
    return np.sqrt(max(m2, 0.0))


def preprocess_events(df, is_mc=False, lumi_fb=10.0, total_mc_events=None):
    """
    Preprocess event-level data or MC for H→4ℓ analysis
    using padded (non-jagged) lepton columns.
    """
    df = df.copy()
    df = df[df["lep_n"] == 4]

    df["channel"] = df.apply(classify_channel_padded, axis=1)
    df["m4l"] = df.apply(compute_m4l_padded, axis=1)
    df["m4l_GeV"] = df["m4l"] / 1000.0

    if is_mc:
        if total_mc_events is None:
            raise ValueError("total_mc_events must be provided for MC normalization")
        df["event_weight"] = df["mcWeight"] * (lumi_fb / total_mc_events)
    else:
        df["event_weight"] = 1.0

    return df
