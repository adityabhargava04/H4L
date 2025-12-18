import numpy as np
import pandas as pd

def define_selections():
    """
    Define physics selections as functions acting on a DataFrame.
    Returns an ordered dict of callables.
    """
    selections = {}
    # Exactly four leptons 
    selections["4_leptons"] = lambda df: (df["lep_n"] == 4)

    # Lepton pT cuts (MeV)
    def pt_cuts(df):
        pts = np.vstack([
            df["lep_pt_0"],
            df["lep_pt_1"],
            df["lep_pt_2"],
            df["lep_pt_3"],
        ]).T

        pts_sorted = np.sort(pts, axis=1)[:, ::-1]

        return (
            (pts_sorted[:, 0] > 20_000) &
            (pts_sorted[:, 1] > 15_000) &
            (pts_sorted[:, 2] > 10_000) &
            (pts_sorted[:, 3] > 10_000)
        )

    selections["pt_cuts"] = pt_cuts

    return selections


def build_cutflow(df_data, df_bkg, df_sig, selection_fns):
    """
    Build a cutflow table from selection functions.
    """
    cutflow = []

    mask_data = np.ones(len(df_data), dtype=bool)
    mask_bkg  = np.ones(len(df_bkg),  dtype=bool)
    mask_sig  = np.ones(len(df_sig),  dtype=bool)

    # All events
    cutflow.append({
        "cut": "All events",
        "data": len(df_data),
        "bkg": df_bkg["event_weight"].sum(),
        "sig": df_sig["event_weight"].sum(),
    })

    # Sequential cuts
    for name, fn in selection_fns.items():
        mask_data &= fn(df_data)
        mask_bkg  &= fn(df_bkg)
        mask_sig  &= fn(df_sig)

        cutflow.append({
            "cut": name,
            "data": mask_data.sum(),
            "bkg": df_bkg.loc[mask_bkg, "event_weight"].sum(),
            "sig": df_sig.loc[mask_sig, "event_weight"].sum(),
        })

    return pd.DataFrame(cutflow)


def apply_selections(df, selection_fns, upto=None):
    """
    Apply selections sequentially and return the cumulative mask.

    """
    mask = np.ones(len(df), dtype=bool)
    for name, fn in selection_fns.items():
        mask &= fn(df)
        if name == upto:
            break

    return mask


# ZZ* Reconstruction

import numpy as np
import pandas as pd

def _p4_from_lepton(row, i):
    pt  = row[f"lep_pt_{i}"]
    eta = row[f"lep_eta_{i}"]
    phi = row[f"lep_phi_{i}"]
    E   = row[f"lep_E_{i}"]

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return E, px, py, pz  # MeV

def _mll(row, i, j):
    Ei, pxi, pyi, pzi = _p4_from_lepton(row, i)
    Ej, pxj, pyj, pzj = _p4_from_lepton(row, j)

    E  = Ei + Ej
    px = pxi + pxj
    py = pyi + pyj
    pz = pzi + pzj

    m2 = E*E - (px*px + py*py + pz*pz)
    return np.sqrt(max(m2, 0.0))  # MeV


MZ_GEV = 91.1876
MZ_MEV = MZ_GEV * 1000.0

def reconstruct_zz_row(row):
    """
    Reconstruct Z1 and Z2 from a 4-lepton event using SFOS pairing.

    Returns
    -------
    dict with keys:
      - mZ1_GeV, mZ2_GeV (floats, NaN if fail)
      - z1_pair, z2_pair (tuples of indices, or None)
    """
    # Require 4 leptons present (padded rows could include NaNs)
    # We'll treat NaN in pt as missing lepton.
    for i in range(4):
        if not np.isfinite(row.get(f"lep_pt_{i}", np.nan)):
            return {"mZ1_GeV": np.nan, "mZ2_GeV": np.nan, "z1_pair": None, "z2_pair": None}

    # Build SFOS pairs
    pairs = []
    for i in range(4):
        ti = row[f"lep_type_{i}"]
        qi = row[f"lep_charge_{i}"]
        for j in range(i+1, 4):
            tj = row[f"lep_type_{j}"]
            qj = row[f"lep_charge_{j}"]

            same_flavor = (abs(ti) == abs(tj)) and (abs(ti) in (11, 13))
            opp_sign    = (qi * qj == -1)

            if same_flavor and opp_sign:
                mll = _mll(row, i, j)  # MeV
                pairs.append((i, j, mll))

    if len(pairs) == 0:
        return {"mZ1_GeV": np.nan, "mZ2_GeV": np.nan, "z1_pair": None, "z2_pair": None}

    # Choose Z1: closest to mZ
    pairs.sort(key=lambda x: abs(x[2] - MZ_MEV))
    i1, j1, mZ1 = pairs[0]

    # Z2 from remaining leptons
    remaining = [k for k in range(4) if k not in (i1, j1)]
    if len(remaining) != 2:
        return {"mZ1_GeV": np.nan, "mZ2_GeV": np.nan, "z1_pair": None, "z2_pair": None}

    i2, j2 = remaining
    t2_ok = (abs(row[f"lep_type_{i2}"]) == abs(row[f"lep_type_{j2}"])) and (abs(row[f"lep_type_{i2}"]) in (11, 13))
    q2_ok = (row[f"lep_charge_{i2}"] * row[f"lep_charge_{j2}"] == -1)

    if not (t2_ok and q2_ok):
        return {"mZ1_GeV": np.nan, "mZ2_GeV": np.nan, "z1_pair": (i1, j1), "z2_pair": None}

    mZ2 = _mll(row, i2, j2)  # MeV

    return {
        "mZ1_GeV": mZ1 / 1000.0,
        "mZ2_GeV": mZ2 / 1000.0,
        "z1_pair": (i1, j1),
        "z2_pair": (i2, j2),
    }


def add_zz_columns(df):
    """
    Add Z1/Z2 reconstructed masses (GeV) to a DataFrame.
    """
    out = df.apply(reconstruct_zz_row, axis=1, result_type="expand")
    df = df.copy()
    df["mZ1_GeV"] = out["mZ1_GeV"]
    df["mZ2_GeV"] = out["mZ2_GeV"]
    # Optional debugging columns:
    # df["z1_pair"] = out["z1_pair"]
    # df["z2_pair"] = out["z2_pair"]
    return df
