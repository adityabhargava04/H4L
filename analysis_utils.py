import numpy as np
import pandas as pd

MZ_GEV = 91.1876
MZ_MEV = MZ_GEV * 1000.0


def define_selections():
    """
    Define sequential event selection criteria for H → ZZ* → 4l analysis.
    Returns an ordered dictionary of selection functions.
    """
    selections = {}

    # exactly 4 reconstructed leptons
    selections["4_leptons"] = lambda df: (df["lep_n"] == 4)

    # pT ordering cuts (MeV)
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

    selections["has_ZZ"] = lambda df: np.isfinite(df["mZ1_GeV"]) & np.isfinite(df["mZ2_GeV"])
    selections["Z1_window"] = lambda df: np.abs(df["mZ1_GeV"] - MZ_GEV) < 15.0
    selections["Z2_min"] = lambda df: df["mZ2_GeV"] > 12.0

    return selections


def apply_selections(df, selection_fns, upto=None):
    """
    Apply selections sequentially and return cumulative boolean mask.
    """
    mask = np.ones(len(df), dtype=bool)
    for name, fn in selection_fns.items():
        mask &= fn(df)
        if name == upto:
            break
    return mask



def build_cutflow(df_data, df_bkg, df_sig, selection_fns):
    """
    Build a cutflow table with data counts and weighted MC yields.
    """
    rows = []

    mask_data = np.ones(len(df_data), dtype=bool)
    mask_bkg = np.ones(len(df_bkg), dtype=bool)
    mask_sig = np.ones(len(df_sig), dtype=bool)

    rows.append({
        "cut": "All events",
        "data": int(len(df_data)),
        "bkg": float(df_bkg["event_weight"].sum()),
        "sig": float(df_sig["event_weight"].sum()),
    })

    for name, fn in selection_fns.items():
        mask_data &= fn(df_data)
        mask_bkg &= fn(df_bkg)
        mask_sig &= fn(df_sig)

        rows.append({
            "cut": name,
            "data": int(mask_data.sum()),
            "bkg": float(df_bkg.loc[mask_bkg, "event_weight"].sum()),
            "sig": float(df_sig.loc[mask_sig, "event_weight"].sum()),
        })

    return pd.DataFrame(rows)


# ZZ* reconstruction

def _p4_from_lepton(row, i):
    pt = row[f"lep_pt_{i}"]
    eta = row[f"lep_eta_{i}"]
    phi = row[f"lep_phi_{i}"]
    E = row[f"lep_E_{i}"]

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return E, px, py, pz  # MeV

def _mll(row, i, j):
    Ei, pxi, pyi, pzi = _p4_from_lepton(row, i)
    Ej, pxj, pyj, pzj = _p4_from_lepton(row, j)

    E = Ei + Ej
    px = pxi + pxj
    py = pyi + pyj
    pz = pzi + pzj

    m2 = E * E - (px * px + py * py + pz * pz)
    return np.sqrt(max(m2, 0.0))  # MeV


def reconstruct_zz_row(row):
    """
    Reconstruct Z1 and Z2 using SFOS pairing.
    Returns NaNs if reconstruction fails.
    """
    for i in range(4):
        if not np.isfinite(row.get(f"lep_pt_{i}", np.nan)):
            return {"mZ1_GeV": np.nan, "mZ2_GeV": np.nan}

    pairs = []
    for i in range(4):
        ti = row[f"lep_type_{i}"]
        qi = row[f"lep_charge_{i}"]
        for j in range(i + 1, 4):
            tj = row[f"lep_type_{j}"]
            qj = row[f"lep_charge_{j}"]

            same_flavor = (abs(ti) == abs(tj)) and (abs(ti) in (11, 13))
            opp_sign = (qi * qj == -1)

            if same_flavor and opp_sign:
                pairs.append((i, j, _mll(row, i, j)))

    if len(pairs) == 0:
        return {"mZ1_GeV": np.nan, "mZ2_GeV": np.nan}

    pairs.sort(key=lambda x: abs(x[2] - MZ_MEV))
    i1, j1, mZ1 = pairs[0]

    remaining = [k for k in range(4) if k not in (i1, j1)]
    if len(remaining) != 2:
        return {"mZ1_GeV": np.nan, "mZ2_GeV": np.nan}

    i2, j2 = remaining
    t2_ok = (abs(row[f"lep_type_{i2}"]) == abs(row[f"lep_type_{j2}"])) and (abs(row[f"lep_type_{i2}"]) in (11, 13))
    q2_ok = (row[f"lep_charge_{i2}"] * row[f"lep_charge_{j2}"] == -1)

    if not (t2_ok and q2_ok):
        return {"mZ1_GeV": mZ1 / 1000.0, "mZ2_GeV": np.nan}

    mZ2 = _mll(row, i2, j2)

    return {
        "mZ1_GeV": mZ1 / 1000.0,
        "mZ2_GeV": mZ2 / 1000.0,
    }


def add_zz_columns(df):
    """
    Add reconstructed Z1 and Z2 invariant masses (GeV) to DataFrame.
    """
    out = df.apply(reconstruct_zz_row, axis=1, result_type="expand")
    df = df.copy()
    df["mZ1_GeV"] = out["mZ1_GeV"]
    df["mZ2_GeV"] = out["mZ2_GeV"]
    return df
