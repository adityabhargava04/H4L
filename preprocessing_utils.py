import uproot
import pandas as pd

# Data Loading

def root_to_pandas(filename, branches=None, verbose=False):
    """
    Load the first TTree found in a ROOT file into a pandas DataFrame.
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

    return df

def load_many_root_files(file_list, label=None, **kwargs):
    dfs = []

    for fname in file_list:
        df = root_to_pandas(fname, **kwargs)
        df["source_file"] = fname
        if label is not None:
            df["sample"] = label
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)
