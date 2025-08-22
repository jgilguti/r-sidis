# %%
import boost_histogram as bh
import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# =========================
# HELPER FUNCTIONS
# =========================
def load_branch(filename, branches, run_type):
    with uproot.open(filename) as file:
        if run_type == "sim":
            tree = file["h10"]  
        else:
            tree = file["T"]     
        data = {br: tree[br].array(library="np") for br in branches}
    return pd.DataFrame(data)


def data_cuts(df):
    mask = (
        (df["H.gtr.dp"] > -8) & (df["H.gtr.dp"] < 8) &
        (df["H.cal.etottracknorm"] > 0.8) &
        (df["P.gtr.dp"] > -10) & (df["P.gtr.dp"] < 22) &
        (df["P.ngcer.npeSum"] < 1) &
        (df["P.cal.etottracknorm"] < 0.4) &
        (df["CTime.ePiCoinTime_ROC1"]>50) & 
        (df["CTime.ePiCoinTime_ROC1"]<52)
    )
    return df[mask]

def random_cuts(df):
    mask = (
        (df["H.gtr.dp"] > -8) & (df["H.gtr.dp"] < 8) &
        (df["H.cal.etottracknorm"] > 0.8) &
        (df["P.gtr.dp"] > -10) & (df["P.gtr.dp"] < 22) &
        (df["P.ngcer.npeSum"] < 1) & 
        (df["P.cal.etottracknorm"] < 0.4) &
        (df["CTime.ePiCoinTime_ROC1"]>26) & 
        (df["CTime.ePiCoinTime_ROC1"]<46)
    )
    return df[mask]


def sim_cuts(df):
    """Apply physics cuts to DataFrame for SIMULATION."""
    mask = (
        (df["hsdelta"] > -8) & (df["hsdelta"] < 8) &
        (df["ssdelta"] > -10) & (df["ssdelta"] < 22)
    )
    return df[mask]


def fill_histogram(data, weights, bins, range):
    """Fill a boost-histogram and return counts and errors."""
    hist = bh.Histogram(
        bh.axis.Regular(bins, range[0],range[1]),
        storage=bh.storage.Weight()
        )
    hist.fill(data, weight=weights)
    hist_view = hist.view()
    counts = hist_view['value']
    errors = np.sqrt(hist_view['variance'])
    bin_centers = hist.axes[0].centers
    
    return bin_centers, counts, errors

# =========================
# LOAD AND PROCESS DATA
# =========================

def process_runs(runs_df, R, normfac, var_to_plot, run_type):
    """Load all runs of a given type, apply cuts, return concatenated variable & weights."""
    subset = runs_df[runs_df["run_type"] == run_type]
    all_data = []
    all_weights = []

    if run_type in ["data", "dummy"]:
        branches = [
            "H.gtr.dp", "H.cal.etottracknorm", "H.cer.npeSum",
            "P.gtr.dp", "P.cal.etottracknorm", "P.ngcer.npeSum", "P.hgcer.npeSum",
            "CTime.ePiCoinTime_ROC1", "H.gtr.y", "P.gtr.y", "H.kin.primary.x_bj",
            "H.kin.primary.Q2", "P.gtr.p", "H.kin.primary.nu"
        ]
        for _, row in subset.iterrows():
            df = load_branch(row.filename, branches, row.run_type)
            df_cut = data_cuts(df)
            expr = var_to_plot[0]  
            # Include derived variables 
            if "/" in expr: 
                left, right = [v.strip() for v in expr.split("/")]
                df_cut["custom_var"] = df_cut[left] / df_cut[right]
                colname = "custom_var"
            else:
                colname = expr
            all_data.append(df_cut[colname].to_numpy())
            all_weights.append(np.full(len(df_cut), row.weight_factor))

    elif run_type == "sim":
        branches = [
            "hsdelta", "ssdelta", "Weight", "hsytar", "ssytar", "xbj", "nu", "Q2", "z"
        ]
        for _, row in subset.iterrows():
            df = load_branch(row.filename, branches, row.run_type)
            df_cut = sim_cuts(df)
            # Normalize sim weights
            w = df_cut["Weight"].to_numpy() * normfac / len(df_cut)
            all_data.append(df_cut[var_to_plot[1]].to_numpy())
            all_weights.append(w)      
    return np.concatenate(all_data), np.concatenate(all_weights)

def process_random(runs_df, R, normfac, var_to_plot,run_type):
    """Load all runs of a given type, apply cuts, return concatenated variable & weights."""
    subset = runs_df[runs_df["run_type"] == run_type]
    all_data = []
    all_weights = []

    if run_type in ["data", "dummy"]:
        branches = [
            "H.gtr.dp", "H.cal.etottracknorm", "H.cer.npeSum",
            "P.gtr.dp", "P.cal.etottracknorm", "P.ngcer.npeSum", "P.hgcer.npeSum",
            "CTime.ePiCoinTime_ROC1", "H.gtr.y", "P.gtr.y", "H.kin.primary.x_bj",
            "H.kin.primary.Q2", "P.gtr.p", "H.kin.primary.nu"
        ]
        for _, row in subset.iterrows():
            df = load_branch(row.filename, branches, row.run_type)
            df_cut = random_cuts(df)
            expr = var_to_plot[0]  
            # Include derived variables 
            if "/" in expr: 
                left, right = [v.strip() for v in expr.split("/")]
                df_cut["custom_var"] = df_cut[left] / df_cut[right]
                colname = "custom_var"
            else:
                colname = expr
            all_data.append(df_cut[colname].to_numpy())
            all_weights.append(np.full(len(df_cut), row.weight_factor))      
    return np.concatenate(all_data), np.concatenate(all_weights)


def make_histograms(runs_df, R, normfac, var_to_plot,bins, range):

    # Process all runs
    data_vals, data_weights = process_runs(runs_df, R, normfac, var_to_plot, "data")
    dummy_vals, dummy_weights = process_runs(runs_df, R, normfac, var_to_plot, "dummy")
    random_vals, random_weights = process_random(runs_df, R, normfac, var_to_plot, "data")
    sim_vals, sim_weights = process_runs(runs_df, R, normfac, var_to_plot, "sim")

    # Fill histograms
    centers_data, counts_data, errors_data = fill_histogram(data_vals, data_weights, bins, range)
    centers_dummy, counts_dummy, errors_dummy = fill_histogram(dummy_vals, dummy_weights, bins, range)
    centers_random, counts_random, errors_random = fill_histogram(random_vals, random_weights, bins, range)
    centers_sim, counts_sim, errors_sim = fill_histogram(sim_vals, sim_weights, bins, range)

    # Vectorized dummy subtraction
    counts_sub = counts_data - R * counts_dummy - (1/5) * counts_random
    errors_sub = np.sqrt(errors_data**2 + (R * errors_dummy)**2 + ((1/5) * errors_random)**2)
    centers_sub = centers_data

    return {
        "centers_data": centers_data,
        "counts_data": counts_data,
        "errors_data": errors_data,
        "centers_dummy": centers_dummy,
        "counts_dummy": counts_dummy,
        "errors_dummy": errors_dummy,
        "centers_sim": centers_sim,
        "counts_sim": counts_sim,
        "errors_sim": errors_sim,
        "centers_sub": centers_sub,
        "counts_sub": counts_sub,
        "errors_sub": errors_sub,
        "centers_random": centers_random,
        "counts_random": counts_random,
        "errors_random": errors_random
    }

# %%
