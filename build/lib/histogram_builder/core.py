# %%
import boost_histogram as bh
import uproot
import os
import csv
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
        (df["H_gtr_dp"] > -8) & (df["H_gtr_dp"] < 8) &
        (df["H_cal_etottracknorm"] > 0.7) &
        (df["H_cer_npeSum"] > 2) &
        (df["P_gtr_dp"] > -10) & (df["P_gtr_dp"] < 22) &
        (df["P_aero_npeSum"] > 2) &
        ((df["P_gtr_p"] <= 2.9) | (df["P_hgcer_npeSum"] > 1)) & # Or P>2.9 Gev
        (df["P_cal_etottracknorm"] < 0.8) &
        (abs(df["CTime_ePiCoinTime_ROC1"] - 51.24)< 2) 
        # & (df["pt"]>0.3) & (df["pt"]<0.4) # cut for p_T binning
    )

    return df[mask]

def random_cuts(df):

    mask = (
        (df["H_gtr_dp"] > -8) & (df["H_gtr_dp"] < 8) &
        (df["H_cal_etottracknorm"] > 0.7) &
        (df["H_cer_npeSum"] > 2) &
        (df["P_gtr_dp"] > -10) & (df["P_gtr_dp"] < 22) &
        (df["P_aero_npeSum"] > 2) &
        ((df["P_gtr_p"] <= 2.9) | (df["P_hgcer_npeSum"] > 1)) & # Or P>2.9 Gev
        (df["P_cal_etottracknorm"] < 0.8) &
        (df["CTime_ePiCoinTime_ROC1"]>30) & 
        (df["CTime_ePiCoinTime_ROC1"]<46) 
        # & (df["pt"]>0.3) & (df["pt"]<0.4) # cut for p_T binning
    )

    return df[mask]


def sim_cuts(df):
    """Apply physics cuts to DataFrame for SIMULATION."""
    mask = (
        (df["hsdelta"] > -8) & (df["hsdelta"] < 8) &
        (df["ssdelta"] > -10) & (df["ssdelta"] < 22) 
        # & (np.sqrt(df["pt2"])>0.3) & (np.sqrt(df["pt2"])<0.4) # cut for p_T binning
    )
    return df[mask]

def _get_eff(run_type: str, var: str, row) -> float:
    """Return tracking efficiency"""
    if var.startswith("H_"):
        return row.p_esing_Eff
    else:
        return row.h_esing_Eff

def _get_dead(run_type: str, var: str, row) -> float:
    """Return deadtime correction"""
    if var.startswith("H_"):
        return row.hDead
    else:
        return row.pDead

def fill_histogram_data(data, weights, bins, range, sf_sum):
    # define bin edges
    edges = np.linspace(range[0], range[1], bins + 1)
    sum_counts = np.zeros(bins)
    sum_variances = np.zeros(bins)

    for run_data, run_weights in zip(data, weights):
        hist = bh.Histogram(bh.axis.Variable(edges), storage=bh.storage.Weight())
        hist.fill(run_data, weight=run_weights)
        view = hist.view()
        sum_counts += view['value']
        sum_variances += view['variance']

    bin_centers = 0.5*(edges[:-1] + edges[1:])
    counts = sum_counts / sf_sum
    errors = np.sqrt(sum_variances) / np.abs(sf_sum)

    return bin_centers, counts, errors



def fill_histogram_sim(data, weights, bins, range, sf_sum):
    """Fill histogram with raw counts and normalize by sf_sum.
    Returns histogram, counts, and errors."""

    hist = bh.Histogram(bh.axis.Regular(bins, range[0],range[1]), storage=bh.storage.Weight())
    hist.fill(data, weight=weights)
    hist_view = hist.view()
    counts = hist_view['value']
    counts_norm = counts / sf_sum
    variances = hist_view['variance']
    errors = np.sqrt(variances) / np.abs(sf_sum)
    bin_centers = hist.axes[0].centers

    return bin_centers, counts_norm, errors

# =========================
# LOAD AND PROCESS DATA
# =========================

def process_runs(runs_df, R, normfac, var_to_plot, run_type):
    """Load all runs of a given type, apply cuts, return concatenated variable & weights."""
    subset = runs_df[runs_df["run_type"] == run_type]
    all_data = []
    all_weights = []

    if run_type in ["data", "dummy"]:
        sf_sum = 0
        branches = [
            "H_gtr_dp", "H_cal_etottracknorm", "H_cer_npeSum", "P_gtr_p",
            "P_gtr_dp", "P_cal_etottracknorm", "P_ngcer_npeSum", "P_hgcer_npeSum", "P_aero_npeSum",
            "CTime_ePiCoinTime_ROC1", "H_gtr_y", "H_kin_primary_x_bj", "H_kin_primary_nu",
            "P_kin_secondary_ph_xq", "P_kin_secondary_th_xq", "pt"
        ]
        for _, row in subset.iterrows():
            df = load_branch(row.filename, branches, row.run_type)
            df_cut = data_cuts(df).copy()
            expr = var_to_plot[0]  
            # Include derived variables 
            df_cut["custom_var"] = df_cut.apply(lambda row: eval(expr, {"np": np, "row": row}),axis=1)
            colname = "custom_var"

             # raw yields → weight = 1
            all_data.append(df_cut[colname].to_numpy())
            all_weights.append(np.ones(len(df_cut)))

            # accumulate denominator
            eff = _get_eff(row.run_type, expr, row)
            dead = _get_dead(row.run_type, expr, row)
            denom = row.BCM2_Q * eff * dead
            if denom > 0:
                sf_sum +=  denom
        return all_data, np.concatenate(all_weights), sf_sum

    elif run_type == "sim":
        sf_sum = 1.0 # No scaling factor for sim
        branches = [
            "hsdelta", "ssdelta", "Weight", "hsytar", "ssytar", "xbj", "nu", "Q2", "z", "phipq", "thetapq", "pt2"
        ]
        for _, row in subset.iterrows():
            df = load_branch(row.filename, branches, row.run_type)
            df_cut = sim_cuts(df).copy()
            # Normalize sim weights
            # w = df_cut["Weight"].to_numpy() * normfac / len(df_cut)
            w = df_cut["Weight"].to_numpy() * normfac / 1000000  # Normalize to 1M events instead of actual number of events
            # Evaluate derived variable for simulation
            expr = var_to_plot[1]
            df_cut["custom_var"] = df_cut.apply(lambda r: eval(expr, {"np": np, "row": r}),axis=1)
            all_data.append(df_cut["custom_var"].to_numpy())
            all_weights.append(w)      
        return np.concatenate(all_data), np.concatenate(all_weights), sf_sum

def process_random(runs_df, R, normfac, var_to_plot,run_type):
    """Load all runs of a given type, apply cuts, return concatenated variable & weights."""
    subset = runs_df[runs_df["run_type"] == run_type]
    all_data = []
    all_weights = []

    if run_type in ["data", "dummy"]:
        sf_sum = 0.0
        branches = [
            "H_gtr_dp", "H_cal_etottracknorm", "H_cer_npeSum", "P_gtr_p",
            "P_gtr_dp", "P_cal_etottracknorm", "P_ngcer_npeSum", "P_hgcer_npeSum", "P_aero_npeSum",
            "CTime_ePiCoinTime_ROC1", "H_gtr_y", "H_kin_primary_x_bj", "H_kin_primary_nu",
            "P_kin_secondary_ph_xq", "P_kin_secondary_th_xq", "pt"
        ]
        for _, row in subset.iterrows():
            df = load_branch(row.filename, branches, row.run_type)
            df_cut = random_cuts(df).copy()
            expr = var_to_plot[0]  
            # Include derived variables 
            df_cut["custom_var"] = df_cut.apply(lambda row: eval(expr, {"np": np, "row": row}),axis=1)
            colname = "custom_var"

            # raw yields → weight = 1
            all_data.append(df_cut[colname].to_numpy())
            all_weights.append(np.ones(len(df_cut)))

            # accumulate denominator
            eff = _get_eff(row.run_type, expr, row)
            dead = _get_dead(row.run_type, expr, row)
            denom = row.BCM2_Q * eff * dead
            if denom > 0:
                sf_sum += denom

        return all_data, np.concatenate(all_weights), sf_sum


def make_histograms_solid(runs_df, R, normfac, var_to_plot,bins, range):

    # Process all runs
    data_vals, data_weights, data_sf = process_runs(runs_df, R, normfac, var_to_plot, "data")
    random_vals, random_weights, random_sf = process_random(runs_df, R, normfac, var_to_plot, "data")
    sim_vals, sim_weights, sim_sf = process_runs(runs_df, R, normfac, var_to_plot, "sim")

    # def fill_histogram(data, weights, bins, range, sf_sum)

    # Fill histograms
    centers_data, counts_data, errors_data = fill_histogram_data(data_vals, data_weights, bins, range, data_sf)
    centers_random, counts_random, errors_random = fill_histogram_data(random_vals, (1/4) * random_weights, bins, range, random_sf)
    centers_sim, counts_sim, errors_sim = fill_histogram_sim(sim_vals, sim_weights, bins, range, sim_sf)

    # Dummy target histograms

    # dummy_random_vals, dummy_random_weights, dummy_random_sf = process_random(runs_df, R, normfac, var_to_plot, "dummy")
    # centers_dummy_tot, counts_dummy_tot, errors_dummy_tot = fill_histogram_data(dummy_vals, R * dummy_weights, bins, range, dummy_sf)
    # centers_dummy_random, counts_dummy_random, errors_dummy_random = fill_histogram_data(dummy_random_vals, (1/4) * R * dummy_random_weights, bins, range, dummy_random_sf)

    # centers_dummy = centers_dummy_tot
    # counts_dummy = counts_dummy_tot - counts_dummy_random
    # errors_dummy = np.sqrt(errors_dummy_tot**2 + errors_dummy_random**2)

     # Subtracted histogram

    centers_sub = centers_data
    # For cryotargets:
    # counts_sub = counts_data - counts_dummy - counts_random 
    # errors_sub = np.sqrt(errors_data**2 + errors_dummy**2 + errors_random**2)

    # For solid targets:
    counts_sub = counts_data  - counts_random 
    errors_sub = np.sqrt(errors_data**2  + errors_random**2)

    yield_sim = float(np.sum(counts_sim))
    yield_random = float(np.sum(counts_random))
    yield_sub = float(np.sum(counts_sub))
    yield_data = float(np.sum(counts_data))

    yield_err_random = np.sqrt(np.sum(errors_random**2))
    yield_err_sim = np.sqrt(np.sum(errors_sim**2))
    yield_err_sub = np.sqrt(np.sum(errors_sub**2))
    yield_err_data = np.sqrt(np.sum(errors_data**2))


    return {
        "centers_data": centers_data,
        "counts_data": counts_data,
        "errors_data": errors_data,
        # "centers_dummy": centers_dummy,
        # "counts_dummy": counts_dummy,
        # "errors_dummy": errors_dummy,
        "centers_sim": centers_sim,
        "counts_sim": counts_sim,
        "errors_sim": errors_sim,
        "centers_sub": centers_sub,
        "counts_sub": counts_sub,
        "errors_sub": errors_sub,
        "centers_random": centers_random,
        "counts_random": counts_random,
        "errors_random": errors_random,
        "yield_sim": yield_sim,
        "yield_err_sim": yield_err_sim,
        "yield_random": yield_random,
        "yield_err_random": yield_err_random,
        "yield_data": yield_data,
        "yield_err_data": yield_err_data,
        "yield_sub": yield_sub,
        "yield_err_sub": yield_err_sub
    }

def make_histograms_cryo(runs_df, R, normfac, var_to_plot, bins, range):

    # Process all runs
    data_vals, data_weights, data_sf = process_runs(runs_df, R, normfac, var_to_plot, "data")
    dummy_vals, dummy_weights, dummy_sf = process_runs(runs_df, R, normfac, var_to_plot, "dummy")
    random_vals, random_weights, random_sf = process_random(runs_df, R, normfac, var_to_plot, "data")
    sim_vals, sim_weights, sim_sf = process_runs(runs_df, R, normfac, var_to_plot, "sim")

    # def fill_histogram(data, weights, bins, range, sf_sum)

    # Fill histograms
    centers_data, counts_data, errors_data = fill_histogram_data(data_vals, data_weights, bins, range, data_sf)
    centers_random, counts_random, errors_random = fill_histogram_data(random_vals, (1/4) * random_weights, bins, range, random_sf)
    centers_sim, counts_sim, errors_sim = fill_histogram_sim(sim_vals, sim_weights, bins, range, sim_sf)

    # Dummy target histograms

    dummy_random_vals, dummy_random_weights, dummy_random_sf = process_random(runs_df, R, normfac, var_to_plot, "dummy")
    centers_dummy_tot, counts_dummy_tot, errors_dummy_tot = fill_histogram_data(dummy_vals, R * dummy_weights, bins, range, dummy_sf)
    centers_dummy_random, counts_dummy_random, errors_dummy_random = fill_histogram_data(dummy_random_vals, (1/4) * R * dummy_random_weights, bins, range, dummy_random_sf)

    centers_dummy = centers_dummy_tot
    counts_dummy = counts_dummy_tot - counts_dummy_random
    errors_dummy = np.sqrt(errors_dummy_tot**2 + errors_dummy_random**2)

     # Subtracted histogram

    centers_sub = centers_data
    # For cryotargets:
    counts_sub = counts_data - counts_dummy - counts_random 
    errors_sub = np.sqrt(errors_data**2 + errors_dummy**2 + errors_random**2)


    yield_sim = float(np.sum(counts_sim))
    yield_random = float(np.sum(counts_random))
    yield_dummy = float(np.sum(counts_dummy))
    yield_sub = float(np.sum(counts_sub))
    yield_data = float(np.sum(counts_data))

    yield_err_random = np.sqrt(np.sum(errors_random**2))
    yield_err_sim = np.sqrt(np.sum(errors_sim**2))
    yield_err_sub = np.sqrt(np.sum(errors_sub**2))
    yield_err_dummy = np.sqrt(np.sum(errors_dummy**2))
    yield_err_data = np.sqrt(np.sum(errors_data**2))


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
        "errors_random": errors_random,
        "yield_sim": yield_sim,
        "yield_err_sim": yield_err_sim,
        "yield_random": yield_random,
        "yield_dummy": yield_dummy,
        "yield_err_random": yield_err_random,
        "yield_err_dummy": yield_err_dummy,
        "yield_data": yield_data,
        "yield_err_data": yield_err_data,
        "yield_sub": yield_sub,
        "yield_err_sub": yield_err_sub
    }

def make_histograms_2D(runs_df, var_x, var_y, run_type="data"):
    subset = runs_df[runs_df["run_type"] == run_type]
    all_x, all_y = [], []
    
    for _, row in subset.iterrows():
        branches = [
            "H_gtr_dp", "H_cal_etottracknorm", "H_cer_npeSum", "P_gtr_p",
            "P_gtr_dp", "P_cal_etottracknorm", "P_ngcer_npeSum", "P_hgcer_npeSum", "P_aero_npeSum",
            "CTime_ePiCoinTime_ROC1", "H_gtr_y", "H_kin_primary_x_bj", "H_kin_primary_nu",
            "P_kin_secondary_ph_xq", "P_kin_secondary_th_xq", "pt"
        ]
        df = load_branch(row.filename, branches, row.run_type)

        if run_type == "data":
            df_cut = data_cuts(df).copy()
        elif run_type == "sim":
            df_cut = sim_cuts(df).copy()
        else:
            df_cut = df.copy()

        # 3. Evaluar las expresiones en pandas
        expr_x = var_x
        expr_y = var_y
        df_cut["xvar"] = df_cut.apply(lambda row: eval(expr_x, {"np": np, "row": row}),axis=1)
        df_cut["yvar"] = df_cut.apply(lambda row: eval(expr_y, {"np": np, "row": row}),axis=1)

        all_x.append(df_cut["xvar"].to_numpy())
        all_y.append(df_cut["yvar"].to_numpy())

    return np.concatenate(all_x), np.concatenate(all_y)


def save_ratio_to_csv(histo, ratio_valid, ratio_err_valid, variable, csv_path):

    # Define output file name
    base_dir = os.path.dirname(csv_path) if csv_path else os.getcwd()
    target_name = os.path.splitext(os.path.basename(csv_path))[0] if csv_path else "ratios"
    output_file = os.path.join(base_dir, f"{target_name}_ratios.csv")
    
    data = {
        "bin_center": np.round(histo, 4),
        "ratio": np.round(ratio_valid, 6),
        "ratio_err": np.round(ratio_err_valid, 6),
        "variable": [variable] * len(ratio_valid),
    }

    df = pd.DataFrame(data)

    # Append or create the CSV
    if not os.path.exists(output_file):
        df.to_csv(output_file, index=False)
        print(f"[INFO] Created new file: {output_file}")
    else:
        df.to_csv(output_file, index=False, mode='a', header=False)
        print(f"[INFO] Appended {len(df)} rows to {output_file}")


def plot_histogram_solid(config, xlabel, title="", csv_path=None):
    histo = make_histograms_solid(**config)
    target_name = os.path.splitext(os.path.basename(csv_path))[0]

    print(f"Data yield:    {histo['yield_data']:.2f} ± {histo['yield_err_data']:.2f}")
    print(f"Sim yield:     {histo['yield_sim']:.2f} ± {histo['yield_err_sim']:.2f}")
    print(f"Random yield:  {histo['yield_random']:.2f} ± {histo['yield_err_random']:.2f}")
    print(f"Sub yield:     {histo['yield_sub']:.2f} ± {histo['yield_err_sub']:.2f}")
    print(f"Ratio (Data/Sim): {histo['yield_sub']/histo['yield_sim']:.3f}")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(28, 17),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True
    )

    colors = plt.cm.tab10.colors

    # Histograms plots
    ax1.errorbar(histo["centers_data"], histo["counts_data"], yerr=np.abs(histo["errors_data"]),
                 fmt='o', color=colors[0], label=f'{target_name} data', capsize=4, elinewidth=2, markeredgewidth=2)
    ax1.errorbar(histo["centers_sim"], histo["counts_sim"], yerr=np.abs(histo["errors_sim"]),
                 fmt='s', color=colors[1], label='SIMC', capsize=4, elinewidth=2, markeredgewidth=2)
    ax1.errorbar(histo["centers_random"], histo["counts_random"], yerr=np.abs(histo["errors_random"]),
                 fmt='o', color=colors[2], label="Random coincidences", capsize=4, elinewidth=2, markeredgewidth=2)
    ax1.errorbar(histo["centers_sub"], histo["counts_sub"], yerr=np.abs(histo["errors_sub"]),
                 fmt='o', color=colors[4], label="Random-subtracted data", capsize=4, elinewidth=2, markeredgewidth=2)

    ax1.set_ylabel("Normalized yield (counts/mC)", fontsize=35)
    ax1.set_ylim(0, max(histo["counts_data"]) * 1.2)
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax1.minorticks_on()
    for spine in ax1.spines.values():
        spine.set_linewidth(3)
        spine.set_color('black')
    ax1.tick_params(which='major', length=15, width=3, direction='in', bottom=True, top=True,
                    left=True, right=True, labelsize=25)
    ax1.tick_params(which='minor', length=8, width=1, direction='in', bottom=True, top=True,
                    left=True, right=True)
    ax1.set_title(title, fontsize=40)
    ax1.legend(fontsize=22)
    ax1.grid(True, which="both", linestyle="--", linewidth=1, alpha=0.4)

    # Ratio plot
    ratio = histo["counts_sub"] / histo["counts_sim"]
    ratio_err = ratio * np.sqrt((histo["errors_sub"] / histo["counts_sub"]) ** 2 +
                                (histo["errors_sim"] / histo["counts_sim"]) ** 2)

    mask = (np.isfinite(ratio) & (ratio_err>0)  & (ratio_err<0.25))

    centers_mask = np.array(histo["centers_sub"])[mask]
    ratio_mask = np.array(ratio)[mask]
    ratio_err_mask = np.array(ratio_err)[mask]

    save_ratio_to_csv(centers_mask, ratio_mask, ratio_err_mask, xlabel, csv_path)

    ax2.errorbar(centers_mask, ratio_mask, yerr=np.abs(ratio_err_mask),
                 fmt='o', color=colors[5], label="Data/SIMC", capsize=4, elinewidth=2, markeredgewidth=2)
    ax2.axhline(1.0, color='black', linestyle='--')
    ax2.set_xlabel(xlabel, fontsize=35)
    ax2.set_ylabel("Data/Sim", fontsize=30)
    ax2.set_ylim(0.5, 1.5)
    ax2.minorticks_on()
    for spine in ax2.spines.values():
        spine.set_linewidth(3)
        spine.set_color('black')
    ax2.tick_params(which='major', length=12, width=2, direction='in',
                    bottom=True, top=True, left=True, right=True, labelsize=25)
    ax2.tick_params(which='minor', length=6, width=1, direction='in',
                    bottom=True, top=True, left=True, right=True)
    ax2.grid(True, which="both", linestyle="--", linewidth=1, alpha=0.4)
    ax2.legend(fontsize=25)

    plt.tight_layout()
    plt.show()

def plot_histogram_cryo(config, xlabel, title="", csv_path=None):
    histo = make_histograms_cryo(**config)
    target_name = os.path.splitext(os.path.basename(csv_path))[0]

    print(f"Data yield:    {histo['yield_data']:.2f} ± {histo['yield_err_data']:.2f}")
    print(f"Sim yield:     {histo['yield_sim']:.2f} ± {histo['yield_err_sim']:.2f}")
    print(f"Dummy yield:     {histo['yield_dummy']:.2f} ± {histo['yield_err_dummy']:.2f}")
    print(f"Random yield:  {histo['yield_random']:.2f} ± {histo['yield_err_random']:.2f}")
    print(f"Sub yield:     {histo['yield_sub']:.2f} ± {histo['yield_err_sub']:.2f}")
    print(f"Ratio (Data/Sim): {histo['yield_sub']/histo['yield_sim']:.3f}")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(28, 17),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True
    )

    colors = plt.cm.tab10.colors

    # Histograms plots
    ax1.errorbar(histo["centers_data"], histo["counts_data"], yerr=np.abs(histo["errors_data"]),
                 fmt='o', color=colors[0], label=f'{target_name} data', capsize=4, elinewidth=2, markeredgewidth=2)
    ax1.errorbar(histo["centers_sim"], histo["counts_sim"], yerr=np.abs(histo["errors_sim"]),
                 fmt='s', color=colors[1], label='SIMC', capsize=4, elinewidth=2, markeredgewidth=2)
    ax1.errorbar(histo["centers_random"], histo["counts_random"], yerr=np.abs(histo["errors_random"]),
                 fmt='o', color=colors[2], label="Random coincidences", capsize=4, elinewidth=2, markeredgewidth=2)
    ax1.errorbar(histo["centers_dummy"], histo["counts_dummy"], yerr=np.abs(histo["errors_dummy"]),
                 fmt='o', color=colors[6], label="Dummy data", capsize=4, elinewidth=2, markeredgewidth=2)
    ax1.errorbar(histo["centers_sub"], histo["counts_sub"], yerr=np.abs(histo["errors_sub"]),
                 fmt='o', color=colors[4], label="Random-subtracted data", capsize=4, elinewidth=2, markeredgewidth=2)

    ax1.set_ylabel("Normalized yield (counts/mC)", fontsize=35)
    ax1.set_ylim(0, max(histo["counts_data"]) * 1.2)
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax1.minorticks_on()
    for spine in ax1.spines.values():
        spine.set_linewidth(3)
        spine.set_color('black')
    ax1.tick_params(which='major', length=15, width=3, direction='in', bottom=True, top=True,
                    left=True, right=True, labelsize=25)
    ax1.tick_params(which='minor', length=8, width=1, direction='in', bottom=True, top=True,
                    left=True, right=True)
    ax1.set_title(title, fontsize=40)
    ax1.legend(fontsize=22)
    ax1.grid(True, which="both", linestyle="--", linewidth=1, alpha=0.4)

    # Ratio plot
    ratio = histo["counts_sub"] / histo["counts_sim"]
    ratio_err = ratio * np.sqrt((histo["errors_sub"] / histo["counts_sub"]) ** 2 +
                                (histo["errors_sim"] / histo["counts_sim"]) ** 2)

    mask = (np.isfinite(ratio) & (ratio_err>0)  & (ratio_err<0.25))

    centers_mask = np.array(histo["centers_sub"])[mask]
    ratio_mask = np.array(ratio)[mask]
    ratio_err_mask = np.array(ratio_err)[mask]

    save_ratio_to_csv(centers_mask, ratio_mask, ratio_err_mask, xlabel, csv_path)

    ax2.errorbar(centers_mask, ratio_mask, yerr=np.abs(ratio_err_mask),
                 fmt='o', color=colors[5], label="Data/SIMC", capsize=4, elinewidth=2, markeredgewidth=2)
    ax2.axhline(1.0, color='black', linestyle='--')
    ax2.set_xlabel(xlabel, fontsize=35)
    ax2.set_ylabel("Data/Sim", fontsize=30)
    ax2.set_ylim(0.5, 1.5)
    ax2.minorticks_on()
    for spine in ax2.spines.values():
        spine.set_linewidth(3)
        spine.set_color('black')
    ax2.tick_params(which='major', length=12, width=2, direction='in',
                    bottom=True, top=True, left=True, right=True, labelsize=25)
    ax2.tick_params(which='minor', length=6, width=1, direction='in',
                    bottom=True, top=True, left=True, right=True)
    ax2.grid(True, which="both", linestyle="--", linewidth=1, alpha=0.4)
    ax2.legend(fontsize=25)

    plt.tight_layout()
    plt.show()


def plot_2D(config, bins=100, range_=[(-1, 1), (-1, 1)]):
    x, y = make_histograms_2D(**config)

    plt.figure(figsize=(17, 17))

    cmap = plt.cm.plasma.copy()
    cmap.set_under("#f0f0f0", alpha=0)
    h = plt.hist2d(x, y, bins=bins, range=range_, cmap=cmap, vmin=1, alpha=1)
    plt.colorbar(h[3], ax=plt.gca(), label="Events") # Add for colorbar


    ax = plt.gca()
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)

    # Concentric arcs
    for r in [0.2, 0.4, 0.6, 0.8]:
        circ = mpl.patches.Circle((0, 0), radius=r, fill=False,
                                  color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.add_patch(circ)

    # Arc labels
    pT_values = [r"$p_T=0.2$", r"$p_T=0.4$", r"$p_T=0.6$", r"$p_T=0.8$"]
    arcs = [(np.sqrt(2) * 0.5 * r) for r in [0.2, 0.4, 0.6, 0.8]]
    for xi, yi, label in zip(arcs, arcs, pT_values):
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(2, 2),
                    ha='left', fontsize=15, color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#f0f0f0", ec="none", alpha=0.7))

    # Compass labels
    ax.text(1.1, 0, r"$0^\circ$", va="center", ha="left", fontsize=20)
    ax.text(-1.1, 0, r"$180^\circ$", va="center", ha="right", fontsize=20)
    ax.text(0, 1.01, r"$90^\circ$", va="bottom", ha="center", fontsize=20)
    ax.text(0, -1.08, r"$270^\circ$", va="top", ha="center", fontsize=20)

    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.minorticks_on()
    plt.tick_params(which='major', length=12, width=2, direction='in',
                    bottom=True, top=True, left=True, right=True)
    plt.tick_params(which='minor', length=6, width=1, direction='in',
                    bottom=True, top=True, left=True, right=True)
    plt.setp(ax.spines.values(), linewidth=3, color='black')
    plt.title(r"$p_T$ vs $\phi$", fontsize=40, pad=15, color="black")
    plt.axis("equal")
    ax.set_axisbelow(False)
    ax.grid(True, linestyle=":", linewidth=0.8, color="gray", alpha=0.5)

    plt.tight_layout()
    plt.show()


# %%
