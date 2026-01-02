#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import boost_histogram as bh
import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import histogram_builder as hb
import mplhep as hep
from scipy.optimize import curve_fit

# Use a plotting style
#plt.style.use("fivethirtyeight")
hep.style.use(hep.style.ROOT)
np.seterr(invalid='ignore', divide='ignore')

target = "LH2_z0p67"


def plot_2D(config):
    runs_df = config['runs_df']

    x = []
    y = []
    for _, row in runs_df.iterrows():
        x.append(
            np.sqrt((row['P_gtr_p']**2) * (row['P_kin_secondary_th_xq']**2))
            * np.cos(row['P_kin_secondary_ph_xq'])
        )
        y.append(
            np.sqrt((row['P_gtr_p']**2) * (row['P_kin_secondary_th_xq']**2))
            * np.sin(row['P_kin_secondary_ph_xq'])
        )

    fig, ax = plt.subplots()

    h = ax.hist2d(
        x, y,
        bins=100,
        cmap="viridis"      # <-- colormap
    )

    plt.colorbar(h[3], ax=ax, label="Events")  # <-- colorbar
    ax.set_xlabel(r"$p_x$")
    ax.set_ylabel(r"$p_y$")

    plt.show()


def main():
    # Load run information
    runs_df = pd.read_csv(f"{target}.csv")
    csv_path = f"{target}.csv"
    # Al_thick = 0.095821  # g/cm2
    # dummy_thick = 0.3380  # g/cm2
    # R = Al_thick / dummy_thick
    # normfac = 0.148396E+12
    
    # 2D histogram
    config_2D = {
        'runs_df': runs_df,
        'var_x': "np.sqrt((row['P_gtr_p']**2) * (row['P_kin_secondary_th_xq']**2)) * np.cos(row['P_kin_secondary_ph_xq'])",
        'var_y': "np.sqrt((row['P_gtr_p']**2) * (row['P_kin_secondary_th_xq']**2)) * np.sin(row['P_kin_secondary_ph_xq'])",
    }
    hb.plot_2D(config_2D)


if __name__ == "__main__":
    main()
