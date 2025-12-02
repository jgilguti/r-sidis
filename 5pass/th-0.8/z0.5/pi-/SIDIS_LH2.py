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

target = "LH2"


def main():
    # Load run information
    runs_df = pd.read_csv(f"{target}.csv")
    csv_path = f"{target}.csv"
    Al_thick = 0.095821  # g/cm2
    dummy_thick = 0.3380  # g/cm2
    R = Al_thick / dummy_thick
    normfac = 0.144594E+12
    
    # Original z histogram
    config_z = {
        'runs_df': runs_df,
        'R': R,
        'normfac': normfac,
        'var_to_plot': ["row['P_gtr_p'] / row['H_kin_primary_nu']", "row['z']"],
        'bins': 100,
        'range': (0, 1.0)
    }
    hb.plot_histogram_cryo(config_z, xlabel="z", title=r"$z$ distribution", csv_path=csv_path)

    # θ_pq histogram
    config_theta = {
        'runs_df': runs_df,
        'R': R,
        'normfac': normfac,
        'var_to_plot': ["np.rad2deg(row['P_kin_secondary_th_xq'])", "np.rad2deg(row['thetapq'])"],
        'bins': 120,
        'range': (-2.0, 10)
    }
    hb.plot_histogram_cryo(config_theta, xlabel=r"$\theta_{pq}$", title=r"$θ_{pq}$ distribution", csv_path=csv_path)

    # pT histogram
    config_pt = {
        'runs_df': runs_df,
        'R': R,
        'normfac': normfac,
        'var_to_plot': [
            "np.sqrt((row['P_gtr_p']**2) * (1 - (np.cos(row['P_kin_secondary_th_xq']))**2))",
            "np.sqrt(row['pt2'])"
        ],
        'bins': 100,
        'range': (0, 1.0)
    }
    hb.plot_histogram_cryo(config_pt, xlabel=r"$p_T$", title=r"$p_T$ distribution", csv_path=csv_path)

    # phi_pq histogram
    config_phi = {
        'runs_df': runs_df,
        'R': R,
        'normfac': normfac,
        'var_to_plot': ["np.mod(row['P_kin_secondary_ph_xq'], 2*np.pi)", "row['phipq']"],
        'bins': 63,
        'range': (0, 2 * np.pi)
    }
    hb.plot_histogram_cryo(config_phi, xlabel=r"$\phi_{pq}$", title=r"$\phi_{pq}$ distribution", csv_path=csv_path)

    # 2D histogram
    config_2D = {
        'runs_df': runs_df,
        'var_x': "np.sqrt((row['P_gtr_p']**2) * (row['P_kin_secondary_th_xq']**2)) * np.cos(row['P_kin_secondary_ph_xq'])",
        'var_y': "np.sqrt((row['P_gtr_p']**2) * (row['P_kin_secondary_th_xq']**2)) * np.sin(row['P_kin_secondary_ph_xq'])",
    }
    hb.plot_2D(config_2D)


if __name__ == "__main__":
    main()
