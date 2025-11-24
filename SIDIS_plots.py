#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import runpy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys

def construct_script_path(pass_energy, th_value, z_value, polarity, target):
    # Determine the path of the plots script
    return os.path.join(
        f"{pass_energy}pass",
        f"th{th_value}",
        f"z{z_value}",
        f"pi{polarity}",
        f"SIDIS_{target}.py"
    )

def run_script_and_save_figs(script_path, pdf_filename, prevent_show=True, prevent_close=False):
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    plt.close("all")

    # Monkeypatch show/close to prevent interactive windows
    orig_show, orig_close = plt.show, plt.close
    if prevent_show:
        plt.show = lambda *a, **k: None
    if prevent_close:
        plt.close = lambda *a, **k: None

    saved_count = 0
    cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(script_path))
    script_file = os.path.basename(script_path)

    try:
        # Run the analysis script in its own directory
        os.chdir(script_dir)
        runpy.run_path(script_file, run_name="__main__")

        fignums = plt.get_fignums()
        if fignums:
            # Ensure output directory exists (relative to cwd where this script was run)
            os.makedirs(os.path.dirname(pdf_filename), exist_ok=True)

            with PdfPages(pdf_filename) as pdf:
                for num in fignums:
                    fig = plt.figure(num)
                    pdf.savefig(fig)
                    plt.close(fig)
                    saved_count += 1
    finally:
        plt.show, plt.close = orig_show, orig_close
        os.chdir(cwd)

    if saved_count == 0 and os.path.exists(pdf_filename):
        os.remove(pdf_filename)

    return saved_count

def main():
    # Interactive input
    pass_energy = input("Enter pass energy: ").strip()
    th_value = input("Enter θ value: ").strip()
    z_value = input("Enter z value: ").strip()
    polarity = input("Enter polarity (+ or -): ").strip()
    target = input("Enter target (LH2, LD2, Carbon, Copper): ").strip()

    # Construct path to the analysis script
    script_path = construct_script_path(pass_energy, th_value, z_value, polarity, target)
    print(f"\nRunning script: {script_path}")

    # PDF output in the current working directory
    output_dir = os.path.join(os.getcwd(), "SIDIS_plots")
    os.makedirs(output_dir, exist_ok=True)

    pdf_filename = os.path.join(
        output_dir,
        f"{target}_{pass_energy}pass_th{th_value}_z{z_value}_pi{polarity}.pdf"
    )
    print(f"Output PDF: {pdf_filename}\n")

    # Run script and save figures
    try:
        saved_count = run_script_and_save_figs(script_path, pdf_filename)
        if saved_count > 0:
            print(f"Saved {saved_count} figure(s) to {pdf_filename}")
        else:
            print("No figures were created by the script (or they were closed). PDF not saved.")
    except Exception as e:
        print("The script raised an exception while running:")
        print(e)

if __name__ == "__main__":
    main()


