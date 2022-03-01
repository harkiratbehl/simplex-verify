#!/usr/bin/env python
import argparse
import glob
import itertools
import seaborn as sns
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import sys


def parse_results(file_content, sp_idx, method_name):
    results = []
    lines = file_content.split('\n')
    timing = float(lines[0])
    all_bounds = lines[1].split('\t')
    all_bounds = list(map(float, all_bounds))

    for neur_idx, bound in enumerate(all_bounds):
        results.append({
            "Neuron index": neur_idx,
            "Sample": sp_idx,
            "Value": bound,
            "Method": method_name,
            "Time": timing
        })
    return results

def do_bound_versus_time_graph(datapoints, filename):
    datapoints = datapoints.reset_index(level=1)

    fig = plt.figure(figsize=(10,10))
    ax_value = plt.subplot(1, 1, 1)
    ax_value.set_xscale("log")
    sns.scatterplot(x="Time", y="Gap to best bound", data=datapoints,
                    hue="Method", legend='brief', alpha=0.2)
    leg = ax_value.legend()
    for lh in leg.legendHandles:
        lh.set_alpha(1)
        lh.set_linewidth(5)

    target_format = "eps" if filename.endswith('.eps') else "png"
    plt.tight_layout()
    plt.savefig(filename, format=target_format, dpi=50)


def do_violin_plots(datapoints, filename, labels, planet_only, imp_from=0):
    datapoints = datapoints.reset_index(level=1)

    fig, (time_ax, bound_ax) = plt.subplots(2, 1, sharex='col',
                                            figsize=(20, 7))

    plt.grid(True, which="both", ls="-")
    sns.boxplot(x="Method", y="Time", data=datapoints, ax=time_ax, order=labels, showfliers=False)
    time_ax.set(yscale="log")
    # bound_ax.set(yticks=([-0.5,0,0.5,1,1.5,2,2.5,3,3.5]))
    # time_ax.set(yscale="log", yticks=([0,1,10,100]))
    # time_ax.set_ylim([0.01, 500])
    
    if imp_from==0:
        bound_label = "Gap to best bound" if planet_only else "Improvement from Planet"
    elif imp_from==1:
        bound_label = "Gap to best bound" if planet_only else "Improvement from Gurobi DP"
    elif imp_from==2:
        bound_label = "Gap to Gurobi DP"
    elif imp_from==3:
        bound_label = "Gap to Pgd"

    for clabel in labels:
        slice = datapoints[(datapoints['Method'] == clabel)]
        slice = slice[slice[bound_label].between(slice[bound_label].quantile(.001), slice[bound_label].quantile(.999))]
        datapoints[(datapoints['Method'] == clabel)] = slice

    sns.violinplot(x="Method", y=bound_label, data=datapoints,
                   inner=None, width=1, scale='area', ax=bound_ax,
                   order=labels, cut=0)
    if planet_only:
        bound_ax.set(yscale="log")
    else:
        # bound_ax.set(yscale="symlog")
        # bound_ax.set(yticks=([-0.5,0,0.5,1,1.5,2,2.5,3,3.5]))
        bound_ax.set(yscale="linear")
    bound_ax.xaxis.set_label_text("")

    target_format = "pdf" if filename.endswith('.pdf') else "png"
    plt.tight_layout()
    # plt.savefig(filename, format=target_format, dpi=50)
    plt.savefig(filename.replace(target_format, 'pdf'), pad_inches=0)

def simplex_main(results_folder, output_image_prefix, files_to_load, legend_names, planet_only=False, violin_opts=None):

    all_results = []
    for sp_idx in sorted(os.listdir(results_folder)):
        sp_results = []
        for method_name, filename in zip(legend_names, files_to_load):
            file_to_load = os.path.join(results_folder, sp_idx, filename)
            if os.path.exists(file_to_load):
                # print(file_to_load)
                with open(file_to_load, 'r') as res_file:
                    sp_results.extend(parse_results(res_file.read(),
                                                    int(sp_idx),
                                                    method_name))
            else:
                break
        else:
            all_results.extend(sp_results)

    all_results = pd.DataFrame(all_results)
    # print(all_results[all_results["Method"] == "Gurobi\nBaseline"]["Sample"])
    # n_completed = len(all_results[all_results["Method"] == "Gurobi\nDP"]["Sample"])/9
    # print(f'Completed images: {n_completed}')

    print(all_results)
    all_results["Optimization Problem"] = all_results["Sample"].astype(str).str.cat(
        all_results["Neuron index"].astype(str), sep=' - ')
    all_results.drop(columns=["Neuron index", "Sample"], inplace=True)
    if planet_only:
        all_results.set_index(["Optimization Problem", "Method"], inplace=True)
        best_bound = all_results["Value"].groupby(["Optimization Problem"]).min()
        all_results["Gap to best bound"] = all_results["Value"] - best_bound

        # Let's check that Gurobi is always the best bound
        gurobi_worst = all_results.query('Method == "Gurobi dp cut"')["Gap to best bound"].max()
        print(f"Gurobi worst gap is: {gurobi_worst}")
    else:

        pgd_bounds = all_results[all_results["Method"] == 'Pgd'].drop(columns=["Method", "Time"])
        pgd_bounds.set_index(["Optimization Problem"], inplace=True)
        pgd_bounds = pgd_bounds.squeeze()

        # gur1cut_bounds = all_results[all_results["Method"] == 'Gurobi\nDP'].drop(columns=["Method", "Time"])
        # gur1cut_bounds.set_index(["Optimization Problem"], inplace=True)
        # gur1cut_bounds = gur1cut_bounds.squeeze()
        # # print('DP bounds')
        # # input(gur1cut_bounds)

        # planet_bounds = all_results[all_results["Method"] == 'Gurobi\nPlanet'].drop(columns=["Method", "Time"])
        # planet_bounds.set_index(["Optimization Problem"], inplace=True)
        # planet_bounds = planet_bounds.squeeze()

        all_results.set_index(["Optimization Problem", "Method"], inplace=True)

        # all_results["Improvement from Gurobi DP"] = gur1cut_bounds - all_results["Value"]
        # all_results["Gap to Gurobi DP"] = all_results["Value"] - gur1cut_bounds

        # all_results["Gap to Planet"] = all_results["Value"] - planet_bounds
        # all_results["Improvement from Planet"] = planet_bounds - all_results["Value"]

        # for lower bounds
        # all_results["Gap to Planet"] = planet_bounds - all_results["Value"] 
        # all_results["Improvement from Planet"] = all_results["Value"] - planet_bounds

        all_results["Gap to Pgd"] =  all_results["Value"] - pgd_bounds
        ## all_results["Gap to Pgd"] = all_results["Value"]

        # print(all_results)
        # print(all_results["Gap to Pgd"])

    # if violin_opts is None:
    #     # do_violin_plots(all_results, output_image_prefix + "_planet" + "_violins.png", legend_names, planet_only, imp_from=0)
    #     # do_violin_plots(all_results, output_image_prefix + "_imp_dp" + "_violins.png", legend_names, planet_only, imp_from=1)
    #     # do_violin_plots(all_results, output_image_prefix + "_gap_dp" + "_violins.png", legend_names, planet_only, imp_from=2)
    #     do_violin_plots(all_results, output_image_prefix + "_gap_pgd" + "_violins.png", legend_names, planet_only, imp_from=3)
    # else:
    #     for cname, clabels in zip(violin_opts["names"], violin_opts["labels"]):
    #         do_violin_plots(all_results, output_image_prefix + cname + "_violins.png", clabels, planet_only)

    # do_pairwise_plot(all_results, output_image_prefix)

    hex_plots = ["gap_pgd"] #"simplex", "gap_pgd", "simplex_planet_improvement"]
    for hex_plot in hex_plots:
        do_selected_pairwise_hex(all_results, output_image_prefix, hex_plot)

def main(results_folder, output_image_prefix, files_to_load, legend_names, planet_only=False, violin_opts=None):

    all_results = []
    for sp_idx in sorted(os.listdir(results_folder)):
        print(sp_idx)
        sp_results = []
        for method_name, filename in zip(legend_names, files_to_load):
            file_to_load = os.path.join(results_folder, sp_idx, filename)
            if os.path.exists(file_to_load):
                print(file_to_load)
                with open(file_to_load, 'r') as res_file:
                    sp_results.extend(parse_results(res_file.read(),
                                                    int(sp_idx),
                                                    method_name))
            else:
                break
        else:
            all_results.extend(sp_results)

    all_results = pd.DataFrame(all_results)
    n_completed = all_results[all_results["Method"] == "Gurobi\nBaseline"]["Sample"].max()
    print(f'Completed images: {n_completed}')
    all_results["Optimization Problem"] = all_results["Sample"].astype(str).str.cat(
        all_results["Neuron index"].astype(str), sep=' - ')
    all_results.drop(columns=["Neuron index", "Sample"], inplace=True)
    if planet_only:
        all_results.set_index(["Optimization Problem", "Method"], inplace=True)
        best_bound = all_results["Value"].groupby(["Optimization Problem"]).min()
        all_results["Gap to best bound"] = all_results["Value"] - best_bound

        # Let's check that Gurobi is always the best bound
        gurobi_worst = all_results.query("Method == 'Gurobi'")["Gap to best bound"].max()
        print(f"Gurobi worst gap is: {gurobi_worst}")
    else:
        gur1cut_bounds = all_results[all_results["Method"] == 'Gurobi\n1 cut'].drop(columns=["Method", "Time"])
        planet_bounds = all_results[all_results["Method"] == 'Gurobi\nPlanet'].drop(columns=["Method", "Time"])
        gur1cut_bounds.set_index(["Optimization Problem"], inplace=True)
        planet_bounds.set_index(["Optimization Problem"], inplace=True)
        gur1cut_bounds = gur1cut_bounds.squeeze()
        planet_bounds = planet_bounds.squeeze()

        all_results.set_index(["Optimization Problem", "Method"], inplace=True)

        all_results["Improvement from 1 cut"] = gur1cut_bounds - all_results["Value"]
        all_results["Gap to Planet"] = all_results["Value"] - planet_bounds
        all_results["Improvement from Planet"] = planet_bounds - all_results["Value"]

    if violin_opts is None:
        do_violin_plots(all_results, output_image_prefix + "_violins.png", legend_names, planet_only)
    else:
        for cname, clabels in zip(violin_opts["names"], violin_opts["labels"]):
            do_violin_plots(all_results, output_image_prefix + cname + "_violins.png", clabels, planet_only)

    # do_pairwise_plot(all_results, output_image_prefix)

    hex_plots = ["anderson_bigm", "anderson_planet_improvement"] #, "anderson_1cut_improvement"]
    for hex_plot in hex_plots:
        do_selected_pairwise_hex(all_results, output_image_prefix, hex_plot)


def do_pairwise_plot(datapoints, filename_prefix):

    sns.set(font_scale=2.5)
    sns.set_palette(sns.color_palette("Set2"))
    sns.set_style("whitegrid")

    datapoints = datapoints.reset_index(level=1)

    all_methods = datapoints.Method.unique()
    target_dir = filename_prefix + "_pairwise_comparisons"
    os.makedirs(target_dir, exist_ok=True)
    all_pairs = itertools.permutations(all_methods, 2)
    for meth1, meth2 in all_pairs:

        filename = "{meth1}_vs_{meth2}.png".format(meth1=meth1, meth2=meth2)
        filename = filename.replace("\n", '_')
        filename = os.path.join(target_dir, filename)
        target_format = "pdf" if filename.endswith(".pdf") else "png"
        meth1_data = datapoints[datapoints.Method == meth1]
        meth2_data = datapoints[datapoints.Method == meth2]

        fig, (time_ax, bound_ax) = plt.subplots(1, 2, figsize=(20, 10))
        sns.scatterplot(x=meth1_data.Time, y=meth2_data.Time, ax=time_ax, s=64)
        time_ax.set_xscale("log")
        time_ax.set_yscale("log")
        time_ax.xaxis.set_label_text(meth1.replace('\n', ' '))
        time_ax.yaxis.set_label_text(meth2.replace('\n', ' '))
        time_ax.set_title("Timing (in s)")
        xlim = time_ax.get_xlim()
        ylim = time_ax.get_ylim()
        rng = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        time_ax.plot(rng, rng, ls="--", c=".3")


        to_plot = "Gap to best bound"
        # to_plot = "Value"
        sns.scatterplot(x=meth1_data[to_plot], y=meth2_data[to_plot], ax=bound_ax, s=64)
        bound_ax.set_xscale("log")
        bound_ax.set_yscale("log")
        bound_ax.set_xlim(1e-3, 1e0)
        bound_ax.set_ylim(1e-3, 1e0)
        bound_ax.xaxis.set_label_text(meth1.replace('\n', ' '))
        bound_ax.yaxis.set_label_text(meth2.replace('\n', ' '))
        bound_ax.set_title(to_plot)
        xlim = bound_ax.get_xlim()
        ylim = bound_ax.get_ylim()
        rng = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        bound_ax.plot(rng, rng, ls="--", c=".3")

        plt.tight_layout()
        plt.savefig(filename, format=target_format, dpi=100)

        plt.close(fig)


def do_selected_pairwise_hex(datapoints, filename_prefix, plot_type):

    allowed_plots = [
        "uai_planet",
        "anderson_bigm",
        "anderson_planet_improvement",
        "anderson_1cut_improvement",
        "simplex_planet_improvement",
        "gap_pgd",
        "simplex"
    ]
    if plot_type not in allowed_plots:
        raise ValueError("Wrong plot type for do_selected_pairwise_hex.")

    sns.set(font_scale=2.5)
    sns.set_palette(sns.color_palette("Set2"))
    sns.set_style("ticks")

    datapoints = datapoints.reset_index(level=1)

    gridsize = 50

    if plot_type == "uai_planet":
        comparisons = [
            ("Proximal\n400 steps", "Supergradient\n740 steps"),
            ("Supergradient\n740 steps", "DSG+\n1040 steps"),
            ("Supergradient\n740 steps", "Dec-DSG+\n1040 steps"),
            ("Dec-DSG+\n1040 steps", "DSG+\n1040 steps"),
            ("WK", "IP")
        ]
        bnd_extent = (-2.3, -0.7, -2.3, -0.7) if filename_prefix == "../results/temp/madry8" else (
        -1.8, -0.4, -1.8, -0.4)
        bnd_scale = 'log'
        time_extent = (1, 4, 1, 4)
        to_plot = "Gap to best bound"

    elif plot_type == "anderson_bigm":
        comparisons = [
            ("BDD+\n400 steps", "Big-M\n850 steps"),
        ]
        time_extent = (1, 4, 1, 4)
        bnd_extent = (-2.3, -0.7, -2.3, -0.7) if "madry8" in filename_prefix else (-2, -0.4, -2, -0.4)
        bnd_scale = 'log'
        to_plot = "Gap to Planet"

    elif plot_type == "anderson_planet_improvement":
        comparisons = [
            ("Big-M\n850 steps", "Active Set\n80 steps"),
            # ("Gurobi\n1 cut", "Active Set\n1650 steps"),  # mnist
            # ("Gurobi\n1 cut", "Active Set\n2500 steps"), # mnist
            # ("Big-M\n850 steps", "SP-FW\n80 steps"),  # TODO: uncomment for JMLR
        ]
        time_extent = (1, 4, 1, 4)
        # time_extent = (20, 130, 20, 130) # mnist
        # time_extent = (2, 3, 2, 3)
        # bnd_extent = (-1.5, 0.1, -1.5, 0.1)
        bnd_extent = (-.1, 0.5, -.1, 0.5)
        # bnd_extent = (0, 1, 0, 1) # mnist
        bnd_scale = 'linear'
        to_plot = "Improvement from Planet"

    elif plot_type == "anderson_1cut_improvement":
        # Both Cuts and SP-FW.
        comparisons = [
            ("Active Set\n1650 steps", "SP-FW\n4000 steps"),
            ("Active Set\n1650 steps", "AS + SP-FW\n3600 steps"),
            ("Active Set\n600 steps", "SP-FW\n1000 steps"),
            ("Active Set\n1050 steps", "SP-FW\n2000 steps"),
        ]
        time_extent = (8, 40, 8, 40)
        bnd_extent = (-0.5, 0.9, -0.5, 0.9)
        bnd_scale = 'linear'
        to_plot = "Improvement from 1 cut"

    elif plot_type == "simplex_planet_improvement":
        comparisons = [
            ("Lirpa 6it\nPlanet", "Lirpa 3it\nDP"),
            # ("Gurobi\n1 cut", "Active Set\n1650 steps"),  # mnist
            # ("Gurobi\n1 cut", "Active Set\n2500 steps"), # mnist
            # ("Big-M\n850 steps", "SP-FW\n80 steps"),  # TODO: uncomment for JMLR
        ]
        time_extent = (0, 0.1, 0, 0.1)
        # time_extent = (20, 130, 20, 130) # mnist
        # time_extent = (2, 3, 2, 3)
        bnd_extent = (-0.5, 1, -0.5, 1)
        # bnd_extent = (0, 3, 0, 3)
        # bnd_extent = (-.1, 0.5, -.1, 0.5)
        # bnd_extent = (0, 1, 0, 1) # mnist
        bnd_scale = 'linear'
        to_plot = "Improvement from Planet"

    elif plot_type == "gap_pgd":
        comparisons = [
            ("Opt-Lirpa Planet", "Simplex Verify"),
            # ("Gurobi Planet", "Gurobi Simplex"),
            # ("Gurobi\n1 cut", "Active Set\n1650 steps"),  # mnist
            # ("Gurobi\n1 cut", "Active Set\n2500 steps"), # mnist
            # ("Big-M\n850 steps", "SP-FW\n80 steps"),  # TODO: uncomment for JMLR
        ]
        time_extent = (0.0, 0.5, 0.0, 0.5)
        # time_extent = (20, 130, 20, 130) # mnist
        # time_extent = (2, 3, 2, 3)
        bnd_extent = [5, 20, 5, 20]
        # bnd_extent = (0, 3, 0, 3)
        # bnd_extent = (-.1, 0.5, -.1, 0.5)
        # bnd_extent = (0, 1, 0, 1) # mnist
        bnd_scale = 'linear'
        to_plot = "Gap to Pgd"
    
    elif plot_type == "simplex":
        comparisons = [
            ("Opt-Lirpa Planet", "Simplex Verify"),
            ("Gurobi Planet", "Gurobi Simplex"),
            # ("Gurobi\n1 cut", "Active Set\n1650 steps"),  # mnist
            # ("Gurobi\n1 cut", "Active Set\n2500 steps"), # mnist
            # ("Big-M\n850 steps", "SP-FW\n80 steps"),  # TODO: uncomment for JMLR
        ]
        time_extent = (0.0, 0.5, 0.0, 0.5)
        # time_extent = (20, 130, 20, 130) # mnist
        # time_extent = (2, 3, 2, 3)
        bnd_extent = [5, 20, 5, 20]
        # bnd_extent = (0, 3, 0, 3)
        # bnd_extent = (-.1, 0.5, -.1, 0.5)
        # bnd_extent = (0, 1, 0, 1) # mnist
        bnd_scale = 'linear'
        to_plot = "Value"

    target_dir = filename_prefix + "_pairwise_comparisons"
    os.makedirs(target_dir, exist_ok=True)

    for meth1, meth2 in comparisons:

        filename = "{meth1}_vs_{meth2}_hex.pdf".format(meth1=meth1, meth2=meth2)
        filename = filename.replace("\n", '_')
        filename = filename.replace(" ", '_')
        filename = os.path.join(target_dir, filename)
        target_format = "pdf" if filename.endswith(".pdf") else "png"
        meth1_data = datapoints[datapoints.Method == meth1]
        meth2_data = datapoints[datapoints.Method == meth2]

        fig, (time_ax, bound_ax) = plt.subplots(1, 2, figsize=(20, 10))
        time_ax.hexbin(x=meth1_data.Time, y=meth2_data.Time, cmap="Reds", gridsize=gridsize, mincnt=1,
                       xscale='linear', yscale='linear', bins='log', extent=time_extent)
                       # xscale='linear', yscale='linear', bins='log')
        time_ax.xaxis.set_label_text(meth1.replace('\n', ' '))
        time_ax.yaxis.set_label_text(meth2.replace('\n', ' '))
        time_ax.set_title("Timing (in s)")
        xlim = time_ax.get_xlim()
        ylim = time_ax.get_ylim()
        rng = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        time_ax.plot(rng, rng, ls="--", c=".3")

        bound_ax.hexbin(x=meth1_data[to_plot], y=meth2_data[to_plot], gridsize=gridsize, mincnt=1, cmap="Blues",
                        # xscale=bnd_scale, yscale=bnd_scale, bins='log', extent=bnd_extent)
                        xscale=bnd_scale, yscale=bnd_scale, bins='log')
        bound_ax.xaxis.set_label_text(meth1.replace('\n', ' '))
        bound_ax.yaxis.set_label_text(meth2.replace('\n', ' '))
        bound_ax.set_title(to_plot)
        bound_ax.set_xlim(4, 17.5)
        bound_ax.set_ylim(4, 17.5)
        xlim = bound_ax.get_xlim()
        ylim = bound_ax.get_ylim()
        print(xlim, ylim)
        rng = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        bound_ax.plot(rng, rng, ls="--", c=".3")

        if plot_type != "uai_planet":
            rng_x = (xlim[0], 0)
            rng_y = (ylim[0], 0)
            val = (0, 0)
            bound_ax.plot(rng_x, val, ls="--", c=".3", alpha=0.5)
            bound_ax.plot(val, rng_y, ls="--", c=".3", alpha=0.5)

        plt.tight_layout()
        plt.savefig(filename, format=target_format, dpi=100)

        plt.close(fig)


def iclr_cifar_plots():

    temp_dir = "../results/anderson_iclr_cr/"
    os.system(f"mkdir -p {temp_dir}")

    # Define the filenames to be loaded for this plot.
    files_to_load = [
        "Gurobi-fromintermediate-fixed.txt",
        "Anderson-1cuts-fromintermediate-fixed.txt",
        "Proximal_finalmomentum_400-fromintermediate.txt",
        "Big-M_850-fromintermediate.txt",
    ]
    for citers in [80, 600, 1050, 1650]:
        files_to_load.append(f"Cuts_{citers}-fromintermediate.txt")
    files_to_load.append("CutsCPU_600-fromintermediate.txt")
    # sensitivity to selection criterion
    for citers in [600, 1050, 1650]:
        files_to_load.append(f"Cuts_randmask_{citers}-fromintermediate.txt")
    # sensitivity to selection frequency
    freq_steps = [(300, [550, 900, 1500]), (600, [650, 1150, 1850])]
    for freq, cut_steps_list in freq_steps:
        for cut_steps in cut_steps_list:
            files_to_load.append(f"Cuts_freq{freq}_{cut_steps}-fromintermediate.txt")

    # Properly label the filenames to be loaded.
    legend_names = [
        "Gurobi\nPlanet",
        "Gurobi\n1 cut",
        "BDD+\n400 steps",
        "Big-M\n850 steps",
    ]
    for citers in [80, 600, 1050, 1650]:
        legend_names.append(f"Active Set\n{citers} steps")
    legend_names.append("Active Set CPU\n600 steps")
    # sensitivity to selection criterion
    for citers in [600, 1050, 1650]:
        legend_names.append(f"Ran. Selection\n{citers} steps")
    # sensitivity to selection frequency
    for freq, cut_steps_list in freq_steps:
        for cut_steps in cut_steps_list:
            legend_names.append(fr"AS, $\omega={freq}$" + f"\n{cut_steps} steps")

    labels_criterion = ["Gurobi\nPlanet"]
    labels_frequency = []
    for i, citers in enumerate([600, 1050, 1650]):
        labels_criterion.append(f"Active Set\n{citers} steps")
        labels_frequency.append(f"Active Set\n{citers} steps")
        labels_criterion.append(f"Ran. Selection\n{citers} steps")
        for freq, cut_steps_list in freq_steps:
            labels_frequency.append(fr"AS, $\omega={freq}$" + f"\n{cut_steps_list[i]} steps")
    violin_opts = {
        "labels": [legend_names[:-9], labels_criterion, labels_frequency],
        "names": ["", "sensitivity", "frequency"],
    }

    # Madry and SGD experiments.
    folders = [("../results/madry8/", f"{temp_dir}/madry8_and"), ("../results/sgd8/", f"{temp_dir}/sgd8_and")]

    for results_folder, output_image_prefix in folders:
        sns.set(font_scale=1.5); sns.set_palette(sns.color_palette("Set2")); sns.set_style("whitegrid")
        main(results_folder, output_image_prefix, files_to_load, legend_names, violin_opts=violin_opts)


def iclr_mnist_plots():

    temp_dir = "../results/anderson_iclr_cr_madry/"
    os.system(f"mkdir -p {temp_dir}")

    # Define the filenames to be loaded for this plot.
    files_to_load = [
        "Gurobi-fromintermediate-fixed.txt",
        "Anderson-1cuts-fromintermediate-fixed.txt",
        "Proximal_finalmomentum_400-fromintermediate.txt",
        "Big-M_850-fromintermediate.txt",
    ]
    for citers in [80, 600, 1050, 1650, 2500]:
        files_to_load.append(f"Cuts_{citers}-fromintermediate.txt")

    # Properly label the filenames to be loaded.
    legend_names = [
        "Gurobi\nPlanet",
        "Gurobi\n1 cut",
        "BDD+\n400 steps",
        "Big-M\n850 steps",
    ]
    for citers in [80, 600, 1050, 1650, 2500]:
        legend_names.append(f"Active Set\n{citers} steps")

    # Madry and SGD experiments.
    folders = [("../results/mnist_wide/", f"{temp_dir}/mnist_wide_and")]  #  ("../results/mnist_deep/", f"{temp_dir}/mnist_deep_and")

    for results_folder, output_image_prefix in folders:
        sns.set(font_scale=1.5); sns.set_palette(sns.color_palette("Set2")); sns.set_style("whitegrid")
        main(results_folder, output_image_prefix, files_to_load, legend_names)

def simplex_cifar_plots():

    # temp_dir = "./results/sgd8_ll/"
    temp_dir = "./results/cifar_model_wide_l1/"
    os.system(f"mkdir -p {temp_dir}")

    # Define the filenames to be loaded for this plot.
    files_to_load = [
        "l_pgd-fromintermediate-fixed.txt",
        "gurobi-baseline-planet-simplex-fromintermediate-fixed.txt",
        "gurobi-dp-simplex-fromintermediate-fixed.txt",
        # "baseline-bigm-adam-simplex_500-fromintermediate.txt",
        # "baseline-bigm-adam-simplex_1000-fromintermediate.txt",
        "baseline-lirpa-fromintermediate-fixed.txt",
        # "l_baseline-lirpa-fromintermediate-fixed.txt",
        "auto-lirpa-dp-3-fromintermediate-fixed.txt",
        # "l_auto-lirpa-dp-fromintermediate-fixed.txt",
    ]
    # for citers in [500, 1000]:
    #     files_to_load.append(f"baseline-bigm-adam-simplex_{citers}-fromintermediate.txt")
    #     files_to_load.append(f"bigm-adam-simplex_{citers}-fromintermediate.txt")


    # Properly label the filenames to be loaded.
    legend_names = [
        "Pgd",
        "Gurobi Planet",
        "Gurobi Simplex",
        # "Dual\nPlanet",
        # "Bigm-adam 1000 it\nPlanet",
        "Opt-Lirpa Planet",
        # "Lirpa 5it\nPlanet",
        "Simplex Verify",
        # "Lirpa 5it\nDP",
    ]
    # for citers in [500, 1000]:
    #     legend_names.append(f"Basline adam\n{citers} steps")
    #     legend_names.append(f"1 cut adam\n{citers} steps")

    # Madry and SGD experiments.
    folders = [("./results/cifar_model_wide_l1/", f"{temp_dir}/cifar_model_wide_l1_and")]
    # folders = [("./results/sgd8_ll/", f"{temp_dir}/sgd8_and")]
    # folders = [("./results/madry_ll/", f"{temp_dir}/madry_and")]

    for results_folder, output_image_prefix in folders:
        sns.set(font_scale=1.5); sns.set_palette(sns.color_palette("Set2")); sns.set_style("whitegrid")
        simplex_main(results_folder, output_image_prefix, files_to_load, legend_names, planet_only=False, violin_opts=None)

def simplex_mmbt_plots():

    # temp_dir = "./results/sgd8_ll/"
    temp_dir = "./results/mmbt_newer_0.5_custom_2btidx/"
    temp_dir = "./results/mmbt_save_newer_1_10_0.65pgd_10_0.26_0.3/"

    os.system(f"mkdir -p {temp_dir}")

    # Define the filenames to be loaded for this plot.
    files_to_load = [
        "l_pgd-fromintermediate-fixed.txt",
        "baseline-lirpa-fromintermediate-fixed.txt",
        "auto-lirpa-dp-fromintermediate-fixed.txt",
    ]

    # Properly label the filenames to be loaded.
    legend_names = [
        "Pgd",
        "Opt-Lirpa Planet",
        "Simplex Verify",
    ]

    # Madry and SGD experiments.
    # folders = [("./results/sgd8_ll/", f"{temp_dir}/sgd8_and")]

    # folders = [("./results/mmbt_newer_0.5_custom_1btidx/", f"./results/mmbt_newer_0.5_custom_1btidx//mmbt_newer_0.5_custom_1btidx_and"), ("./results/mmbt_newer_0.5_custom_2btidx/", f"./results/mmbt_newer_0.5_custom_2btidx/mmbt_newer_0.5_custom_2btidx_and"), ("./results/mmbt_newer_0.5_custom_3btidx/", f"./results/mmbt_newer_0.5_custom_3btidx/mmbt_newer_0.5_custom_3btidx_and"), ("./results/mmbt_newer_0.5_custom_4btidx/", f"./results/mmbt_newer_0.5_custom_4btidx/mmbt_newer_0.5_custom_4btidx_and"), ("./results/mmbt_newer_0.5/", f"./results/mmbt_newer_0.5/mmbt_newer_0.5_and"), ("./results/mmbt_newer_0.5_custom_1/", f"./results/mmbt_newer_0.5_custom_1/mmbt_newer_0.5_custom_1_and")]
    
    # folders = [("./results/mmbt_newer_0.5_custom_1/", f"./results/mmbt_newer_0.5_custom_1/mmbt_newer_0.5_custom_1_and")]
    # folders = [("./results/mmbt_newer_0.5/", f"./results/mmbt_newer_0.5/mmbt_newer_0.5_and")]
    # folders = [("./results/mmbt_newer_0.5_custom_2btidx/", f"./results/mmbt_newer_0.5_custom_2btidx/mmbt_newer_0.5_custom_2btidx_and")]
    folders = [("./results/mmbt_save_newer_1_10_0.65pgd_10_0.26_0.3/", f"./results/mmbt_save_newer_1_10_0.65pgd_10_0.26_0.3/mmbt_save_newer_1_10_0.65pgd_10_0.26_0.3_and")]


    for results_folder, output_image_prefix in folders:
        sns.set(font_scale=1.5); sns.set_palette(sns.color_palette("Set2")); sns.set_style("whitegrid")
        simplex_main(results_folder, output_image_prefix, files_to_load, legend_names, planet_only=False, violin_opts=None)

def simplex_cifar_plots_eps():

    temp_dir = "./results/madry75/"
    os.system(f"mkdir -p {temp_dir}")

    # Define the filenames to be loaded for this plot.
    files_to_load = [
        "gurobi-baseline-planet-simplex-fromintermediate-fixed.txt",
        "gurobi-planet-simplex-fromintermediate-fixed.txt",
    ]
    for citers in [500, 1000]:
        files_to_load.append(f"baseline-bigm-adam-simplex_{citers}-fromintermediate.txt")
        files_to_load.append(f"bigm-adam-simplex_{citers}-fromintermediate.txt")


    # Properly label the filenames to be loaded.
    legend_names = [
        "Gurobi\nBaseline",
        "Gurobi\n1 cut",
    ]
    for citers in [500, 1000]:
        legend_names.append(f"Basline adam\n{citers} steps")
        legend_names.append(f"1 cut adam\n{citers} steps")

    # Madry and SGD experiments.
    folders = [("./results/madry75/", f"{temp_dir}/madry75_and")]

    for results_folder, output_image_prefix in folders:
        sns.set(font_scale=1.5); sns.set_palette(sns.color_palette("Set2")); sns.set_style("whitegrid")
        simplex_main(results_folder, output_image_prefix, files_to_load, legend_names, planet_only=False, violin_opts=None)
if __name__ == '__main__':

    if int(sys.argv[1]) == 0:
        iclr_cifar_plots()
    elif int(sys.argv[1]) == 1:
        iclr_mnist_plots()
    elif int(sys.argv[1]) == 2:
        simplex_cifar_plots()
    elif int(sys.argv[1]) == 3:
        simplex_mmbt_plots()
    else:
        raise IOError("This file expects a single parameter, in [0,1]")
