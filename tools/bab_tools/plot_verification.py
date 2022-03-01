import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.titlepad'] = 10
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
import itertools
import os
from tools.pd2csv import pd2csv


def load_verification_data():

    # whether to take only the properties for which prox terminated
    intersect = True

    folder = "batch_verification_results/"
    m2_easy = pd.read_pickle(folder + "jodie-base_easy.pkl")
    m2_med = pd.read_pickle(folder + "jodie-base_med.pkl")
    m2_hard = pd.read_pickle(folder + "jodie-base_hard.pkl")
    wide = pd.read_pickle(folder + "jodie-wide.pkl")
    deep = pd.read_pickle(folder + "jodie-deep.pkl")

    # create column uniquely identifying the verification task (image id + class number)
    for m in [m2_easy, m2_med, m2_hard, wide, deep]:
        m["unique_id"] = m["Idx"].map(str) + "_" + m["prop"].map(str)

    m2_total = pd.concat([m2_easy, m2_med, m2_hard], ignore_index=True, sort=False)

    m2_easy_final = m2_total.loc[m2_total['BTime_KW'] < 800]
    m2_med_final = m2_total.loc[(m2_total['BTime_KW'] > 800) & (m2_total['BTime_KW'] < 2400)]
    m2_hard_final = m2_total.loc[(m2_total['BTime_KW'] > 2400)]

    # m2_easy_final = m2_easy
    # m2_med_final = m2_med
    # m2_hard_final = m2_hard

    # sanity check
    for m in [m2_easy_final, m2_med_final, m2_hard_final, wide, deep]:
        true = m.index[m['BSAT_KW'] == 'True']
        assert len(true) == 0

    nomomentum = True  # nomomentum looks much much better, keep it True
    largerlr = True  # largelr looks much much better, keep it True (unfortunately)
    largerlrdj = True  # largelr looks much much better, keep it True (unfortunately)
    nomomentum_string = "-nomomentum" if nomomentum else ""
    largerlr_string = "-largerlr" if largerlr else ""
    largerlr_string_dj = "-largerlr" if largerlrdj else ""

    # get new batch results. Dropna discards missing data. W/o how="all" timeouts would be discarded.
    batch_easy = pd.read_pickle(folder + f"base_easy_KW_prox_100-pinit{nomomentum_string}.pkl").dropna(how="all")
    # merge w/ the intersection (how='inner') w/ adam results
    batch_easy = pd.merge(
        batch_easy,
        pd.read_pickle(folder + f"base_easy_KW_adam_160-pinit{largerlr_string}.pkl").dropna(how="all"),
        on=['Idx', 'prop', 'Eps'], how='inner')
    # merge w/ the intersection (how='inner') w/ dj-adam results
    batch_easy = pd.merge(
        batch_easy,
        pd.read_pickle(folder + f"base_easy_KW_dj-adam_260-pinit{largerlr_string_dj}.pkl").dropna(how="all"),
        on=['Idx', 'prop', 'Eps'], how='inner')

    batch_med = pd.read_pickle(folder + f"base_med_KW_prox_100-pinit{nomomentum_string}.pkl").dropna(how="all")
    batch_med = pd.merge(
        batch_med,
        pd.read_pickle(folder + f"base_med_KW_adam_160-pinit{largerlr_string}.pkl").dropna(how="all"),
        on=['Idx', 'prop', 'Eps'], how='inner')
    # merge w/ the intersection (how='inner') w/ dj-adam results
    batch_med = pd.merge(
        batch_med,
        pd.read_pickle(folder + f"base_med_KW_dj-adam_260-pinit{largerlr_string_dj}.pkl").dropna(how="all"),
        on=['Idx', 'prop', 'Eps'], how='inner')

    batch_hard = pd.read_pickle(folder + f"base_hard_KW_prox_100-pinit{nomomentum_string}.pkl").dropna(how="all")
    batch_hard = pd.merge(
        batch_hard,
        pd.read_pickle(folder + f"base_hard_KW_adam_160-pinit{largerlr_string}.pkl").dropna(how="all"),
        on=['Idx', 'prop', 'Eps'], how='inner')
    # merge w/ the intersection (how='inner') w/ dj-adam results
    batch_hard = pd.merge(
        batch_hard,
        pd.read_pickle(folder + f"base_hard_KW_dj-adam_260-pinit{largerlr_string_dj}.pkl").dropna(how="all"),
        on=['Idx', 'prop', 'Eps'], how='inner')

    batch_wide = pd.read_pickle(folder + f"wide_KW_prox_100-pinit{nomomentum_string}.pkl").dropna(how="all")
    batch_wide = pd.merge(
        batch_wide,
        pd.read_pickle(folder + f"wide_KW_adam_160-pinit{largerlr_string}.pkl").dropna(how="all"),
        on=['Idx', 'prop', 'Eps'], how='inner')
    # merge w/ the intersection (how='inner') w/ dj-adam results
    batch_wide = pd.merge(
        batch_wide,
        pd.read_pickle(folder + f"wide_KW_dj-adam_260-pinit{largerlr_string_dj}.pkl").dropna(how="all"),
        on=['Idx', 'prop', 'Eps'], how='inner')

    batch_deep = pd.read_pickle(folder + f"deep_KW_prox_100-pinit{nomomentum_string}.pkl").dropna(how="all")
    batch_deep = pd.merge(
        batch_deep,
        pd.read_pickle(folder + f"deep_KW_adam_160-pinit{largerlr_string}.pkl").dropna(how="all"),
        on=['Idx', 'prop', 'Eps'], how='inner')
    # merge w/ the intersection (how='inner') w/ dj-adam results
    batch_deep = pd.merge(
        batch_deep,
        pd.read_pickle(folder + f"deep_KW_dj-adam_260-pinit{largerlr_string_dj}.pkl").dropna(how="all"),
        on=['Idx', 'prop', 'Eps'], how='inner')

    # create column uniquely identifying the verification task (image id + class number)
    for m in [batch_easy, batch_med, batch_hard, batch_wide, batch_deep]:
        m["unique_id"] = m["Idx"].map(str) + "_" + m["prop"].map(str)

    # group them into easy/med/hard as Jodie's
    batch_base_total = pd.concat([batch_easy, batch_med, batch_hard], ignore_index=True, sort=False)

    easy_idxs = m2_easy_final["unique_id"].values
    batch_easy_final = batch_base_total[batch_base_total['unique_id'].isin(easy_idxs)]

    med_idxs = m2_med_final["unique_id"].values
    batch_med_final = batch_base_total[batch_base_total['unique_id'].isin(med_idxs)]
    print(batch_med_final)
    # print(m2_med_final)

    hard_idxs = m2_hard_final["unique_id"].values
    batch_hard_final = batch_base_total[batch_base_total['unique_id'].isin(hard_idxs)]

    # retain only jodie's properties which we have already terminated
    if intersect:
        m2_easy_final = m2_easy_final[m2_easy_final['unique_id'].isin(batch_easy_final["unique_id"].values)]
        m2_med_final = m2_med_final[m2_med_final['unique_id'].isin(batch_med_final["unique_id"].values)]
        m2_hard_final = m2_hard_final[m2_hard_final['unique_id'].isin(batch_hard_final["unique_id"].values)]
        m2_total = m2_total[m2_total['unique_id'].isin(batch_base_total["unique_id"].values)]
        wide = wide[wide['unique_id'].isin(batch_wide["unique_id"].values)]
        deep = deep[deep['unique_id'].isin(batch_deep["unique_id"].values)]

    for m in [batch_easy_final, batch_med_final, batch_hard_final, batch_wide, batch_deep]:
        sat_labels = ['BSAT_KW_prox_100.0', 'BSAT_KW_adam_160.0', 'BSAT_KW_dj-adam_260.0']
        for sat_label in sat_labels:
            true = m.index[m[sat_label] == 'True']
            assert len(true) == 0

    # check no baseline run uses more nodes than prox.
    for jodie, mine in [(m2_easy_final, batch_easy_final), (m2_med_final, batch_med_final), (m2_hard_final, batch_hard_final),
              (wide, batch_wide), (deep, batch_deep)]:
        merged = pd.merge(
            jodie, mine, on=['Idx', 'prop', 'Eps', 'unique_id'], how='inner')
        assert len(merged[merged['BBran_KW_prox_100.0'] < merged['BBran_KW']]) == 0

    return (m2_easy_final, m2_med_final, m2_hard_final, m2_total, wide, deep), \
           (batch_easy_final, batch_med_final, batch_hard_final, batch_base_total, batch_wide, batch_deep)


def plot_from_tables(jodie_table, batch_table, x_axis_max, timeout, fig_name, title):

    Gtime = []
    for i in jodie_table['GTime'].values:
        if i >= timeout:
            Gtime += [float('inf')]
        else:
            Gtime += [i]
    Btime_KW = []
    for i in jodie_table['BTime_KW'].values:
        if i >= timeout:
            Btime_KW += [float('inf')]
        else:
            Btime_KW += [i]
    prox_time = []
    for i in batch_table['BTime_KW_prox_100.0'].values:
        if i >= timeout:
            prox_time += [float('inf')]
        else:
            prox_time += [i]
    adam_time = []
    for i in batch_table['BTime_KW_adam_160.0'].values:
        if i >= timeout:
            adam_time += [float('inf')]
        else:
            adam_time += [i]
    djadam_time = []
    for i in batch_table['BTime_KW_dj-adam_260.0'].values:
        if i >= timeout:
            djadam_time += [float('inf')]
        else:
            djadam_time += [i]
    Gtime.sort()
    Btime_KW.sort()
    prox_time.sort()
    adam_time.sort()
    djadam_time.sort()

    # check that they have the same length.
    assert len(Btime_KW) == len(prox_time)
    assert len(adam_time) == len(prox_time)
    assert len(adam_time) == len(djadam_time)
    print(f"{title} has {len(prox_time)} entries")

    starting_point = min(Gtime[0], Btime_KW[0], prox_time[0], adam_time[0])

    method_2_color = {}
    method_2_color['MIPplanet'] = 'red'
    method_2_color['Gurobi BaBSR'] = 'green'
    method_2_color['Proximal BaBSR'] = 'skyblue'
    method_2_color['Supergradient BaBSR'] = 'gold'
    method_2_color['DSG+ BaBSR'] = 'darkmagenta'
    fig = plt.figure(figsize=(10,10))
    ax_value = plt.subplot(1, 1, 1)
    ax_value.axhline(linewidth=3.0, y=100, linestyle='dashed', color='grey')

    y_min = 0
    y_max = 100
    ax_value.set_ylim([y_min, y_max+5])

    min_solve = float('inf')
    max_solve = float('-inf')
    for timings in [Gtime, Btime_KW, prox_time, adam_time, djadam_time]:
        min_solve = min(min_solve, min(timings))
        finite_vals = [val for val in timings if val != float('inf')]
        if len(finite_vals) > 0:
            max_solve = max(max_solve, max([val for val in timings if val != float('inf')]))


    axis_min = starting_point
    #ax_value.set_xscale("log")
    axis_min = min(0.5 * min_solve, 1)
    ax_value.set_xlim([axis_min, x_axis_max])

    # Plot all the properties
    linestyle_dict = {
        'MIPplanet': 'solid',
        'Gurobi BaBSR': 'solid',
        'Proximal BaBSR': 'solid',
        'Supergradient BaBSR': 'solid',
        'DSG+ BaBSR': 'solid',
    }

    for method, m_timings in [('MIPplanet', Gtime), ('Gurobi BaBSR', Btime_KW), ('Proximal BaBSR', prox_time),
                              ('Supergradient BaBSR', adam_time), ('DSG+ BaBSR', djadam_time)]:
        # Make it an actual cactus plot
        xs = [axis_min]
        ys = [y_min]
        prev_y = 0
        for i, x in enumerate(m_timings):
            if x <= x_axis_max:
                # Add the point just before the rise
                xs.append(x)
                ys.append(prev_y)
                # Add the new point after the rise, if it's in the plot
                xs.append(x)
                new_y = 100*(i+1)/len(m_timings)
                ys.append(new_y)
                prev_y = new_y
        # Add a point at the end to make the graph go the end
        xs.append(x_axis_max)
        ys.append(prev_y)

        ax_value.plot(xs, ys, color=method_2_color[method],
                      linestyle=linestyle_dict[method], label=method, linewidth=4.0)

    ax_value.set_ylabel("% of properties verified", fontsize=22)
    ax_value.set_xlabel("Computation time [s]", fontsize=22)
    plt.xscale('log', nonposx='clip')
    ax_value.legend(fontsize=19.5)
    plt.grid(True)
    plt.title(title)
    plt.savefig(fig_name, format='pdf', dpi=300)


def plot_batched_verification():

    jodie, batch = load_verification_data()

    m2_easy, m2_med, m2_hard, m2_total, wide, deep = jodie
    batch_easy, batch_med, batch_hard, batch_base_total, batch_wide, batch_deep = batch

    # correct for shorter time-out
    for m in [wide, deep]:
        m.loc[m['BTime_KW'] >= 3600, 'BTime_KW'] = 3600
        m.loc[m['GTime'] >= 3600, 'GTime'] = 3600
        m.loc[m['BTime_KW'] >= 3600, 'BSAT_KW'] = 'timeout'
        m.loc[m['GTime'] >= 3600, 'GSAT'] = 'timeout'
        m.loc[m['BTime_KW'] >= 3600, 'BBran_KW'] = None

    # check how many properties timed out
    for m in [m2_easy, m2_med, m2_hard, m2_total, wide, deep]:
        gurobi_timeout = len(m.loc[(m['BSAT_KW'] == 'timeout') | (m['BTime_KW'] >= 3600)])
        m_len = len(m)
        print('Gurobi ', gurobi_timeout / m_len)
        mip_timeout = len(m.loc[(m['GSAT'] == 'timeout') | (m['GTime'] >= 3600)])
        m_len = len(m)
        print('MIP ', mip_timeout / m_len)
    for m in [batch_easy, batch_med, batch_hard, batch_base_total, batch_wide, batch_deep]:
        prox_timeout = len(m.loc[m['BSAT_KW_prox_100.0'] == 'timeout'])
        m_len = len(m)
        print('Prox ', prox_timeout / m_len)
        adam_timeout = len(m.loc[m['BSAT_KW_adam_160.0'] == 'timeout'])
        m_len = len(m)
        print('Adam ', adam_timeout / m_len)
        dj_timeout = len(m.loc[m['BSAT_KW_dj-adam_260.0'] == 'timeout'])
        m_len = len(m)
        print('DJ ', dj_timeout / m_len)

    import pdb; pdb.set_trace()
    # to get means: batch_hard.mean()

    x_axis_max = 3600  # 800
    fig_name = "base_easy.pdf"
    title = "Base model, easy properties"
    timeout = 3600
    plot_from_tables(m2_easy, batch_easy, x_axis_max, timeout, fig_name, title)

    x_axis_max = 3600  # 2400
    fig_name = "base_med.pdf"
    title = "Base model, medium properties"
    timeout = 3600
    plot_from_tables(m2_med, batch_med, x_axis_max, timeout, fig_name, title)

    x_axis_max = 3600
    fig_name = "base_hard.pdf"
    title = "Base model, hard properties"
    timeout = 3600
    plot_from_tables(m2_hard, batch_hard, x_axis_max, timeout, fig_name, title)

    x_axis_max = 3600
    fig_name = "base_total.pdf"
    title = "Base model"
    timeout = 3600
    plot_from_tables(m2_total, batch_base_total, x_axis_max, timeout, fig_name, title)

    x_axis_max = 3600  # 7200
    fig_name = "wide.pdf"
    title = "Wide large model"
    timeout = 3600
    plot_from_tables(wide, batch_wide, x_axis_max, timeout, fig_name, title)

    x_axis_max = 3600  # 7200
    fig_name = "deep.pdf"
    title = "Deep large model"
    timeout = 3600
    plot_from_tables(deep, batch_deep, x_axis_max, timeout, fig_name, title)

    plt.show()

    # import pdb; pdb.set_trace()


def two_way_comparison():

    # folder1 = "cifar_results/"
    folder2 = "batch_verification_results/"
    timeout = 3600

    # problem = "wide"
    problem = "wide"
    # TODO: test gurobi-anderson_1 on wide (running)
    n_cuts = 1  # 100 is worse than 1 but as 2
    method1 = pd.read_pickle(folder2 + f"jodie-{problem}_KW_gurobi.pkl").dropna(how="all")
    # method2 = pd.read_pickle(folder2 + f"jodie-{problem}.pkl").dropna(how="all")
    method2 = pd.read_pickle(folder2 + f"jodie-{problem}_KW_gurobi-anderson_{n_cuts}.pkl").dropna(how="all")

    for m in [method1, method2]:
        m["unique_id"] = m["Idx"].map(str) + "_" + m["prop"].map(str)

    method2 = method2[method2['unique_id'].isin(method1["unique_id"].values)]
    method1 = method1[method1['unique_id'].isin(method2["unique_id"].values)]

    assert len(method1) == len(method2)

    # check number of branches is the same
    if not (method1["BBran_KW_gurobi"] == method2[f"BBran_KW_gurobi-anderson_{n_cuts}"]).all():
        diff = (method1["BBran_KW_gurobi"] - method2[f"BBran_KW_gurobi-anderson_{n_cuts}"])
        print(diff)
        import pdb; pdb.set_trace()

    new_gurobi = []
    for i in method1["BTime_KW_gurobi"].values:
        if i >= timeout:
            new_gurobi += [float('inf')]
        else:
            new_gurobi += [i]
    new_gurobi.sort()
    gurobi_baseline = []
    for i in method2[f"BTime_KW_gurobi-anderson_{n_cuts}"].values:
        if i >= timeout:
            gurobi_baseline += [float('inf')]
        else:
            gurobi_baseline += [i]
    gurobi_baseline.sort()

    fig = plt.figure(figsize=(10, 10))
    ax_value = plt.subplot(1, 1, 1)
    ax_value.axhline(linewidth=3.0, y=100, linestyle='dashed', color='grey')

    y_min = 0
    y_max = 100
    ax_value.set_ylim([y_min, y_max + 5])
    axis_min = min(new_gurobi[0], gurobi_baseline[0])
    axis_min = 1
    x_axis_max = 3600

    min_solve = float('inf')
    max_solve = float('-inf')
    for timings in [new_gurobi, gurobi_baseline]:
        min_solve = min(min_solve, min(timings))
        finite_vals = [val for val in timings if val != float('inf')]
        if len(finite_vals) > 0:
            max_solve = max(max_solve, max([val for val in timings if val != float('inf')]))

    # Plot all the properties
    linestyle_all = 'solid'
    for method, m_timings in [('Gurobi-BaBSR', new_gurobi), (f'And. Gurobi-BaBSR - {n_cuts} cuts', gurobi_baseline)]:
        # Make it an actual cactus plot
        xs = [axis_min]
        ys = [y_min]
        prev_y = 0
        for i, x in enumerate(m_timings):
            if x <= x_axis_max:
                # Add the point just before the rise
                xs.append(x)
                ys.append(prev_y)
                # Add the new point after the rise, if it's in the plot
                xs.append(x)
                new_y = 100 * (i + 1) / len(m_timings)
                ys.append(new_y)
                prev_y = new_y
        # Add a point at the end to make the graph go the end
        xs.append(x_axis_max)
        ys.append(prev_y)

        ax_value.plot(xs, ys, linestyle=linestyle_all, label=method, linewidth=5.0)

    ax_value.set_ylabel("% of properties verified", fontsize=22)
    ax_value.set_xlabel("Computation time [s]", fontsize=22)
    plt.xscale('log', nonposx='clip')
    ax_value.legend(fontsize=22)
    plt.grid(True)
    plt.show()


def create_varying_epsilon_tests():

    folder = "batch_verification_results/"
    problem = "base_easy"
    method = pd.read_pickle(folder + f"jodie-{problem}.pkl").dropna(how="all")
    original_trial = method[1:2].copy()
    true_trial = original_trial.copy()
    n_trials = 9
    for i in range(n_trials):
        c_trial_eps = 2 ** (i + 1) * original_trial['Eps'][0:1]
        print(c_trial_eps)
        c_trial = original_trial.copy()
        c_trial['Eps'] = c_trial_eps
        true_trial = true_trial.append(c_trial, ignore_index=True)
    true_trial.to_pickle(folder + "true_trial.pkl")

    original_trial = method[1:2].copy()
    true_trial2 = original_trial.copy()
    n_trials = 19
    for i in range(n_trials):
        c_trial_eps = 0.01 * (i + 1) + original_trial['Eps'][0:1]
        print(c_trial_eps)
        c_trial = original_trial.copy()
        c_trial['Eps'] = c_trial_eps
        true_trial2 = true_trial2.append(c_trial, ignore_index=True)
    true_trial2.to_pickle(folder + "true_trial2.pkl")


def plot_from_tables_custom(folder, file_list, time_name_list, timeout, labels, fig_name, title, create_csv=False,
                            linestyles=None, one_vs_all=False):

    mpl.rcParams['font.size'] = 12
    tables = []
    if create_csv and not os.path.exists(folder + "/csv/"):
        os.makedirs(folder + "/csv/")
    for filename in file_list:
        m = pd.read_pickle(folder + filename).dropna(how="all")
        if create_csv:
            pd2csv(folder, folder + "/csv/", filename)
        tables.append(m)
    # keep only the properties in common
    for m in tables:
        if not one_vs_all:
            m["unique_id"] = m["Idx"].map(int).map(str) + "_" + m["prop"].map(int).map(str)
        else:
            m["unique_id"] = m["Idx"].map(int).map(str)
        m["Eps"] = m["Idx"].map(float)
    for m1, m2 in itertools.product(tables, tables):
        m1.drop(m1[(~m1['unique_id'].isin(m2["unique_id"]))].index, inplace=True)

    timings = []
    for idx in range(len(tables)):
        timings.append([])
        for i in tables[idx][time_name_list[idx]].values:
            if i >= timeout:
                timings[-1].append(float('inf'))
            else:
                timings[-1].append(i)
        timings[-1].sort()
    # check that they have the same length.
    for m1, m2 in itertools.product(timings, timings):
        assert len(m1) == len(m2)
    print(len(m1))

    starting_point = timings[0][0]
    for timing in timings:
        starting_point = min(starting_point, timing[0])

    fig = plt.figure(figsize=(6, 6))
    ax_value = plt.subplot(1, 1, 1)
    ax_value.axhline(linewidth=3.0, y=100, linestyle='dashed', color='grey')

    y_min = 0
    y_max = 100
    ax_value.set_ylim([y_min, y_max + 5])

    min_solve = float('inf')
    max_solve = float('-inf')
    for timing in timings:
        min_solve = min(min_solve, min(timing))
        finite_vals = [val for val in timing if val != float('inf')]
        if len(finite_vals) > 0:
            max_solve = max(max_solve, max([val for val in timing if val != float('inf')]))

    axis_min = starting_point
    axis_min = min(0.5 * min_solve, 1)
    ax_value.set_xlim([axis_min, timeout + 1])

    for idx, (clabel, timing) in enumerate(zip(labels, timings)):
        # Make it an actual cactus plot
        xs = [axis_min]
        ys = [y_min]
        prev_y = 0
        for i, x in enumerate(timing):
            if x <= timeout:
                # Add the point just before the rise
                xs.append(x)
                ys.append(prev_y)
                # Add the new point after the rise, if it's in the plot
                xs.append(x)
                new_y = 100 * (i + 1) / len(timing)
                ys.append(new_y)
                prev_y = new_y
        # Add a point at the end to make the graph go the end
        xs.append(timeout)
        ys.append(prev_y)

        linestyle = linestyles[idx] if linestyles is not None else "solid"
        ax_value.plot(xs, ys, color=colors[idx], linestyle=linestyle, label=clabel, linewidth=3.0)

    ax_value.set_ylabel("% of properties verified", fontsize=15)
    ax_value.set_xlabel("Computation time [s]", fontsize=15)
    plt.xscale('log', nonposx='clip')
    ax_value.legend(fontsize=9.5)
    plt.grid(True)
    plt.title(title)
    plt.tight_layout()

    figures_path = "./plots/"
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)
    plt.savefig(figures_path + fig_name, format='pdf', dpi=300)


def plot_vnn_results():

    ## MNIST-eth
    # folder = './mnist_results/'
    # bases = ["mnist_0.1", "mnist_0.3"]
    # for base in bases:
    #     file_list = [
    #         # f"{base}_KW_prox_100-pinit-eta100.0-feta100.0.pkl",
    #         f"{base}_KW_adam_175-pinit-ilr0.1,flr0.001.pkl",
    #         # f"{base}_KW_gurobi.pkl",
    #         # f"{base}_KW_gurobi-anderson_1.pkl",
    #     ]
    #     time_base = "BTime_KW"
    #     time_name_list = [
    #         # f"{time_base}_prox_100",
    #         f"{time_base}_adam_175",
    #         # f"{time_base}_gurobi",
    #         # f"{time_base}_gurobi-anderson_1",
    #     ]
    #     labels = [
    #         # "Proximal BaBSR",
    #         "Adam-BaBSR",
    #         # "Gurobi-BaBSR",
    #         # "Gur-And-BaBSR",
    #     ]
    #     timeout = 300
    #     plot_from_tables_custom(folder, file_list, time_name_list, timeout, labels, base + ".pdf", f"MNIST ETH: {base}",
    #                             create_csv=True)

    # ## CIFAR-eth
    # folder = './cifar_results/'
    # bases = ["cifar10_2_255", "cifar10_8_255"]
    # for base in bases:
    #     file_list = [
    #         # f"{base}_KW_prox_100-pinit-eta1.0-feta1.0.pkl",
    #         f"{base}_KW_adam_160-pinit-ilr0.0001,flr1e-06.pkl",
    #         # f"{base}_KW_gurobi.pkl",
    #         # f"{base}_KW_gurobi-anderson_1.pkl",
    #     ]
    #     time_base = "BTime_KW"
    #     time_name_list = [
    #         # f"{time_base}_prox_100",
    #         f"{time_base}_adam_160",
    #         # f"{time_base}_gurobi",
    #         # f"{time_base}_gurobi-anderson_1",
    #     ]
    #     labels = [
    #         # "Prox-BaBSR",
    #         "Supergradient BaBSR",
    #         # "Gurobi-BaBSR",
    #         # "Gur-And-BaBSR",
    #     ]
    #     timeout = 300
    #     plot_from_tables_custom(folder, file_list, time_name_list, timeout, labels, base + ".pdf", f"CIFAR ETH: {base}",
    #                             create_csv=True)

    ## CIFAR-eran
    folder = './mnist_results/'
    bases = ["mnist_convSmallRELU__Point"]
    for base in bases:
        file_list = [
            # f"{base}_KW_prox_100-pinit-eta100.0-feta100.0.pkl",
            # f"{base}_KW_bigm-adam_100-pinit-ilr0.001,flr1e-06.pkl",
            f"{base}_KW_cut_100-pinit-ilr0.001,flr1e-06-diilr0.1,diflr0.001.pkl",
            # f"{base}_KW_gurobi.pkl",
            # f"{base}_KW_gurobi-anderson_1.pkl",
        ]
        time_base = "BBran_KW"
        time_name_list = [
            # f"{time_base}_prox_100",
            # f"{time_base}_bigm-adam_100",
            f"{time_base}_cut_100",
            # f"{time_base}_gurobi",
            # f"{time_base}_gurobi-anderson_1",
        ]
        labels = [
            "Prox-BaBSR",
            # "Bigm Adam BaBSR",
            # "Cut BaBSR",
            # "Gurobi-BaBSR",
            # "Gur-And-BaBSR",
        ]
        timeout = 3600
        plot_from_tables_custom(folder, file_list, time_name_list, timeout, labels, base + ".pdf", f"CIFAR ERAN: {base}",
                                create_csv=True)

    # folder = './cifar_results/'
    # bases = ["base_100", "wide_100", "deep_100"]
    # for base in bases:
    #     file_list = [
    #         f"{base}_GNN_prox_100-pinit-eta100.0-feta100.0.pkl",
    #     ]
    #     time_base = "BTime_GNN"
    #     time_name_list = [
    #         f"{time_base}_prox_100",
    #     ]
    #     labels = [
    #         "Proximal BaBGNN",
    #     ]
    #     timeout = 3600
    #     plot_from_tables_custom(folder, file_list, time_name_list, timeout, labels, base + ".pdf", f"CIFAR OVAL: {base}",
    #                             create_csv=True)

    # plt.show()


# TODO: this plot averages over timed-out subproblems as well
def to_latex_table(folder, bases, file_list, time_name_list, labels, plot_names, timeout, one_vs_all=False):

    # latex_tables
    latex_tables = []
    for base, plt_name in zip(bases, plot_names):
        # Create latex table.
        tables = []
        for filename in file_list:
            m = pd.read_pickle(folder + f"{base}_" + filename).dropna(how="all")
            tables.append(m)
        # keep only the properties in common
        for m in tables:
            if not one_vs_all:
                m["unique_id"] = m["Idx"].map(int).map(str) + "_" + m["prop"].map(int).map(str)
            else:
                m["unique_id"] = m["Idx"].map(int).map(str)
            m["Eps"] = m["Idx"].map(float)
        for m1, m2 in itertools.product(tables, tables):
            m1.drop(m1[(~m1['unique_id'].isin(m2["unique_id"]))].index, inplace=True)

        # Set all timeouts to <timeout> seconds.
        for table, time_name in zip(tables, time_name_list):
            for column in table:
                if "SAT" in column:
                    table.loc[table[column] == 'timeout', time_name] = timeout

        full_table = tables[0]
        for c_table in tables[1:]:
            full_table = pd.merge(full_table, c_table, on=['Idx', 'prop', 'Eps', 'unique_id'], how='inner')

        # Create summary table.
        summary_dict = {}
        for column in full_table:
            if column not in ['Idx', 'prop', 'Eps', 'unique_id']:
                if "SAT" not in column:
                    c_mean = full_table[column].mean()
                    summary_dict[column] = c_mean
                else:
                    # Handle SAT status
                    n_timeouts = len(full_table.loc[full_table[column] == 'timeout'])
                    m_len = len(full_table)
                    summary_dict[column + "_perc_timeout"] = n_timeouts / m_len * 100

        # Re-sort by method, exploiting that the columns are ordered per method.
        latex_table_dict = {}
        for counter, key in enumerate(summary_dict):
            c_key = key.split("_")[0]
            if c_key in latex_table_dict:
                latex_table_dict[c_key].append(summary_dict[key])
            else:
                latex_table_dict[c_key] = [summary_dict[key]]

        latex_table = pd.DataFrame(latex_table_dict)
        latex_table = latex_table.rename(columns={"BSAT": "%Timeout", "BBran": "Sub-problems", "BTime": "Time [s]"})
        latex_tables.append(latex_table[latex_table.columns[::-1]])

    merged_plot_names = [cname.split(" ")[0] for cname in plot_names]
    merged_latex_table = pd.concat(latex_tables, axis=1, keys=merged_plot_names)
    print(merged_latex_table)
    converted_latex_table = merged_latex_table.to_latex(float_format="%.2f")
    print(f"latex table.\n Row names: {labels} \n Table: \n{converted_latex_table}")


def iclr_plots():

    # Plots for OVAL-CIFAR
    folder = './cifar_results/'
    timeout = 3600
    bases = ["base_100", "wide_100", "deep_100"]
    plot_names = ["Base model", "Wide large model", "Deep large model"]
    time_base = "BTime_KW"
    file_list_nobase = [
        "KW_prox_100-pinit-eta100.0-feta100.0.pkl",
        "KW_bigm-adam_180-pinit-ilr0.01,flr0.0001.pkl",
        "KW_cut_100_no_easy-pinit-ilr0.001,flr1e-06-cut_add2.0-diilr0.01,diflr0.0001.pkl",
        "KW_cut_100-pinit-ilr0.001,flr1e-06-cut_add2.0-diilr0.01,diflr0.0001.pkl",
        "anderson-mip.pkl",
        "KW_gurobi-anderson_1.pkl",
        "eran.pkl"
    ]
    time_name_list = [
        f"{time_base}_prox_100",
        f"{time_base}_bigm-adam_180",
        f"{time_base}_cut_100_no_easy",
        f"{time_base}_cut_100",
        f"BTime_anderson-mip",
        f"{time_base}_gurobi-anderson_1",
        f"{time_base}_eran"
    ]
    labels = [
        "BDD+ BaBSR",
        "Big-M BaBSR",
        "Active Set BaBSR",
        "Big-M + Active Set BaBSR",
        r"MIP $\mathcal{A}_k$",
        "G. Planet + G. 1 cut BaBSR",
        "ERAN"
    ]
    line_styles = [
        "dotted",
        "solid",
        "solid",
        "solid",
        "dotted",
        "dotted",
        "dotted",
    ]
    for base, plt_name in zip(bases, plot_names):
        file_list = [f"{base}_" + cfile for cfile in file_list_nobase]
        plot_from_tables_custom(folder, file_list, time_name_list, timeout, labels, base + ".pdf", plt_name,
                                create_csv=False, linestyles=line_styles)
    to_latex_table(folder, bases, file_list_nobase, time_name_list, labels, plot_names, timeout)
    plt.show()


def mnist_plots():
    # Plots for MNIST (Anderson-based methods don't pay off on these datasets)
    folder = './mnist_results/'
    timeout = 3600
    # bases = ["mnist_0.3"]
    # plot_names = ["MNIST large model"]
    bases = ["mnist_base_kw", "mnist_wide_kw", "mnist_deep_kw"]
    plot_names = ["MNIST base model", "MNIST wide model", "MNIST deep model"]
    time_base = "BTime_KW"
    file_list_nobase = [
        "KW_adam_160-pinit-ilr0.1,flr0.001.pkl",  # adam planet seems to be better than prox, here
        # "KW_prox_100-pinit-eta10.0-feta10.0.pkl",
        "KW_bigm-adam_180-pinit-ilr0.1,flr0.001.pkl",
        # "KW_cut_100_no_easy-pinit-ilr0.01,flr1e-05-cut_add2-diilr0.1,diflr0.001.pkl",
        "KW_cut_100-pinit-ilr0.01,flr1e-05-cut_add2-diilr0.1,diflr0.001.pkl",
    ]
    time_name_list = [
        f"{time_base}_adam_160",
        # f"{time_base}_prox_100",
        f"{time_base}_bigm-adam_180",
        # f"{time_base}_cut_100_no_easy",
        f"{time_base}_cut_100",
    ]
    labels = [
        "BDD+ BaBSR",
        # "BDD+ BaBSR",
        "Big-M BaBSR",
        # "Active Set BaBSR",
        "Big-M + Active Set BaBSR",
    ]
    line_styles = [
        "dotted",
        "solid",
        # "solid",
        "solid",
    ]
    for base, plt_name in zip(bases, plot_names):
        file_list = [f"{base}_" + cfile for cfile in file_list_nobase]
        plot_from_tables_custom(folder, file_list, time_name_list, timeout, labels, base + ".pdf", plt_name,
                                create_csv=False, linestyles=line_styles, one_vs_all=True)
    # to_latex_table(folder, bases, file_list_nobase, time_name_list, labels, plot_names, timeout, one_vs_all=True)
    plt.show()


if __name__ == "__main__":

    ###############################################################
    ## CIFAR-eran
    ###############################################################
    # folder = './mnist_results/'
    # bases = ["mnist_convSmallRELU__Point"]
    # for base in bases:
    #     file_list = [
    #         f"{base}_KW_prox_100-pinit-eta100.0-feta100.0.pkl",
    #         f"{base}_KW_bigm-adam_100-pinit-ilr0.001,flr1e-06.pkl",
    #         f"{base}_KW_cut_100-pinit-ilr0.001,flr1e-06-diilr0.1,diflr0.001.pkl",
    #         f"{base}_KW_poly.pkl",
    #         # f"{base}_KW_gurobi.pkl",
    #         # f"{base}_KW_gurobi-anderson_1.pkl",
    #     ]
    #     time_base = "BTime_KW"
    #     time_name_list = [
    #         f"{time_base}_prox_100",
    #         f"{time_base}_bigm-adam_100",
    #         f"{time_base}_cut_100",
    #         f"Time",
    #         # f"{time_base}_gurobi",
    #         # f"{time_base}_gurobi-anderson_1",
    #     ]
    #     labels = [
    #         "Prox-BaBSR",
    #         "Bigm Adam BaBSR",
    #         "Cut BaBSR",
    #         "Poly BaBSR",
    #         # "Gurobi-BaBSR",
    #         # "Gur-And-BaBSR",
    #     ]
    #     timeout = 3600
    #     plot_from_tables_custom(folder, file_list, time_name_list, timeout, labels, base + ".pdf", f"CIFAR ERAN: {base}",
    #                                 create_csv=False)


    # plot_spfw = True
    # if plot_spfw:
    #     # bases = ["base_100", "wide_100", "deep_100"]
    #     bases = ["base_100", "wide_100"]
    #     for base in bases:
    #         file_list = [
    #             f"{base}_KW_prox_100-pinit-eta100.0-feta100.0.pkl",
    #             f"{base}_KW_bigm-adam_180-pinit-ilr0.01,flr0.0001.pkl",
    #             f"{base}_KW_cut_600-pinit-ilr0.001,flr1e-06-cut_add2.0-diilr0.01,diflr0.0001.pkl",
    #             # f"{base}_KW_cut_600_no_easy-pinit-ilr0.001,flr1e-06-cut_add2.0-diilr0.01,diflr0.0001.pkl",
    #             f"{base}_KW_sp-fw_1000-pinit-fw_start10.0-diilr0.01,diflr0.0001.pkl",
    #             # f"{base}_KW_sp-fw_1000_no_hardthr-pinit-fw_start10.0-diilr0.01,diflr0.0001.pkl",  # TODO: use back
    #             # f"{base}_KW_sp-fw_1000_no_easy-pinit-fw_start10.0-diilr0.01,diflr0.0001.pkl",
    #             # f"{base}_KW_sp-fw_500-pinit-fw_start10.0-diilr0.01,diflr0.0001.pkl",
    #             # f"{base}_KW_sp-fw_1000-pinit-fw_start100.0-diilr0.01,diflr0.0001.pkl",
    #             # f"{base}_KW_cut-sp-fw_100-900-pinit-ilr0.001,flr1e-06-cut_add2.0-fw_start100.0-diilr0.01,diflr0.0001.pkl",
    #         ]
    #         time_base = "BTime_KW"
    #         time_name_list = [
    #             f"{time_base}_prox_100",
    #             f"{time_base}_bigm-adam_180",
    #             f"{time_base}_cut_600",
    #             # f"{time_base}_cut_600_no_easy",
    #             f"{time_base}_sp-fw_1000",
    #             # f"{time_base}_sp-fw_1000_no_hardthr",
    #             # f"{time_base}_sp-fw_1000_no_easy",
    #             # f"{time_base}_sp-fw_500",
    #             # f"{time_base}_sp-fw_1000",
    #             # f"{time_base}_cut-sp-fw_100-900",
    #         ]
    #         labels = [
    #             "BDD+ BaBSR",
    #             "Big-M BaBSR",
    #             "Big-M + Active Set 600 BaBSR",
    #             # "Active Set 600 BaBSR",
    #             "Big-M + SP-FW 1k BaBSR",
    #             # "Big-M + SP-FW 1k nohard BaBSR",
    #             # "SP-FW 1k BaBSR",
    #             # "Big-M + SP-FW 500 BaBSR",
    #             # "Big-M + SP-FW 1k 100_start BaBSR",
    #             # "Big-M + AS + SP-FW 1k BaBSR",
    #         ]
    #         timeout = 3600
    #         plot_from_tables_custom(folder, file_list, time_name_list, timeout, labels, base + ".pdf",
    #                                 f"CIFAR OVAL: {base}", create_csv=False)
    # plt.show()
    # import pdb; pdb.set_trace()

    """
    Comments: 
    - SP-FW 1k (with 10 start) sucks on wide and deep. 100 start won't help, I think. 
    Wide would improve without Harkirat's heuristic. 
    Deep would improve with smaller impr_thresh (1e-2?) 
    
    - Re-run both cuts and SP-FW without Harkirat's heuristic
    
    - Experiments with 600 + 3000? Trying AS 1650 first.
    
    - Experiments with fewer AS iters don't pay off.
    """

    iclr_plots()

    # plt.show()