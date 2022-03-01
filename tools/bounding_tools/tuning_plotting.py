import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
from tools.plot_utils import custom_plot


def time_vs_bounds_plot():
    # Plot time vs. bound plots, assumes pickle structure of anderson_bound_per_time

    folder = "./anderson_results/time_vs_bounds/"
    bigm_prox_file = folder + "expLP-cpu-bigmprox-eta1-ACAS-prop1-1_1_data.pth" #"expLP-cpu-bigmprox-etasmaller-ACAS-prop1-1_1_data.pth"  #"expLP-cpu-bigmprox-eta1-ACAS-prop1-1_1_data.pth"
    gurobi_pla_file = folder + "gurobi-planet-6threads-ACAS-prop1-1_1_data.pth"
    proxlp_file = folder + "proxLP-cpu-ACAS-prop1-1_1_data.pth" #"proxLP-cpu-naiveinit-ACAS-prop1-1_1_data.pth"
    bigm_subg_file = folder + "expLP-cpu-bigm-subgradient-ACAS-prop1-1_1_data.pth"

    bigm_prox_measurements = torch.load(bigm_prox_file)
    proxlp_measurements = torch.load(proxlp_file)
    gurobi_pla_measurements = torch.load(gurobi_pla_file)
    bigm_subg_measurements = torch.load(bigm_subg_file)

    bigmprox_lbs = bigm_prox_measurements['lbs']
    bigmprox_time = bigm_prox_measurements['times']
    bigmprox_ubs = bigm_prox_measurements['ubs']
    n_bigmprox_lines = len(bigmprox_lbs)
    n_proxlp_lines = len(proxlp_measurements['lbs'])

    ylog = False  # bounds can be negative, so bad idea (don't want to rescale)
    xlog = True

    n_gurobi_pla_threads = list(gurobi_pla_measurements['lbs'].keys())[0]
    plt.figure(0)
    plt.axvline(gurobi_pla_measurements['times'][n_gurobi_pla_threads], color=colors[n_bigmprox_lines + n_proxlp_lines + 1],
              label="Gurobi PLANET", dashes=(5,5), lw=1)
    plt.axhline(gurobi_pla_measurements['lbs'][n_gurobi_pla_threads], color=colors[n_bigmprox_lines + n_proxlp_lines + 1],
              dashes=(5,5), lw=1)
    plt.subplots_adjust(left=0.17)
    plt.figure(1)
    plt.axvline(gurobi_pla_measurements['times'][n_gurobi_pla_threads], color=colors[n_bigmprox_lines + n_proxlp_lines + 1],
                label="Gurobi PLANET", dashes=(5, 5), lw=1)
    plt.axhline(gurobi_pla_measurements['ubs'][n_gurobi_pla_threads], color=colors[n_bigmprox_lines + n_proxlp_lines + 1],
                dashes=(5, 5), lw=1)
    plt.subplots_adjust(left=0.15)

    for id, key in enumerate(bigmprox_lbs):
        custom_plot(0, bigmprox_time[key], bigmprox_lbs[key], None, "Time [s]", "Lower Bound", "Lower bound vs time", errorbars=False,
                    labelname=r"big-m Prox $\eta=$" + f"{key[0]}" + r", $it_{inner}=$" + f"{key[1]}", dotted="-", xlog=xlog,
                    ylog=ylog, color=colors[id])
        custom_plot(1, bigmprox_time[key], bigmprox_ubs[key], None, "Time [s]", "Upper Bound", "Upper bound vs time",
                    errorbars=False,
                    labelname=r"big-m Prox $\eta=$" + f"{key[0]}" + r", $it_{inner}=$" + f"{key[1]}", dotted="-", xlog=xlog,
                    ylog=ylog, color=colors[id])

    for id, key in enumerate(proxlp_measurements['lbs']):
        custom_plot(0, proxlp_measurements['times'][key], proxlp_measurements['lbs'][key], None, "Time [s]", "Lower Bound", "Lower bound vs time",
                    errorbars=False,
                    labelname=r"ProxLP $\eta=$" + f"{key[0]}" + r", $it_{inner}=$" + f"{key[1]}", dotted="-", xlog=xlog,
                    ylog=ylog, color=colors[n_bigmprox_lines + id])
        custom_plot(1, proxlp_measurements['times'][key], proxlp_measurements['ubs'][key], None, "Time [s]", "Upper Bound", "Upper bound vs time",
                    errorbars=False,
                    labelname=r"ProxLP $\eta=$" + f"{key[0]}" + r", $it_{inner}=$" + f"{key[1]}", dotted="-", xlog=xlog,
                    ylog=ylog, color=colors[n_bigmprox_lines + id])

    for id, key in enumerate(bigm_subg_measurements['lbs']):
        custom_plot(0, bigm_subg_measurements['times'][key], bigm_subg_measurements['lbs'][key], None, "Time [s]",
                    "Lower Bound", "Lower bound vs time", errorbars=False, labelname="big-M adam", dotted="-", xlog=xlog,
                    ylog=ylog, color=colors[n_bigmprox_lines + n_proxlp_lines + 2 + id])
        custom_plot(1, bigm_subg_measurements['times'][key], bigm_subg_measurements['ubs'][key], None, "Time [s]",
                    "Upper Bound", "Upper bound vs time", errorbars=False, labelname="big-M adam", dotted="-", xlog=xlog,
                    ylog=ylog, color=colors[n_bigmprox_lines + n_proxlp_lines + 2 + id])

    # this is a bit faster than KW init this context
    # proxlp_file = folder + "proxLP-cpu-naiveinit-ACAS-prop1-1_1_data.pth"
    # proxlp_measurements = torch.load(proxlp_file)
    # for id, key in enumerate(proxlp_measurements['lbs']):
    #     custom_plot(0, proxlp_measurements['times'][key], proxlp_measurements['lbs'][key], None, "Time [s]", "Lower Bound", "Lower bound vs time",
    #                 errorbars=False,
    #                 labelname=r"Naive init ProxLP $\eta=$" + f"{key[0]}" + r", $it_{inner}=$" + f"{key[1]}", dotted="-", xlog=xlog,
    #                 ylog=ylog, color=colors[-1])
    #     custom_plot(1, proxlp_measurements['times'][key], proxlp_measurements['ubs'][key], None, "Time [s]", "Upper Bound", "Upper bound vs time",
    #                 errorbars=False,
    #                 labelname=r"Naive init ProxLP $\eta=$" + f"{key[0]}" + r", $it_{inner}=$" + f"{key[1]}", dotted="-", xlog=xlog,
    #                 ylog=ylog, color=colors[-1])

    # this is terribly slow in this context
    # proxlp_subg_file = folder + "proxLP-cpu-subgradient-ACAS-prop1-1_1_data.pth"  # "proxLP-cpu-vanillasubg-ACAS-prop1-1_1_data.pth"
    # proxlp_subg_measurements = torch.load(proxlp_subg_file)
    # for id, key in enumerate(proxlp_subg_measurements['lbs']):
    #     custom_plot(0, proxlp_subg_measurements['times'][key], proxlp_subg_measurements['lbs'][key], None, "Time [s]",
    #                 "Lower Bound", "Lower bound vs time", errorbars=False, labelname=r"ProxLP adam",
    #                 dotted="-", xlog=xlog, ylog=ylog, color=colors[-2])
    #     custom_plot(1, proxlp_subg_measurements['times'][key], proxlp_subg_measurements['ubs'][key], None, "Time [s]",
    #                 "Upper Bound", "Upper bound vs time", errorbars=False, labelname=r"ProxLP adam",
    #                 dotted="-", xlog=xlog, ylog=ylog, color=colors[-2])


def adam_cifar_comparison():
    # old planet adam vs big-m adam. Needs to be rerun after the adam fix (big-m will be beaten)

    img_idx = 0 #50  # 0
    step_size = 1e-3

    pickle_name = "../results/icml20/timings-img{}-{},stepsize:{}.pickle"

    planet = torch.load(pickle_name.format(img_idx, "planet-adam", step_size), map_location=torch.device('cpu'))
    custom_plot(0, planet.get_last_layer_time_trace(), planet.get_last_layer_bounds_means_trace(first_half_only_as_ub=True),
                None, "Time [s]", "Upper Bound", "Upper bound vs time", errorbars=False,
                labelname=r"Planet ADAM $\alpha=$" + f"{step_size}", dotted="-", xlog=False,
                ylog=False, color=colors[0])

    bigm = torch.load(pickle_name.format(img_idx, "bigm-adam", step_size), map_location=torch.device('cpu'))
    custom_plot(0, bigm.get_last_layer_time_trace(), bigm.get_last_layer_bounds_means_trace(first_half_only_as_ub=True),
                None, "Time [s]", "Upper Bound", "Upper bound vs time", errorbars=False,
                labelname=r"Big-M ADAM $\alpha=$" + f"{step_size}", dotted="-", xlog=False,
                ylog=False, color=colors[1])


def adam_vs_prox():
    # adam bounds vs prox bounds over time (KW initialization)

    img = 10

    folder = "../../Mini-Projects/convex-relaxations/results/icml20/"
    adam_name = folder + f"timings-img{img}-planet-adam,istepsize:0.01,fstepsize1e-06.pickle"
    prox_name = folder + f"timings-img{img}-proxlp,eta:100.0-feta100.0-mom0.0.pickle"

    momentum = 0.4
    optimized_prox_name = folder + f"timings-img{img}-proxlp,eta:1000.0-feta1000.0-mom{momentum}.pickle"

    adam = torch.load(adam_name, map_location=torch.device('cpu'))
    custom_plot(0, adam.get_last_layer_time_trace(), adam.get_last_layer_bounds_means_trace(first_half_only_as_ub=True),
                None, "Time [s]", "Upper Bound", "Upper bound vs time", errorbars=False,
                labelname=r"ADAM $\alpha \in$" + "[1e-3, 1e-6]", dotted="-", xlog=False,
                ylog=False, color=colors[0])

    prox = torch.load(prox_name, map_location=torch.device('cpu'))
    custom_plot(0, prox.get_last_layer_time_trace(), prox.get_last_layer_bounds_means_trace(first_half_only_as_ub=True),
                None, "Time [s]", "Upper Bound", "Upper bound vs time", errorbars=False,
                labelname=r"GS Prox $\eta=$" + f"1e2", dotted="-", xlog=False,
                ylog=False, color=colors[1])

    optimized_prox = torch.load(optimized_prox_name, map_location=torch.device('cpu'))
    custom_plot(0, optimized_prox.get_last_layer_time_trace(),
                optimized_prox.get_last_layer_bounds_means_trace(first_half_only_as_ub=True),
                None, "Time [s]", "Upper Bound", "Upper bound vs time", errorbars=False,
                labelname=r"Momentum Prox $\eta=$" + f"1e2", dotted="-", xlog=False,
                ylog=False, color=colors[2])


def five_plots_comparison():

    define_linear_approximation = False

    cut_algorithms = [
        # algo, inlr, finlr, init_out_iters, cut_frequency, max_cuts, cut_add, init_init_step, init_fin_step
        # ("cut", 100, 1e-3, 1e-6, 100, 450, 3, 10, 1e-2, 1e-4),
        # ("cut", 100, 1e-3, 1e-6, 100, 450, 3, 10, 1e-1, 1e-3),
        # ("cut", 100, 1e-3, 1e-6, 1000, 450, 3, 10, 1e-1, 1e-3),##best option for cifar10_8_255 with 100 or 1000 iters
        # ("cut", 500, 1e-3, 1e-6, 1000, 450, 3, 10, 1e-1, 1e-3),##best option for cifar10_8_255 with 100 or 1000 iters
        # ("cut", 1000, 1e-3, 1e-6, 1000, 450, 3, 10, 1e-1, 1e-3),##best option for cifar10_8_255 with 100 or 1000 iters
        # ("cut", 100, 1e-3, 1e-6, 500, 450, 3, 10, 1e-1, 1e-3), ##best for mnist_convSmallRELU__Point, cifar10_convSmallRELU__Point
    ]

    images = [0, 5, 10, 15, 20]
    images = [0, 1, 5, 10, 25, 50, 75]
    images = [0, 5, 10, 25, 50, 75]
    # images = [0]
    adam_algorithms = [
        # algo, beta1, inlr, finlr
        # ("bigm-adam-simplex", 0.9, 1e-2, 1e-3),
        # ("bigm-adam-simplex", 0.9, 1e-2, 1e-4),
        # ("bigm-adam-simplex", 0.9, 1e-2, 1e-5),
        # ("bigm-adam-simplex", 0.9, 1e-2, 1e-6),
        # ("bigm-adam-simplex", 0.9, 1e-3, 1e-3),
        ("bigm-adam-simplex", 0.9, 1e-3, 1e-4),
        # ("bigm-adam-simplex", 0.9, 1e-3, 1e-5),
        # ("bigm-adam-simplex", 0.9, 1e-3, 1e-6),
        # ("bigm-adam-simplex", 0.9, 0.0002, 0.0001),
        # ("bigm-adam-simplex", 0.9, 0.0003, 0.0001),
        ("bigm-adam-simplex", 0.9, 1e-4, 1e-4),
        ("bigm-adam-simplex", 0.9, 1e-4, 0.00011),
        # ("bigm-adam-simplex", 0.9, 1e-4, 1e-5),
        # ("simplex", 0.9, 1e-3, 1e-6),
        ("baseline-bigm-adam-simplex", 0.9, 1e-2, 1e-4),
        # ("simplex", 0.9, 1e-3, 1e-5),

        ("gurobi-simplex", 0.9, 0.001, 0.001),
        ("gurobi-baseline-planet-simplex", 0.9, 0.001, 0.001),
    ]
    prox_algorithms = [  # momentum tuning
        # algo, momentum, ineta, fineta
        # ("proxlp", 0.0, 1e2, 1e2),  # Best for mnist
        # ("proxlp", 0.0, 1e2, 1e2),  # Best for mnist, mnist_convSmallRELU__Point
        # ("proxlp", 0.0, 1e1, 1e1), # Best for cifar10_8_255 and cifar10_2_255
        # ("proxlp", 0.0, 1e3, 1e3),
        # ("proxlp", 0.0, 1e0, 1e0),  # Best for cifar10_2_255 and cifar10_8_255
        # ("proxlp", 0.0, 1e4, 1e4),
        # ("proxlp", 0.0, 5e1, 1e2),
        # ("proxlp", 0.0, 5e2, 1e3),
        # ("proxlp", 0.0, 5e0, 1e1),
        # ("proxlp", 0.0, 5e3, 1e4),
    ]

    simplex_algorithms = [
        # algo, inlr, finlr, init_out_iters, cut_frequency, max_cuts, cut_add, init_init_step, init_fin_step
        # ("simplex", 185, 1e-3, 1e-5, 50, 450, 10, 1, 1e-2, 1e-4),
        # ("simplex", 185, 1e-3, 1e-5, 10, 450, 10, 1, 1e-2, 1e-4),
        # ("simplex", 185, 1e-3, 1e-5, 50, 50, 10, 1, 1e-2, 1e-4),
        # ("simplex", 1000, 1e-3, 1e-5, 50, 300, 10, 1, 1e-2, 1e-4),
    ]
    folder = "./timings_cifar/"

    ##### mnist-eth #################
    # net = "mnist_0.1"  # mnist_0.1 or mnist_0.3 
    # folder = f"./timings_mnist/{net}_"
    # if net == 'mnist_0.1':
    #     images = [18, 92]
    # elif net == 'mnist_0.3':
    #     images = [7, 19, 44, 58, 68, 69, 78, 80, 93, 95]
    # net = "mnist_0.1"  # mnist_0.1 or mnist_0.3 
    # folder = f"./timings_mnist/{net}_"
    # if net == 'mnist_0.1':
    #     images = [18, 92]
    # elif net == 'mnist_0.3':
    #     images = [7, 19, 44, 58, 68, 69, 78, 80, 93, 95]
    ##################################

    ##### cifar10-eth #################
    # net = "cifar10_8_255"   # cifar10_8_255 or cifar10_2_255
    # folder = f"./timings_cifar10/{net}_"
    # images = list(range(19)) # all 100 images for workshop
    # if net == 'cifar10_8_255':
    #     images = [4,38,49,55,89,96]
    # elif net == 'cifar10_2_255':
    #     images = [5,39,77,93]
    #     images = [2,5,10,14,15,28,45,51,64,66,71,72,73,76,80,84,97]
    ##################################


    ###########   ERAN   #############
    # data = "mnist"#cifar10
    # net = "mnist_convSmallRELU__Point"  # mnist_convSmallRELU__Point, mnist_convSmallRELU__DiffAI, mnist_convSmallRELU__PGDK
    # folder = f"./timings_mnist/{net}_"
    # images = list(range(100)) # all 100 images for workshop
    # # images=[1,2,3,4,5,11,23,57,85,97]
    # images=[4,7,21,26,35,40,44,58,59,66,84]
    
    # data = "cifar10"
    # net = "cifar10_convSmallRELU__Point"  # cifar10_convSmallRELU__Point, cifar10_convSmallRELU__DiffAI, cifar10_convSmallRELU__PGDK
    # folder = f"./timings_cifar10/{net}_"
    # images=[12,14,23,28,34,40,44,48,54,77,92]
    ##################################

    algorithm_name_dict = {
        "planet-adam": "ADAM",
        "planet-auto-adagrad": "AdaGrad",
        "planet-auto-adam": "Autograd's ADAM",
        "proxlp": "Proximal",
        "jacobi-proxlp": "Jacobi Proximal",
        "dj-adam": "Dvijotham ADAM",
        "cut": "Cut",
        "bigm-adam-simplex": "1 simplex-constraint",
        "simplex": "simplex",
        "baseline-bigm-adam-simplex": "no simplex-constraint",
        #
        "gurobi-simplex": "gurobi 1 simplex-constraint"
    }

    lin_approx_string = "" if not define_linear_approximation else "-allbounds"

    fig_idx = 0
    for img in images:
        color_id = 0
        for algo, beta1, inlr, finlr in adam_algorithms:
            adam_name = folder + f"timings-img{img}-{algo},istepsize:{inlr},fstepsize:{finlr},beta1:{beta1}{lin_approx_string}.pickle"
            nomomentum = " w/o momentum" if beta1 == 0 else ""

            try:
                adam = torch.load(adam_name, map_location=torch.device('cpu'))
            except:
                continue
            custom_plot(fig_idx, adam.get_last_layer_time_trace(), adam.get_last_layer_bounds_means_trace(second_half_only=True),
                        None, "Time [s]", "Lower Bound", "Lower bound vs time", errorbars=False,
                        labelname=rf"{algorithm_name_dict[algo]} $\alpha \in$" + f"[{inlr}, {finlr}]" + nomomentum,
                        dotted="-", xlog=False,
                        ylog=False, color=colors[color_id])
            color_id += 1

        for algo, momentum, ineta, fineta in prox_algorithms:

            acceleration_string = ""
            if algo != "jacobi-proxlp":
                acceleration_string += f"-mom:{momentum}"
            prox_name = folder + f"timings-img{img}-{algo},eta:{ineta}-feta:{fineta}{acceleration_string}{lin_approx_string}.pickle"

            acceleration_label = ""
            if momentum:
                acceleration_label += f"momentum {momentum}"

            try:
                prox = torch.load(prox_name, map_location=torch.device('cpu'))
            except:
                continue
            custom_plot(fig_idx, prox.get_last_layer_time_trace(), prox.get_last_layer_bounds_means_trace(second_half_only=True),
                        None, "Time [s]", "Lower Bound", "Lower bound vs time", errorbars=False,
                        labelname=rf"{algorithm_name_dict[algo]} $\eta \in$" + f"[{ineta}, {fineta}], " +
                                  f"{acceleration_label}",
                        dotted="-", xlog=False, ylog=False, color=colors[color_id])
            color_id += 1

        for algo, iters_out, inlr, finlr, init_out_iters, cut_frequency, max_cuts, cut_add, init_init_step, init_fin_step in cut_algorithms:
            pickle_name = folder + f"timings-img{img}-{algo},outer:{iters_out},{inlr},{finlr}" \
                               f"cut:{cut_frequency},{max_cuts},{cut_add}" \
                               f"init:{init_out_iters}{init_init_step}{init_fin_step},{lin_approx_string}.pickle"

            try:
                adam = torch.load(pickle_name, map_location=torch.device('cpu'))
            except:
                print('could not find', pickle_name)
                continue
            custom_plot(fig_idx, adam.get_last_layer_time_trace(), adam.get_last_layer_bounds_means_trace(second_half_only=True),
                        None, "Time [s]", "Lower Bound", "Lower bound vs time", errorbars=False,
                        labelname=rf"{algorithm_name_dict[algo]} $\alpha \in$" + f"[{inlr}, {finlr}] init:{init_out_iters},{init_init_step},{init_fin_step}",
                        dotted="-", xlog=False,
                        ylog=False, color=colors[color_id])
            color_id += 1

        for algo, iters_out, inlr, finlr, init_out_iters, cut_frequency, max_cuts, cut_add, init_init_step, init_fin_step in simplex_algorithms:
            pickle_name = folder + f"timings-img{img}-{algo},outer:{iters_out},{inlr},{finlr}" \
                               f"cut:{cut_frequency},{max_cuts},{cut_add}" \
                               f"init:{init_out_iters}{init_init_step}{init_fin_step}.pickle"

            try:
                adam = torch.load(pickle_name, map_location=torch.device('cpu'))
            except:
                print('could not find', pickle_name)
                continue
            custom_plot(fig_idx, adam.get_last_layer_time_trace(), adam.get_last_layer_bounds_means_trace(second_half_only=True),
                        None, "Time [s]", "Lower Bound", "Lower bound vs time", errorbars=False,
                        labelname=rf"{algorithm_name_dict[algo]} $\alpha \in$" + f"[{inlr}, {finlr}] init:{init_out_iters},{init_init_step},{init_fin_step}",
                        dotted="-", xlog=False,
                        ylog=False, color=colors[color_id])
            color_id += 1
        fig_idx += 1
        plt.savefig('%d.png'%img)
        # plt.show()





if __name__ == "__main__":

    # time_vs_bounds_plot()
    # adam_cifar_comparison()
    # adam_vs_prox()
    five_plots_comparison()

    # plt.show()
