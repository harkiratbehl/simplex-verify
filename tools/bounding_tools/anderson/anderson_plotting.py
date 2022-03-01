import torch
import matplotlib.pyplot as plt
import matplotlib
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
from tools.plot_utils import custom_plot



def cifar_incomplete_comparison():
    define_linear_approximation = False

    images = [100]
    # images = [100, 200, 300, 400, 500, 600, 700]
    gurobi = [
        # algorithm, n_cuts
        # ("anderson-gurobi", 5000.),
        # ("anderson-gurobi", 1.),
        # ("anderson-gurobi", 2.),
        # ("anderson-gurobi", 3.),
        # ("anderson-gurobi", 4.),
        # ("anderson-gurobi", 10.),
    ]
    saddle_anderson = [

        # algo, outer_iters, M_factor, step_type, init_step, fin_step
        # ("saddle-explp", 20014, 1.1, "grad", 1e-2, 1e-3),  # 1.1 bigm factor, 1e-2 to 1e-3

        # ("saddle-explp", 800000, 1.1, "fw", None, None),  # as above, blockless, FW step size

        # ("saddle-explp", 40001, 1.1, "fw", None, None),  # as above, blockless, FW step size
        # ("saddle-explp", 40001, 1.5, "fw", None, None),  # as above, blockless, FW step size
        # ("saddle-explp", 40001, 1.05, "fw", None, None),  # as above, blockless, FW step size

        # ("saddle-explp", 100000, 1.02, "fw", None, None),  # as above, blockless, FW step size

        # ("saddle-explp", 40001, 1.1, "fw", None, None),  # as above, blockless, FW step size

        # ("saddle-explp", 40000, 1.01, "fw", None, None),  # as above, blockless, FW step size
        # ("saddle-explp", 40000, 1.02, "fw", None, None),  # as above, blockless, FW step size
        # ("saddle-explp", 40000, 1.0, "fw", None, None),  # as above, blockless, FW step size
        # ("saddle-explp", 40001, 1.0, "fw", None, None),  # as above, blockless, FW step size -- less primal init

        # ("saddle-explp", 6000, 1.0, "fw", None, None),  # as above, blockless, FW step size

        # new format
        # algo, outer_iters, M_factor, step_type, init_step, fin_step, init_type
        # ("saddle-explp", 2000, 1.0, "fw", None, None, "bigm"),  # as above, blockless, FW step size
        # ("saddle-explp", 1000, 1.0, "fw", None, None, "cuts"),  # as above, blockless, FW step size
        ("saddle-explp", 2000, 1.0, "fw", None, None, "bigm"),  # as above, blockless, FW step size
        ("saddle-explp", 1, 1.0, "fw", None, None, "cuts"),  # as above, blockless, FW step size

    ]

    lin_approx_string = "" if not define_linear_approximation else "-allbounds"

    algorithm_name_dict = {
        "anderson-gurobi": "Gurobi Anderson",
        "saddle-explp": "SP-Anderson"
    }

    fig_idx = 0
    for img in images:
        title = f"CIFAR-10, image {img}"
        folder = "./timings_cifar_anderson/"
        color_id = 0
        for algo, n_cuts in gurobi:
            gurobi_name = folder + f"timings-img{img}-{algo},n_cuts:{n_cuts}{lin_approx_string}.pickle"

            gur_log = torch.load(gurobi_name, map_location=torch.device('cpu'))
            custom_plot(fig_idx, gur_log.get_last_layer_time_trace(),
                        gur_log.get_last_layer_bounds_means_trace(first_half_only_as_ub=True),
                        None, "Time [s]", "Upper Bound", title, errorbars=False,
                        labelname=f"{algorithm_name_dict[algo]} cuts: {n_cuts}",
                        dotted="-", xlog=False, ylog=False, color=colors[color_id])
            color_id += 1

        for algo, outer_iters, M_factor, step_type, init_step, fin_step, init in saddle_anderson:

            step_str = f"step:{step_type}"
            if step_type == "grad":
                step_str += f",istepsize:{init_step},fstepsize:{fin_step}"

            log_name = folder + f"timings-img{img}-{algo},outer:{outer_iters},{step_str},M:{M_factor}" \
                                f",init:{init},{lin_approx_string}.pickle"

            saddle_log = torch.load(log_name, map_location=torch.device('cpu'))
            # custom_plot(fig_idx, saddle_log.get_last_layer_time_trace(),
            #             saddle_log.get_last_layer_current_bounds_means_trace(first_half_only_as_ub=True),
            #             None, "Time [s]", "Upper Bound", title, errorbars=False,
            #             labelname=f"{algorithm_name_dict[algo]} iters: {outer_iters} - current bounds",
            #             dotted="-", xlog=False, ylog=False, color=colors[color_id])
            # color_id += 1
            custom_plot(fig_idx, saddle_log.get_last_layer_time_trace(),
                        saddle_log.get_last_layer_bounds_means_trace(first_half_only_as_ub=True),
                        None, "Time [s]", "Upper Bound", title, errorbars=False,
                        labelname=f"{algorithm_name_dict[algo]} iters: {outer_iters}, M: {M_factor}, init: {init}",
                        dotted="-", xlog=False, ylog=False, color=colors[color_id])
            color_id += 1
        fig_idx += 1


if __name__ == "__main__":

    cifar_incomplete_comparison()

    plt.show()
