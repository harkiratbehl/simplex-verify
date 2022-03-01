import argparse
import torch
from tools.cifar_bound_comparison import load_network, make_elided_models, cifar_loaders
from plnn.proxlp_solver.solver import SaddleLP
from plnn.explp_solver.solver import ExpLP
from plnn.proxlp_solver.dj_relaxation import DJRelaxationLP
from plnn.anderson_linear_approximation import AndersonLinearizedNetwork
import copy
import time


def runner():
    parser = argparse.ArgumentParser(description="Compute a bound and plot the results")
    parser.add_argument('network_filename', type=str,
                        help='Path ot the network')
    parser.add_argument('eps', type=float, help='Epsilon')
    parser.add_argument('--img_idx', type=int, default=0)
    parser.add_argument('--eta', type=float)
    parser.add_argument('--feta', type=float)
    parser.add_argument('--init_step', type=float)
    parser.add_argument('--fin_step', type=float)
    parser.add_argument('--M_factor', type=float, default=1.1)
    parser.add_argument('--nb_primal_anderson', type=float, default=100)
    parser.add_argument('--anderson_step_type', type=str, choices=["fw", "grad"])
    parser.add_argument('--out_iters', type=int)
    parser.add_argument('--inner_iters', type=int)
    parser.add_argument('--fw_cut_iters', type=int)
    parser.add_argument('--init_out_iters', type=int, default=100)
    parser.add_argument('--cut_frequency', type=int, default=450)
    parser.add_argument('--max_cuts', type=int, default=6)
    parser.add_argument('--cut_add', type=int, default=2)#nubmer of cuts added at frequency
    parser.add_argument('--init_dual_type', type=str, choices=["bigm", "cuts"], help="Dual algorithm for Anderson init.",
                        default="bigm")
    parser.add_argument('--init_inner_iters', type=int)
    parser.add_argument('--prox_momentum', type=float, default=0)
    parser.add_argument('--n_cuts', type=float, default=1e3)
    parser.add_argument('--fw_start', type=float, default=10)
    parser.add_argument('--fw_blockwise', action='store_true')
    parser.add_argument('--replicate_factor', type=int, default=1)
    parser.add_argument('--define_linear_approximation', action='store_true', help="if this flag is true, compute all intermediate bounds w/ the selected algorithm")
    parser.add_argument('--algorithm', type=str, choices=["saddle-explp", "prox-explp", "cut", "explp", "bigm-prox", "bigm-adam", "proxlp", "anderson-gurobi"],
                        help="which algorithm to use, in case one does init or uses it alone")

    args = parser.parse_args()

    # Load all the required data, setup the model
    model = load_network(args.network_filename)
    elided_models = make_elided_models(model)
    _, test_loader = cifar_loaders(1)
    for idx, (X, y) in enumerate(test_loader):
        if idx != args.img_idx:
            continue
        elided_model = elided_models[y.item()]
    domain = torch.stack([X.squeeze(0) - args.eps, X.squeeze(0) + args.eps], dim=-1)
    domain = domain.unsqueeze(0).expand(args.replicate_factor, *domain.shape).clone()

    lin_approx_string = "" if not args.define_linear_approximation else "-allbounds"

    # compute intermediate bounds with KW. Use only these for every method to allow comparison on the last layer
    # and optimize only the last layer
    cuda_elided_model = copy.deepcopy(elided_model).cuda()
    cuda_domain = domain.cuda()
    intermediate_net = SaddleLP([lay for lay in cuda_elided_model])
    with torch.no_grad():
        intermediate_net.set_solution_optimizer('best_naive_kw', None)
        intermediate_net.define_linear_approximation(cuda_domain, no_conv=False)
    intermediate_ubs = intermediate_net.upper_bounds
    intermediate_lbs = intermediate_net.lower_bounds

    folder = "./timings_cifar_anderson/"

    if args.algorithm == "cut":
        # Big-M Prox
        explp_params = {
            "initial_eta": args.eta,  # 1e-2 seems the way to go on CIFAR
            'final_eta': args.feta if args.feta else None,  # increasing seems potentially helpful (not more than 1e-1)
            "nb_inner_iter": args.inner_iters,  # 5?
            "nb_iter": args.out_iters,
            'bigm': "init",
            'cut': "only",
            "bigm_algorithm": "adam",# either prox or adam
            'cut_frequency': args.cut_frequency,
            'max_cuts': args.max_cuts,
            'cut_add': args.cut_add,
            'betas': (0.9, 0.999),
            'initial_step_size': args.init_step,
            'final_step_size': args.fin_step,
            "init_params": {
                "nb_outer_iter": args.init_out_iters,
                'initial_step_size': 1e-2,
                'final_step_size': 1e-4,
                'betas': (0.9, 0.999)
            },
        }
        cuda_elided_model = copy.deepcopy(elided_model).cuda()
        cuda_domain = domain.cuda()
        exp_net = ExpLP([lay for lay in cuda_elided_model], params=explp_params, use_preactivation=True,
                        store_bounds_progress=len(intermediate_net.weights))
        exp_start = time.time()
        with torch.no_grad():
            if not args.define_linear_approximation:
                exp_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                _, ub = exp_net.compute_lower_bound()
            else:
                exp_net.define_linear_approximation(cuda_domain)
                ub = exp_net.upper_bounds[-1]
        exp_end = time.time()
        exp_time = exp_end - exp_start
        exp_ubs = ub.cpu().mean()
        print(f"ExpLP Time: {exp_time}")
        print(f"ExpLP UB: {exp_ubs}")
        with open("tunings.txt", "a") as file:
            file.write(folder + f"{args.algorithm},UB:{exp_ubs},Time:{exp_time},Eta{args.eta},"
                         f"In-iters:{args.inner_iters},Out-iters:{args.out_iters}"
                         f"init:{args.init_out_iters}{args.init_step}{args.fin_step}\n")
        pickle_name = folder + f"timings-img{args.img_idx}-{args.algorithm},outer:{args.out_iters}," \
                               f"eta:{args.eta}-feta:{args.feta}," \
                               f"cut:{args.cut_frequency},{args.max_cuts},{args.cut_add}" \
                               f"init:{args.init_out_iters}{args.init_step}{args.fin_step},{lin_approx_string}.pickle"
        torch.save(exp_net.logger, pickle_name)

    if args.algorithm == "prox-explp":
        explp_params = {
            "anderson_algorithm": "prox",
            "initial_eta": args.eta,  # 1e-2 seems the way to go on CIFAR
            'final_eta': args.feta if args.feta else None,  # increasing seems potentially helpful (not more than 1e-1)
            "nb_inner_iter": args.inner_iters,  # 5?
            "nb_outer_iter": args.out_iters,
            "bigm": "init",
            "bigm_algorithm": "adam",  # either prox or adam
            "init_params": {
                "nb_outer_iter": args.init_out_iters,
                'initial_step_size': 1e-2,
                'final_step_size': 1e-4,
                'betas': (0.9, 0.999),
                'M_factor': args.M_factor
            },
            "primal_init_params": {
                'nb_bigm_iter': 900,
                'nb_anderson_iter': args.nb_primal_anderson,  # 100
                'initial_step_size': 1e-2,  # 1e-2,
                'final_step_size': 1e-5,  # 1e-3,
                'betas': (0.9, 0.999)
            }
        }
        cuda_elided_model = copy.deepcopy(elided_model).cuda()
        cuda_domain = domain.cuda()
        exp_net = ExpLP(
            [lay for lay in cuda_elided_model], params=explp_params, use_preactivation=True,
            store_bounds_progress=len(intermediate_net.weights))
        exp_start = time.time()
        with torch.no_grad():
            if not args.define_linear_approximation:
                exp_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                _, ub = exp_net.compute_lower_bound()
            else:
                exp_net.define_linear_approximation(cuda_domain)
                ub = exp_net.upper_bounds[-1]
        exp_end = time.time()
        exp_time = exp_end - exp_start
        exp_ubs = ub.cpu().mean()
        print(f"Prox-ExpLP Time: {exp_time}")
        print(f"Prox-ExpLP UB: {exp_ubs}")
        with open("tunings.txt", "a") as file:
            file.write(
                folder + f"{args.algorithm},UB:{exp_ubs},Time:{exp_time},Eta{args.eta},"
                         f"In-iters:{args.inner_iters},Out-iters:{args.out_iters}"
                         f"init:{args.init_out_iters}{args.init_step}{args.fin_step}\n")
        pickle_name = folder + f"timings-img{args.img_idx}-{args.algorithm},inner:{args.inner_iters}," \
                               f"eta:{args.eta}-feta:{args.feta}," \
                               f"init:{args.init_out_iters}{args.init_step}{args.fin_step},{lin_approx_string}.pickle"
        torch.save(exp_net.logger, pickle_name)

    if args.algorithm == "saddle-explp":

        if args.anderson_step_type == "grad":
            step_size_dict = {
                'type': 'grad',
                'initial_step_size': args.init_step,  # 1e-2,  # 5e-3, constant, also performs quite well
                'final_step_size': args.fin_step,  # 1e-3
            }
        else:
            step_size_dict = {
                'type': 'fw',
                'fw_start': args.fw_start  # 10 is better
            }

        explp_params = {
            "anderson_algorithm": "saddle",
            "nb_iter": args.out_iters,
            'blockwise': args.fw_blockwise,  # False is better
            "step_size_dict": step_size_dict,
            "init_params": {
                "nb_outer_iter": args.init_out_iters,  # 500 is fine
                'initial_step_size': 1e-2,
                'final_step_size': 1e-4,
                'betas': (0.9, 0.999),
                'M_factor': args.M_factor  # constant factor of the big-m variables to use for M -- 1.1 works very well
            },
            "primal_init_params": {
                'nb_bigm_iter': 100,
                'nb_anderson_iter': args.nb_primal_anderson,  # 0 for bigm init, at least 1000 for cuts init
                'initial_step_size': 1e-2,  # 1e-2,
                'final_step_size': 1e-5,  # 1e-3,
                'betas': (0.9, 0.999)
            }
        }
        if args.init_dual_type == "bigm":
            explp_params.update({"bigm": "init", "bigm_algorithm": "adam"})
        else:
            # args.init_dual_type == "cuts"
            cut_init_params = {
                'cut_frequency': args.cut_frequency,
                'max_cuts': args.max_cuts,
                'cut_add': args.cut_add,
                'nb_iter': args.fw_cut_iters,
                'initial_step_size': 1e-3,
                'final_step_size': 1e-6,
            }
            explp_params.update({"cut": "init", "cut_init_params": cut_init_params})

        cuda_elided_model = copy.deepcopy(elided_model).cuda()
        cuda_domain = domain.cuda()
        exp_net = ExpLP(
            [lay for lay in cuda_elided_model], params=explp_params, use_preactivation=True,
            store_bounds_progress=len(intermediate_net.weights), fixed_M=True)  # fixed_M=(args.M_factor == 1.0)
        exp_start = time.time()
        with torch.no_grad():
            if not args.define_linear_approximation:
                exp_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                _, ub = exp_net.compute_lower_bound()
            else:
                exp_net.define_linear_approximation(cuda_domain)
                ub = exp_net.upper_bounds[-1]
        exp_end = time.time()
        exp_time = exp_end - exp_start
        exp_ubs = ub.cpu().mean()
        print(f"Saddle-ExpLP Time: {exp_time}")
        print(f"Saddle-ExpLP UB: {exp_ubs}")

        step_str = args.anderson_step_type + f",istepsize:{args.init_step},fstepsize:{args.fin_step}" \
            if args.anderson_step_type == "grad" else args.anderson_step_type
        with open("tunings.txt", "a") as file:
            file.write(
                folder + f"{args.algorithm},UB:{exp_ubs},Time:{exp_time},Out-iters:{args.out_iters},step:{step_str},"
                         f"\n")
        pickle_name = folder + f"timings-img{args.img_idx}-{args.algorithm},outer:{args.out_iters}," \
                               f"step:{step_str},M:{args.M_factor}" \
                               f",init:{args.init_dual_type},{lin_approx_string}.pickle"
        torch.save(exp_net.logger, pickle_name)

    elif args.algorithm == "bigm-prox":
        # Big-M Prox
        bigm_prox_params = {
            "initial_eta": args.eta if args.feta else None,
            'final_eta': args.feta if args.feta else None,
            "nb_inner_iter": int(args.inner_iters),  # 5?
            "nb_outer_iter": args.out_iters,
            "bigm_algorithm": "prox",
            "bigm": "only",
        }
        cuda_elided_model = copy.deepcopy(elided_model).cuda()
        cuda_domain = domain.cuda()
        bigmprox_net = ExpLP([lay for lay in cuda_elided_model], params=bigm_prox_params,
                             store_bounds_progress=len(intermediate_net.weights))
        bigmprox_start = time.time()
        with torch.no_grad():
            if not args.define_linear_approximation:
                bigmprox_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                _, ub = bigmprox_net.compute_lower_bound()
            else:
                bigmprox_net.define_linear_approximation(cuda_domain)
                ub = bigmprox_net.upper_bounds[-1]
        bigmprox_end = time.time()
        bigmprox_time = bigmprox_end - bigmprox_start
        bigmprox_ubs = ub.cpu().mean()
        print(f"ExpLP Time: {bigmprox_time}")
        print(f"ExpLP UB: {bigmprox_ubs}")
        with open("tunings.txt", "a") as file:
            file.write(folder + f"{args.algorithm},UB:{bigmprox_ubs},Time:{bigmprox_time},Eta{args.eta},"
                                f"In-iters:{args.inner_iters},Out-iters:{args.out_iters}\n")
        pickle_name = folder + f"timings-img{args.img_idx}-{args.algorithm},inner:{args.inner_iters}," \
                               f"eta:{args.eta}-feta:{args.feta}{lin_approx_string}.pickle"
        torch.save(bigmprox_net.logger, pickle_name)

    elif args.algorithm == "bigm-adam":
        bigm_adam_params = {
            "bigm_algorithm": "adam",
            "bigm": "only",
            "nb_outer_iter": args.out_iters,
            'initial_step_size': args.init_step,
            'final_step_size': args.fin_step,
            'betas': (0.9, 0.999)
        }
        cuda_elided_model = copy.deepcopy(elided_model).cuda()
        cuda_domain = domain.cuda()
        bigmadam_net = ExpLP([lay for lay in cuda_elided_model], params=bigm_adam_params,
                             store_bounds_progress=len(intermediate_net.weights))
        bigmadam_start = time.time()
        with torch.no_grad():
            if not args.define_linear_approximation:
                bigmadam_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                _, ub = bigmadam_net.compute_lower_bound()
            else:
                bigmadam_net.define_linear_approximation(cuda_domain)
                ub = bigmadam_net.upper_bounds[-1]
        bigmadam_end = time.time()
        bigmadam_time = bigmadam_end - bigmadam_start
        bigmadam_ubs = ub.cpu().mean()
        print(f"BigM adam Time: {bigmadam_time}")
        print(f"BigM adam UB: {bigmadam_ubs}")
        with open("tunings.txt", "a") as file:
            file.write(folder + f"{args.algorithm},UB:{bigmadam_ubs},Time:{bigmadam_time},Out-iters:{args.out_iters}\n")
        pickle_name = folder + f"timings-img{args.img_idx}-{args.algorithm},outer:{args.out_iters},istepsize:{args.init_step},fstepsize:{args.fin_step}{lin_approx_string}.pickle"
        torch.save(bigmadam_net.logger, pickle_name)

    elif args.algorithm == "proxlp":
        # ProxLP
        acceleration_dict = {
            'momentum': args.prox_momentum,  # decent momentum: 0.6 w/ increasing eta
            'nesterov': 0,
            'adam': False
        }

        optprox_params = {
            'nb_total_steps': args.out_iters,
            'max_nb_inner_steps': 2,  # this is 2/5 as simpleprox
            'eta': args.eta,  # eta is kept the same as in simpleprox
            'initial_eta': args.eta if args.feta else None,
            'final_eta': args.feta if args.feta else None,
            'log_values': False,
            'inner_cutoff': 0,
            'maintain_primal': True,
            'acceleration_dict': acceleration_dict
        }
        cuda_elided_model = copy.deepcopy(elided_model).cuda()
        cuda_domain = domain.cuda()
        optprox_net = SaddleLP([lay for lay in cuda_elided_model], store_bounds_progress=len(intermediate_net.weights))
        optprox_start = time.time()
        with torch.no_grad():
            optprox_net.set_decomposition('pairs', 'naive')
            optprox_net.set_solution_optimizer('optimized_prox', optprox_params)
            if not args.define_linear_approximation:
                optprox_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                _, ub = optprox_net.compute_lower_bound()
            else:
                optprox_net.define_linear_approximation(cuda_domain)
                ub = optprox_net.upper_bounds[-1]
        optprox_end = time.time()
        optprox_time = optprox_end - optprox_start
        optprox_ubs = ub.cpu().mean()
        print(f"ProxLP Time: {optprox_time}")
        print(f"ProxLP UB: {optprox_ubs}")
        with open("tunings.txt", "a") as file:
            file.write(folder + f"{args.algorithm},UB:{optprox_ubs},Time:{optprox_time},Eta{args.eta},Out-iters:{args.out_iters}\n")

        acceleration_string = ""
        if not acceleration_dict['adam']:
            acceleration_string += f"-mom:{args.prox_momentum}"

        pickle_name = folder + f"timings-img{args.img_idx}-{args.algorithm},eta:{args.eta}-feta:{args.feta}{acceleration_string}{lin_approx_string}.pickle"
        torch.save(optprox_net.logger, pickle_name)

    elif args.algorithm == "anderson-gurobi":
        # Gurobi Anderson baseline
        lp_and_grb_net = AndersonLinearizedNetwork(
            [lay for lay in elided_model], mode="lp-cut", n_cuts=int(args.n_cuts), use_preactivation=True,
            store_bounds_progress=len(intermediate_net.weights), cuts_per_neuron=True)
        lp_and_grb_start = time.time()
        if not args.define_linear_approximation:
            lp_and_grb_net.build_model_using_bounds(domain, ([lbs.cpu() for lbs in intermediate_lbs],
                                                             [ubs.cpu() for ubs in intermediate_ubs]))
            _, ub = lp_and_grb_net.compute_lower_bound(ub_only=True)
        else:
            lp_and_grb_net.define_linear_approximation(domain, n_threads=4)
            ub = lp_and_grb_net.upper_bounds[-1]

        lp_and_grb_end = time.time()
        lp_and_grb_time = lp_and_grb_end - lp_and_grb_start
        lp_and_grb_ubs = torch.Tensor(ub).cpu().mean()
        print(f"Anderson Gurobi Time: {lp_and_grb_time}")
        print(f"Anderson Gurobi UB: {lp_and_grb_ubs}")
        with open("tunings.txt", "a") as file:
            file.write(f"{args.algorithm},n_cuts:{args.n_cuts},UB:{lp_and_grb_ubs},Time:{lp_and_grb_time}\n")

        pickle_name = folder + f"timings-img{args.img_idx}-{args.algorithm},n_cuts:{args.n_cuts}{lin_approx_string}.pickle"
        torch.save(lp_and_grb_net.logger, pickle_name)


if __name__ == '__main__':
    runner()
