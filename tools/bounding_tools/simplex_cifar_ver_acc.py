import argparse
import os
import torch
import time
import copy
import sys
from plnn.network_linear_approximation import LinearizedNetwork
from plnn.anderson_linear_approximation import AndersonLinearizedNetwork
from tools.cifar_bound_comparison import load_network, make_elided_models, cifar_loaders, dump_bounds, read_bounds
from tools.bab_tools.model_utils import load_cifar_l1_network

from plnn.simplex_solver.solver import SimplexLP
from plnn.simplex_solver.baseline_solver import Baseline_SimplexLP
from plnn.simplex_solver import utils
from plnn.simplex_solver.baseline_gurobi_linear_approximation import Baseline_LinearizedNetwork
from plnn.simplex_solver.gurobi_linear_approximation import Simp_LinearizedNetwork
from plnn.simplex_solver.disjunctive_gurobi import DP_LinearizedNetwork

import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Compute and time a bunch of bounds.")
    parser.add_argument('--network_filename', type=str,
                        help='Path to the network')
    parser.add_argument('--eps', type=float,
                        help='Epsilon - default: 0.5')
    parser.add_argument('--target_directory', type=str,
                        help='Where to store the results')
    parser.add_argument('--modulo', type=int,
                        help='Numbers of a job to split the dataset over.')
    parser.add_argument('--modulo_do', type=int,
                        help='Which job_id is this one.')
    parser.add_argument('--from_intermediate_bounds', action='store_true',
                        help="if this flag is true, intermediate bounds are computed w/ best of naive-KW")
    parser.add_argument('--nn_name', type=str, help='network architecture name')
    args = parser.parse_args()

    np.random.seed(0)

    mnist = 0
    if mnist:
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_loader = datasets.MNIST("./mnistdata/", train=False, download=True, transform =transform)
    else:
        _, test_loader = cifar_loaders(1)

    if args.nn_name:
        model = load_cifar_l1_network(args.nn_name)
    else:
        model = load_network(args.network_filename)

    elided_models = make_elided_models(model, True)
    # elided_models = make_elided_models(model)


    results_dir = args.target_directory
    os.makedirs(results_dir, exist_ok=True)


    crown_correct_sum = 0
    planet_correct_sum = 0
    dp_correct_sum = 0
    total_images = 0
    pgd_correct_sum = 0
    gur_planet_correct_sum = 0
    gur_dp_correct_sum = 0
    basline_bigm_adam_correct_sum = 0
    basline_cut_correct_sum = 0

    crown_time_sum = 0
    planet_time_sum = 0
    dp_time_sum = 0
    gur_planet_time_sum = 0
    gur_dp_time_sum = 0
    basline_bigm_adam_time_sum = 0
    basline_cut_time_sum = 0

    ft = open(os.path.join(results_dir, "ver_acc_new.txt"), "a")

    for idx, (X, y) in enumerate(test_loader):
        if (args.modulo is not None) and (idx % args.modulo != args.modulo_do):
            continue

        if idx>=1000:
            sys.exit()

        target_dir = os.path.join(results_dir, f"{idx}")
        os.makedirs(target_dir, exist_ok=True)


        # ### predicting
        out = model(X)
        pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()
        print(idx, y.item(), pred[0])
        if y.item()!=pred[0]:
            # print("Incorrect prediction")
            continue

        total_images +=1

        lin_approx_string = "" if not args.from_intermediate_bounds else "-fromintermediate"
    
        # # ## baseline-lirpa Bounds
        lirpa_target_file = os.path.join(target_dir, f"baseline-lirpa6{lin_approx_string}-fixed.txt")
        if os.path.exists(lirpa_target_file):

            time, ub = read_bounds(lirpa_target_file)
            ###########################
            ## verified accuracy
            correct=1
            for bn in ub:
                if bn >0:
                    correct=0
                    break
            planet_correct_sum += correct
            planet_time_sum += float(time)
            ###########################

        # # # ## baseline-crown Bounds
        # lirpa_target_file = os.path.join(target_dir, f"baseline-lirpa-crown{lin_approx_string}-fixed.txt")
        # if os.path.exists(lirpa_target_file):
            
        #     time, ub = read_bounds(lirpa_target_file)
        #     ###########################
        #     ## verified accuracy
        #     correct=1
        #     for bn in ub:
        #         if bn >0:
        #             correct=0
        #             break
        #     crown_correct_sum += correct
        #     crown_time_sum += float(time)
        #     ###########################


        # Gurobi baseline-planet-simplex Bounds
        grb_target_file = os.path.join(target_dir, f"gurobi-baseline-planet-simplex{lin_approx_string}-fixed.txt")
        if os.path.exists(grb_target_file):
            

            time, ub = read_bounds(grb_target_file)
            ###########################
            ## verified accuracy
            correct=1
            for bn in ub:
                if bn >0:
                    correct=0
                    break
            gur_planet_correct_sum += correct
            gur_planet_time_sum += float(time)
            ###########################

        # baseline-bigm-adam-simplex
        bigm_target_file = os.path.join(target_dir, f"baseline-bigm-adam-simplex_850{lin_approx_string}.txt")
        if os.path.exists(bigm_target_file):
            

            time, ub = read_bounds(bigm_target_file)
            ###########################
            ## verified accuracy
            correct=1
            for bn in ub:
                if bn >0:
                    correct=0
                    break
            basline_bigm_adam_correct_sum += correct
            basline_bigm_adam_time_sum += float(time)
            ###########################

        # baseline-cut-simplex
        bigm_target_file = os.path.join(target_dir, f"baseline-cut-simplex_{lin_approx_string}.txt")
        if os.path.exists(bigm_target_file):
            

            time, ub = read_bounds(bigm_target_file)
            ###########################
            ## verified accuracy
            correct=1
            for bn in ub:
                if bn >0:
                    correct=0
                    break
            basline_cut_correct_sum += correct
            basline_cut_time_sum += float(time)
            ###########################


        #######################################
        ### FOR SIMPLEX CONDITIONED METHODS ###
        #######################################

        # # ## auto-lirpa-dp Bounds
        lirpa_target_file = os.path.join(target_dir, f"auto-lirpa-dp-3{lin_approx_string}-fixed.txt")
        if os.path.exists(lirpa_target_file):
            time, ub = read_bounds(lirpa_target_file)
            ###########################
            ## verified accuracy
            correct=1
            for bn in ub:
                if bn >0:
                    correct=0
                    break
            dp_correct_sum += correct
            dp_time_sum += float(time)
            ###########################
            

        ## Gurobi dp-simplex Bounds
        grb_target_file = os.path.join(target_dir, f"gurobi-dp-simplex{lin_approx_string}-fixed.txt")
        if os.path.exists(grb_target_file):
            time, ub = read_bounds(grb_target_file)
            ###########################
            ## verified accuracy
            correct=1
            for bn in ub:
                if bn >0:
                    correct=0
                    break
            gur_dp_correct_sum += correct
            gur_dp_time_sum += float(time)
            ###########################

        print('Tot images: ', total_images, 'Crown, Planet, dp acc: ', crown_correct_sum/float(total_images), planet_correct_sum/float(total_images), dp_correct_sum/float(total_images))

        print('Tot imges: ', total_images, 'Gurobi Planet, dp acc: ', gur_planet_correct_sum/float(total_images), gur_dp_correct_sum/float(total_images))

        ft.write(str(crown_time_sum/float(total_images)))
        ft.write(",")
        ft.write(str(planet_time_sum/float(total_images)))
        ft.write(",")
        ft.write(str(dp_time_sum/float(total_images)))
        ft.write(",")
        ft.write(str(gur_planet_time_sum/float(total_images)))
        ft.write(",")
        ft.write(str(gur_dp_time_sum/float(total_images)))
        ft.write(",")
        ft.write(str(basline_bigm_adam_time_sum/float(total_images)))
        ft.write(",")
        ft.write(str(basline_cut_time_sum/float(total_images)))
        ft.write(",")
        ft.write(str(gur_planet_correct_sum/float(total_images)))
        ft.write(",")
        ft.write(str(gur_dp_correct_sum/float(total_images)))
        ft.write(",")
        ft.write(str(crown_correct_sum/float(total_images)))
        ft.write(",")
        ft.write(str(planet_correct_sum/float(total_images)))
        ft.write(",")
        ft.write(str(dp_correct_sum/float(total_images)))
        ft.write(",")
        ft.write(str(basline_bigm_adam_correct_sum/float(total_images)))
        ft.write(",")
        ft.write(str(basline_cut_correct_sum/float(total_images)))
        ft.write(",")
        ft.write(str(total_images))
        ft.write(",")
        ft.write(str(idx))
        ft.write("\n")

        ######################################################
        ################ DUAL BOUNDS ###################
        ######################################################

        # baseline-bigm-adam-simplex. 
        # TO-DO (iters need to betuned)
        # for bigm_steps in [500, 1000]:
            # bigm_adam_params = {
            #     "bigm_algorithm": "adam",
            #     "bigm": "only",
            #     "nb_outer_iter": bigm_steps,
            #     'initial_step_size': 1e-2,
            #     'final_step_size': 1e-4,
            #     'betas': (0.9, 0.999)
            # }
            # bigm_target_file = os.path.join(target_dir, f"baseline-bigm-adam-simplex_{bigm_steps}{lin_approx_string}.txt")
            # if not os.path.exists(bigm_target_file):
            #     cuda_elided_model = copy.deepcopy(elided_model).cuda()
            #     bigm_net = Baseline_SimplexLP(cuda_elided_model, params=bigm_adam_params)
            #     bigm_start = time.time()
            #     with torch.no_grad():
            #         bigm_net.optimize = bigm_net.bigm_subgradient_optimizer
            #         bigm_net.logger = utils.OptimizationTrace()
            #         if not args.from_intermediate_bounds:
            #             bigm_net.define_linear_approximation(cuda_domain)
            #             ub = bigm_net.upper_bounds[-1]
            #         else:
            #             print('using build bounds')
            #             bigm_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
            #             _, ub = bigm_net.compute_lower_bound()
            #     bigm_end = time.time()
            #     bigm_time = bigm_end - bigm_start
            #     bigm_ubs = ub.cpu()

            #     del bigm_net
            #     dump_bounds(bigm_target_file, bigm_time, bigm_ubs)

        
        # del intermediate_net
        # intermediate_net = SimplexLP([lay for lay in cuda_elided_model])
        # cuda_domain = (X.cuda(), args.eps)
        # domain = (X, args.eps)
        # with torch.no_grad():
        #     intermediate_net.set_solution_optimizer('best_naive_kw', None)
        #     intermediate_net.define_linear_approximation(cuda_domain, no_conv=False,
        #                                                  override_numerical_errors=True)
        # intermediate_ubs = intermediate_net.upper_bounds
        # intermediate_lbs = intermediate_net.lower_bounds


        # ## Gurobi planet-simplex Bounds
        # grb_target_file = os.path.join(target_dir, f"gurobi-planet-simplex{lin_approx_string}-fixed.txt")
        # if not os.path.exists(grb_target_file):
        #     grb_net = Simp_LinearizedNetwork([lay for lay in elided_model])
        #     grb_start = time.time()
        #     if not args.from_intermediate_bounds:
        #         grb_net.define_linear_approximation(domain, n_threads=4)
        #         ub = grb_net.upper_bounds[-1]
        #     else:
        #         grb_net.build_model_using_bounds(domain, ([lbs.cpu().squeeze(0) for lbs in intermediate_lbs], [ubs.cpu().squeeze(0) for ubs in intermediate_ubs]), intermediate_net.init_cut_coeffs, n_threads=4)
        #         _, ub = grb_net.compute_lower_bound(ub_only=True)
        #     grb_end = time.time()
        #     grb_time = grb_end - grb_start
        #     grb_ubs = torch.Tensor(ub).cpu()
        #     dump_bounds(grb_target_file, grb_time, grb_ubs)
        #     del grb_net

        ## Gurobi planet-simplex-false Bounds
        # grb_target_file = os.path.join(target_dir, f"gurobi-planet-simplex-false{lin_approx_string}-fixed.txt")
        # if not os.path.exists(grb_target_file):
            # grb_net = Simp_LinearizedNetwork([lay for lay in elided_model])
            # grb_start = time.time()
            # if not args.from_intermediate_bounds:
            #     grb_net.define_linear_approximation(domain, n_threads=4)
            #     ub = grb_net.upper_bounds[-1]
            # else:
            #     grb_net.build_model_using_bounds(domain, ([lbs.cpu().squeeze(0) for lbs in intermediate_lbs], [ubs.cpu().squeeze(0) for ubs in intermediate_ubs]), intermediate_net.init_cut_coeffs, n_threads=4, simplex_constraint=False)
            #     _, ub = grb_net.compute_lower_bound(ub_only=True)
            # grb_end = time.time()
            # grb_time = grb_end - grb_start
            # grb_ubs = torch.Tensor(ub).cpu()
            # dump_bounds(grb_target_file, grb_time, grb_ubs)
            # del grb_net

        ## Gurobi dp-simplex Bounds
        # grb_target_file = os.path.join(target_dir, f"gurobi-dp-simplex{lin_approx_string}-fixed.txt")
        # if not os.path.exists(grb_target_file):
        #     grb_net = DP_LinearizedNetwork([lay for lay in elided_model])
        #     grb_start = time.time()
        #     if not args.from_intermediate_bounds:
        #         grb_net.define_linear_approximation(domain, n_threads=4)
        #         ub = grb_net.upper_bounds[-1]
        #     else:
        #         grb_net.build_model_using_bounds(domain, ([lbs.cpu().squeeze(0) for lbs in intermediate_lbs], [ubs.cpu().squeeze(0) for ubs in intermediate_ubs]), intermediate_net.init_cut_coeffs, n_threads=4)
        #         _, ub = grb_net.compute_lower_bound(ub_only=True)
        #     grb_end = time.time()
        #     grb_time = grb_end - grb_start
        #     grb_ubs = torch.Tensor(ub).cpu()
        #     dump_bounds(grb_target_file, grb_time, grb_ubs)
        #     del grb_net


        # bigm-adam-simplex. 
        # TO-DO (iters need to betuned)
        # for bigm_steps in [500, 1000]:
            # bigm_adam_params = {
            #     "bigm_algorithm": "adam",
            #     "bigm": "only",
            #     "nb_outer_iter": bigm_steps,
            #     'initial_step_size': 1e-4,
            #     'final_step_size': 1e-4,
            #     'betas': (0.9, 0.999)
            # }
            # bigm_target_file = os.path.join(target_dir, f"bigm-adam-simplex_{bigm_steps}{lin_approx_string}.txt")
            # if not os.path.exists(bigm_target_file):
            #     cuda_elided_model = copy.deepcopy(elided_model).cuda()
            #     bigm_net = SimplexLP(cuda_elided_model, params=bigm_adam_params)
            #     bigm_start = time.time()
            #     with torch.no_grad():
            #         bigm_net.set_solution_optimizer('bigm_subgradient_optimizer', None)
            #         if not args.from_intermediate_bounds:
            #             bigm_net.define_linear_approximation(cuda_domain)
            #             ub = bigm_net.upper_bounds[-1]
            #         else:
            #            bigm_net.build_model_using_intermediate_net(cuda_domain, (intermediate_lbs, intermediate_ubs), intermediate_net)
            #            _, ub = bigm_net.compute_lower_bound()
            #     bigm_end = time.time()
            #     bigm_time = bigm_end - bigm_start
            #     bigm_ubs = ub.cpu()

            #     del bigm_net
            #     dump_bounds(bigm_target_file, bigm_time, bigm_ubs)


if __name__ == '__main__':
    main()
