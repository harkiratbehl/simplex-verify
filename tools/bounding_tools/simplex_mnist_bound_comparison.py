import argparse
import os
import torch
import time
import copy
import sys
from plnn.network_linear_approximation import LinearizedNetwork
from plnn.anderson_linear_approximation import AndersonLinearizedNetwork
from tools.cifar_bound_comparison import load_network, make_elided_models, cifar_loaders, dump_bounds
from tools.bab_tools.model_utils import load_cifar_l1_network, mnist_model_wide_l1, mnist_model

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
    args = parser.parse_args()


    np.random.seed(0)


    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    transform=transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_test = datasets.MNIST("./mnistdata/", train=False, download=True, transform =transform)

    model_name = './data/advertorch/mnist_model_wide0.3_adv.pt'
    sudo_model = mnist_model_wide_l1()
    model = mnist_model()
    sudo_model.load_state_dict(torch.load(model_name, map_location = "cpu"))
    model_dict = dict(model.named_parameters())
    for (name1, param1), (name2, param2) in zip(sudo_model.state_dict().items(), model_dict.items()):
        model_dict[name2].data.copy_(param1.data)


    results_dir = args.target_directory
    os.makedirs(results_dir, exist_ok=True)

    elided_models = make_elided_models(model, True)
    # elided_models = make_elided_models(model)

    crown_correct_sum = 0
    planet_correct_sum = 0
    dp_correct_sum = 0
    total_images = 0
    pgd_correct_sum = 0
    gur_planet_correct_sum = 0
    gur_dp_correct_sum = 0

    basline_bigm_adam_correct_sum = 0
    basline_cut_correct_sum = 0
    basline_bigm_adam_time_sum = 0
    basline_cut_time_sum = 0

    ft = open(os.path.join(results_dir, "ver_acc.txt"), "a")

    for idx, (X, y) in enumerate(mnist_test):
        X = X.unsqueeze(0)
        if (args.modulo is not None) and (idx % args.modulo != args.modulo_do):
            continue

        target_dir = os.path.join(results_dir, f"{idx}")
        os.makedirs(target_dir, exist_ok=True)


        ### predicting
        out = model(X)
        pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()
        print(idx, y, pred[0])

        if y!=pred[0]:
            # print("Incorrect prediction")
            continue

        total_images +=1


        elided_model = elided_models[y]
        to_ignore = y

        domain = torch.stack([X.squeeze(0) - args.eps,
                              X.squeeze(0) + args.eps], dim=-1).unsqueeze(0)

        lin_approx_string = "" if not args.from_intermediate_bounds else "-fromintermediate"

        # compute intermediate bounds with KW. Use only these for every method to allow comparison on the last layer
        #######################################
        ###### FOR BASELINE METHODS FIRST #####
        #######################################
        
        # LIRPA SIMPLEX
        # grb_target_file = os.path.join(target_dir, f"lirpa-baseline-planet-simplex{lin_approx_string}-fixed.txt")
        cuda_elided_model = copy.deepcopy(elided_model).cuda()
        intermediate_net = Baseline_SimplexLP([lay for lay in cuda_elided_model], max_batch=3000)
        cuda_domain = (X.cuda(), args.eps)
        domain = (X, args.eps)

        grb_start = time.time()

        with torch.no_grad():
            intermediate_net.set_solution_optimizer("best_naive_simplex", None)
            intermediate_net.define_linear_approximation(cuda_domain, no_conv=False)
        intermediate_ubs = intermediate_net.upper_bounds
        intermediate_lbs = intermediate_net.lower_bounds

        # ub = intermediate_ubs[-1]
        # grb_end = time.time()
        # grb_time = grb_end - grb_start
        # grb_ubs = torch.Tensor(ub.cpu())
        # dump_bounds(grb_target_file, grb_time, grb_ubs)


        # # ## baseline-lirpa Bounds
        # lirpa_target_file = os.path.join(target_dir, f"baseline-lirpa{lin_approx_string}-fixed.txt")
        # lirpa_l_target_file = os.path.join(target_dir, f"l_baseline-lirpa{lin_approx_string}-fixed.txt")
        # if not os.path.exists(lirpa_l_target_file):
        #     lirpa_params = {
        #         "nb_outer_iter": 6
        #     }
        #     lirpa_net = Baseline_SimplexLP(cuda_elided_model, params=lirpa_params,
        #                          store_bounds_progress=len(intermediate_net.weights), debug=True)
        #     lirpa_start = time.time()
        #     with torch.no_grad():
        #         lirpa_net.optimize = lirpa_net.auto_lirpa_optimizer
        #         lirpa_net.logger = utils.OptimizationTrace()
        #         lirpa_net.build_model_using_intermediate_net(cuda_domain, (intermediate_lbs, intermediate_ubs), intermediate_net)
        #         lb, ub = lirpa_net.compute_lower_bound()
        #     lirpa_end = time.time()
        #     lirpa_time = lirpa_end - lirpa_start
        #     lirpa_lbs = lb.detach().cpu()
        #     lirpa_ubs = ub.detach().cpu()
        #     dump_bounds(lirpa_target_file, lirpa_time, lirpa_ubs)
        #     dump_bounds(lirpa_l_target_file, lirpa_time, lirpa_lbs)

        #     ###########################
        #     ## verified accuracy
        #     correct=1
        #     for bn in ub.cpu()[0]:
        #         if bn >0:
        #             correct=0
        #             break
        #     planet_correct_sum += correct
        #     ###########################

        #     del lirpa_net

        # # ## baseline-crown Bounds
        # lirpa_target_file = os.path.join(target_dir, f"crown{lin_approx_string}-fixed.txt")
        # lirpa_l_target_file = os.path.join(target_dir, f"l_crown{lin_approx_string}-fixed.txt")
        # if not os.path.exists(lirpa_l_target_file):
        #     lirpa_params = {
        #         "nb_outer_iter": 1,
        #     }
        #     lirpa_net = Baseline_SimplexLP(cuda_elided_model, params=lirpa_params,
        #                          store_bounds_progress=len(intermediate_net.weights), debug=True)
        #     lirpa_start = time.time()
        #     with torch.no_grad():
        #         lirpa_net.optimize = lirpa_net.auto_lirpa_optimizer
        #         lirpa_net.logger = utils.OptimizationTrace()
        #         lirpa_net.build_model_using_intermediate_net(cuda_domain, (intermediate_lbs, intermediate_ubs), intermediate_net)
        #         lb, ub = lirpa_net.compute_lower_bound()
        #     lirpa_end = time.time()
        #     lirpa_time = lirpa_end - lirpa_start
        #     lirpa_lbs = lb.detach().cpu()
        #     lirpa_ubs = ub.detach().cpu()
        #     dump_bounds(lirpa_target_file, lirpa_time, lirpa_ubs)
        #     dump_bounds(lirpa_l_target_file, lirpa_time, lirpa_lbs)

        #     ###########################
        #     ## verified accuracy
        #     correct=1
        #     for bn in ub.cpu()[0]:
        #         if bn >0:
        #             correct=0
        #             break
        #     crown_correct_sum += correct
        #     ###########################

        #     del lirpa_net

        # ##########################
        # ### pgd bounds
        # pgd_target_file = os.path.join(target_dir, f"pgd{lin_approx_string}-fixed.txt")
        # l_pgd_target_file = os.path.join(target_dir, f"l_pgd{lin_approx_string}-fixed.txt")

        # pgd_bounds = intermediate_net.advertorch_pgd_upper_bound()
        # pgd_time = 0
        # pgd_lbs = pgd_bounds[:9]
        # pgd_ubs = pgd_bounds[9:]
        # dump_bounds(pgd_target_file, pgd_time, pgd_ubs)
        # dump_bounds(l_pgd_target_file, pgd_time, pgd_lbs)
        # ###########################

        # ###########################
        # ## pgd accuracy
        # correct=1
        # for bn in pgd_lbs:
        #     if bn >0:
        #         correct=0
        #         break
        # pgd_correct_sum += correct
        # ##########################

        # Gurobi baseline-planet-simplex Bounds
        # grb_target_file = os.path.join(target_dir, f"gurobi-baseline-planet-simplex{lin_approx_string}-fixed.txt")
        # # grb_l_target_file = os.path.join(target_dir, f"l_gurobi-baseline-planet-simplex{lin_approx_string}-fixed.txt")
        # if not os.path.exists(grb_target_file):
        #     grb_net = Baseline_LinearizedNetwork([lay.cpu() for lay in elided_model])
        #     grb_start = time.time()
        #     if not args.from_intermediate_bounds:
        #         grb_net.define_linear_approximation(domain, n_threads=4)
        #         ub = grb_net.upper_bounds[-1]
        #     else:
        #         grb_net.build_model_using_bounds(domain, ([lbs.cpu().squeeze(0) for lbs in intermediate_lbs], [ubs.cpu().squeeze(0) for ubs in intermediate_ubs]), n_threads=4)
        #         _ , ub = grb_net.compute_lower_bound(ub_only=True)
        #     grb_end = time.time()
        #     grb_time = grb_end - grb_start
        #     grb_ubs = torch.Tensor(ub).cpu()
        #     # grb_lbs = torch.Tensor(lb).cpu()
        #     dump_bounds(grb_target_file, grb_time, grb_ubs)
        #     # dump_bounds(grb_l_target_file, grb_time, grb_lbs)
        #     del grb_net


        #     ###########################
        #     ## verified accuracy
        #     correct=1
        #     for bn in ub.cpu()[0]:
        #         if bn >0:
        #             correct=0
        #             break
        #     gur_planet_correct_sum += correct
        #     ###########################


        ## baseline-bigm-adam-simplex. 
        ## TO-DO (iters need to betuned)
        for bigm_steps in [850]:
            bigm_adam_params = {
                "bigm_algorithm": "adam",
                "bigm": "only",
                "nb_outer_iter": bigm_steps,
                'initial_step_size': 1e-2,
                'final_step_size': 1e-4,
                'betas': (0.9, 0.999)
            }
            bigm_target_file = os.path.join(target_dir, f"baseline-bigm-adam-simplex_{bigm_steps}{lin_approx_string}.txt")
            # if not os.path.exists(bigm_target_file):
            cuda_elided_model = copy.deepcopy(elided_model).cuda()
            bigm_net = Baseline_SimplexLP(cuda_elided_model, params=bigm_adam_params)
            bigm_start = time.time()
            with torch.no_grad():
                bigm_net.optimize = bigm_net.bigm_subgradient_optimizer
                bigm_net.logger = utils.OptimizationTrace()
                if not args.from_intermediate_bounds:
                    bigm_net.define_linear_approximation(cuda_domain)
                    ub = bigm_net.upper_bounds[-1]
                else:
                    bigm_net.build_model_using_intermediate_net(cuda_domain, (intermediate_lbs, intermediate_ubs), intermediate_net)
                    _, ub = bigm_net.compute_lower_bound()
            bigm_end = time.time()
            bigm_time = bigm_end - bigm_start
            bigm_ubs = ub.cpu()

            del bigm_net
            dump_bounds(bigm_target_file, bigm_time, bigm_ubs)

            ###########################
            ## verified accuracy
            correct=1
            for bn in ub.cpu()[0]:
                if bn >0:
                    correct=0
                    break
            basline_bigm_adam_correct_sum += correct
            basline_bigm_adam_time_sum += float(bigm_time)
            ###########################

        ## baseline-cut-simplex. 
        ## TO-DO (iters need to betuned)
        bigm_adam_params = {
            "bigm_algorithm": "adam",
            "bigm": "only",
            "nb_outer_iter": 500,
            'initial_step_size': 1e-2,
            'final_step_size': 1e-4,
            'betas': (0.9, 0.999)
        }
        bigm_target_file = os.path.join(target_dir, f"baseline-cut-simplex_{lin_approx_string}.txt")
        # if not os.path.exists(bigm_target_file):
        cuda_elided_model = copy.deepcopy(elided_model).cuda()
        bigm_net = Baseline_SimplexLP(cuda_elided_model, params=bigm_adam_params)
        bigm_start = time.time()
        with torch.no_grad():
            bigm_net.optimize = bigm_net.cut_anderson_optimizer
            bigm_net.logger = utils.OptimizationTrace()
            if not args.from_intermediate_bounds:
                bigm_net.define_linear_approximation(cuda_domain)
                ub = bigm_net.upper_bounds[-1]
            else:
                bigm_net.build_model_using_intermediate_net(cuda_domain, (intermediate_lbs, intermediate_ubs), intermediate_net)
                _, ub = bigm_net.compute_lower_bound()
        bigm_end = time.time()
        bigm_time = bigm_end - bigm_start
        bigm_ubs = ub.cpu()

        del bigm_net
        dump_bounds(bigm_target_file, bigm_time, bigm_ubs)

        ###########################
        ## verified accuracy
        correct=1
        for bn in ub.cpu()[0]:
            if bn >0:
                correct=0
                break
        basline_cut_correct_sum += correct
        basline_cut_time_sum += float(bigm_time)
        ###########################

        del intermediate_net

        #######################################
        ### FOR SIMPLEX CONDITIONED METHODS ###
        #######################################

        # # # LIRPA Simplex
        # grb_target_file = os.path.join(target_dir, f"lirpa-dp-simplex{lin_approx_string}-fixed.txt")
        # cuda_elided_model = copy.deepcopy(elided_model).cuda()
        # intermediate_net = SimplexLP([lay for lay in cuda_elided_model], max_batch=3000)
        # cuda_domain = (X.cuda(), args.eps)
        # domain = (X, args.eps)

        # grb_start = time.time()

        # with torch.no_grad():
        #     intermediate_net.set_solution_optimizer('best_naive_simplex', None)
        #     intermediate_net.define_linear_approximation(cuda_domain, no_conv=False,
        #                                                  override_numerical_errors=True)
        # intermediate_ubs = intermediate_net.upper_bounds
        # intermediate_lbs = intermediate_net.lower_bounds

        # ub = intermediate_ubs[-1]
        # grb_end = time.time()
        # grb_time = grb_end - grb_start
        # grb_ubs = torch.Tensor(ub.cpu())
        # dump_bounds(grb_target_file, grb_time, grb_ubs)
        # del intermediate_net


        # ## auto-lirpa-dp Bounds
        # lirpa_target_file = os.path.join(target_dir, f"auto-lirpa-dp{lin_approx_string}-fixed.txt")
        # lirpa_l_target_file = os.path.join(target_dir, f"l_auto-lirpa-dp{lin_approx_string}-fixed.txt")
        # if not os.path.exists(lirpa_l_target_file):
        #     lirpa_params = {
        #         "nb_outer_iter": 5,
        #     }
        #     lirpa_net = SimplexLP(cuda_elided_model, params=lirpa_params,
        #                      store_bounds_progress=len(intermediate_net.weights), debug=True, dp=True)
        #     lirpa_start = time.time()
        #     with torch.no_grad():
        #         lirpa_net.optimize = lirpa_net.auto_lirpa_optimizer
        #         lirpa_net.logger = utils.OptimizationTrace()
        #         lirpa_net.build_model_using_intermediate_net(cuda_domain, (intermediate_lbs, intermediate_ubs), intermediate_net)
        #         lb, ub = lirpa_net.compute_lower_bound()
        #     lirpa_end = time.time()
        #     lirpa_time = lirpa_end - lirpa_start
        #     lirpa_lbs = lb.detach().cpu()
        #     lirpa_ubs = ub.detach().cpu()
        #     dump_bounds(lirpa_target_file, lirpa_time, lirpa_ubs)
        #     dump_bounds(lirpa_l_target_file, lirpa_time, lirpa_lbs)
        #     del lirpa_net

        # ## auto-lirpa-dp Bounds
        # lirpa_target_file = os.path.join(target_dir, f"auto-lirpa-dp-3{lin_approx_string}-fixed.txt")
        # lirpa_l_target_file = os.path.join(target_dir, f"l_auto-lirpa-dp-3{lin_approx_string}-fixed.txt")
        # if not os.path.exists(lirpa_l_target_file):
        #     lirpa_params = {
        #         "nb_outer_iter": 3,
        #     }
        #     lirpa_net = SimplexLP(cuda_elided_model, params=lirpa_params,
        #                      store_bounds_progress=len(intermediate_net.weights), debug=True, dp=True)
        #     lirpa_start = time.time()
        #     with torch.no_grad():
        #         lirpa_net.optimize = lirpa_net.auto_lirpa_optimizer
        #         lirpa_net.logger = utils.OptimizationTrace()
        #         lirpa_net.build_model_using_intermediate_net(cuda_domain, (intermediate_lbs, intermediate_ubs), intermediate_net)
        #         lb, ub = lirpa_net.compute_lower_bound()
        #     lirpa_end = time.time()
        #     lirpa_time = lirpa_end - lirpa_start
        #     lirpa_lbs = lb.detach().cpu()
        #     lirpa_ubs = ub.detach().cpu()
        #     dump_bounds(lirpa_target_file, lirpa_time, lirpa_ubs)
        #     dump_bounds(lirpa_l_target_file, lirpa_time, lirpa_lbs)
        #     print(ub)
        #     ###########################
        #     ## verified accuracy
        #     correct=1
        #     for bn in ub.cpu()[0]:
        #         if bn >0:
        #             correct=0
        #             break
        #     dp_correct_sum += correct
        #     ###########################
            
        #     del lirpa_net

        # ## Gurobi dp-simplex Bounds
        # grb_target_file = os.path.join(target_dir, f"gurobi-dp-simplex{lin_approx_string}-fixed.txt")
        # # grb_l_target_file = os.path.join(target_dir, f"l_gurobi-dp-simplex{lin_approx_string}-fixed.txt")
        # if not os.path.exists(grb_target_file):
        #     grb_net = DP_LinearizedNetwork([lay for lay in elided_model], intermediate_net.weights)
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
        #     # grb_lbs = torch.Tensor(lb).cpu()
        #     dump_bounds(grb_target_file, grb_time, grb_ubs)
        #     # dump_bounds(grb_l_target_file, grb_time, grb_lbs)
        #     del grb_net

        #     ###########################
        #     ## verified accuracy
        #     correct=1
        #     for bn in ub.cpu()[0]:
        #         if bn >0:
        #             correct=0
        #             break
        #     gur_dp_correct_sum += correct
        #     ###########################

        # del intermediate_net


        # print('Tot imges: ', total_images, 'pgd, Crown, Planet, dp acc: ', pgd_correct_sum/float(total_images), crown_correct_sum/float(total_images), planet_correct_sum/float(total_images), dp_correct_sum/float(total_images))
        # print('Tot imges: ', total_images, 'Gurobi Planet, dp acc: ', gur_planet_correct_sum/float(total_images), gur_dp_correct_sum/float(total_images))

        print('Tot imges: ', total_images, 'Bigm-adam, cut acc: ', basline_bigm_adam_correct_sum/float(total_images), basline_cut_correct_sum/float(total_images), basline_bigm_adam_time_sum/float(total_images), basline_cut_time_sum/float(total_images))

        # dump_bounds(grb_target_file, grb_time, grb_ubs)
        # input('')

        # ft.write(str(pgd_correct_sum/float(total_images)))
        # ft.write(",")
        # ft.write(str(crown_correct_sum/float(total_images)))
        # ft.write(",")
        # ft.write(str(planet_correct_sum/float(total_images)))
        # ft.write(",")
        # ft.write(str(dp_correct_sum/float(total_images)))
        # ft.write(",")
        # ft.write(str(total_images))
        # ft.write(",")
        # ft.write(str(idx))
        # ft.write("\n")


        # ft.write(str(gur_planet_correct_sum/float(total_images)))
        # ft.write(",")
        # ft.write(str(gur_dp_correct_sum/float(total_images)))
        # ft.write(",")

        ft.write(str(basline_bigm_adam_time_sum))
        ft.write(",")
        ft.write(str(basline_cut_time_sum))
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
