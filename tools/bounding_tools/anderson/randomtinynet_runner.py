#!/usr/bin/env python
import argparse
import time, os

from plnn.network_linear_approximation import LinearizedNetwork
from plnn.anderson_linear_approximation import AndersonLinearizedNetwork
from plnn.model import load_and_simplify, load_adversarial_problem
from plnn.modules import Flatten
from plnn.explp_solver.solver import ExpLP
from plnn.proxlp_solver.solver import SaddleLP

import torch, copy


"""
    Comparison between the Big-M relaxation for a ReLU-based feedforward NN and the Anderson relaxation.
    Both are solved using Gurobi.
"""


def generate_tiny_random_cnn():
    """
    Generate a very small CNN with random weight for testing purposes.
    :return: a LinearizedNetwork object containing the network as a list of layers, the input domain
    """
    # Input dimensions.
    in_chan = 3
    in_row = 2
    in_col = 2
    #Generate input domain.
    input_domain = torch.zeros((in_chan, in_row, in_col, 2))
    in_lower = (20 - -10) * torch.rand((in_chan, in_row, in_col)) + -10
    in_upper = (50 - (in_lower + 1)) * torch.rand((in_chan, in_row, in_col)) + (in_lower + 1)
    input_domain[:, :, :, 0] = in_lower
    input_domain[:, :, :, 1] = in_upper

    # Generate layers: 2 convolutional (followed by one ReLU each), one final linear.
    out_chan_c1 = 7
    ker_size = 2
    conv1 = torch.nn.Conv2d(in_chan, out_chan_c1, ker_size, stride=2, padding=1)
    conv1.weight = torch.nn.Parameter(torch.randn((out_chan_c1, in_chan, ker_size, ker_size)), requires_grad=False)
    conv1.bias = torch.nn.Parameter(torch.randn(out_chan_c1), requires_grad=False)
    relu1 = torch.nn.ReLU()
    ker_size = 2
    out_chan_c2 = 5
    conv2 = torch.nn.Conv2d(out_chan_c1, out_chan_c2, ker_size, stride=5, padding=0)
    conv2.weight = torch.nn.Parameter(torch.randn((out_chan_c2, out_chan_c1, ker_size, ker_size)), requires_grad=False)
    conv2.bias = torch.nn.Parameter(torch.randn(out_chan_c2), requires_grad=False)
    relu2 = torch.nn.ReLU()
    final = torch.nn.Linear(out_chan_c2, 1)
    final.weight = torch.nn.Parameter(torch.randn((1, out_chan_c2)), requires_grad=False)
    final.bias = torch.nn.Parameter(torch.randn(1), requires_grad=False)
    layers = [conv1, relu1, conv2, relu2, Flatten(), final]

    return LinearizedNetwork(layers), input_domain


def generate_tiny_random_linear(precision):
    """
    Generate a very small fully connected network with random weight for testing purposes.
    :return: a LinearizedNetwork object containing the network as a list of layers, the input domain
    """
    # Input dimensions.
    input_size = 2
    #Generate input domain.
    input_domain = torch.zeros((input_size, 2))
    in_lower = (20 - -10) * torch.rand(input_size) + -10
    in_upper = (50 - (in_lower + 1)) * torch.rand(input_size) + (in_lower + 1)
    input_domain[:, 0] = in_lower
    input_domain[:, 1] = in_upper

    # Generate layers: 2 convolutional (followed by one ReLU each), one final linear.
    out_size1 = 3
    lin1 = torch.nn.Linear(input_size, out_size1)
    lin1.weight = torch.nn.Parameter(torch.randn((out_size1, input_size)), requires_grad=False)
    lin1.bias = torch.nn.Parameter(torch.randn(out_size1), requires_grad=False)
    relu1 = torch.nn.ReLU()
    out_size2 = 3
    lin2 = torch.nn.Linear(out_size1, out_size2)
    lin2.weight = torch.nn.Parameter(torch.randn((out_size2, out_size1)), requires_grad=False)
    lin2.bias = torch.nn.Parameter(torch.randn(out_size2), requires_grad=False)
    relu2 = torch.nn.ReLU()
    final = torch.nn.Linear(out_size2, 1)
    final.weight = torch.nn.Parameter(torch.randn((1, out_size2)), requires_grad=False)
    final.bias = torch.nn.Parameter(torch.randn(1), requires_grad=False)

    input_domain = (input_domain).type(precision)
    lin1.weight = torch.nn.Parameter(lin1.weight.type(precision))
    lin1.bias = torch.nn.Parameter(lin1.bias.type(precision))
    lin2.weight = torch.nn.Parameter(lin2.weight.type(precision))
    lin2.bias = torch.nn.Parameter(lin2.bias.type(precision))
    final.weight = torch.nn.Parameter(final.weight.type(precision))
    final.bias = torch.nn.Parameter(final.bias.type(precision))

    layers = [lin1, relu1, lin2, relu2, final]

    return LinearizedNetwork(layers), input_domain


def domain_split_batch_halves(domain):
    # make the intermediate bounds and the input domain a batch of domains.
    # specifically: divide the input domain into halves

    domain_lb = domain.select(-1, 0)
    domain_ub = domain.select(-1, 1)
    domain_mid = (domain_ub - domain_lb)/2 + domain_lb
    batch_domain = torch.stack([torch.stack([domain_lb, domain_mid], dim=-1),
                                torch.stack([domain_mid, domain_ub], dim=-1)],
                               dim=0)
    assert (batch_domain.select(-1, 1) - batch_domain.select(-1, 0) >= 0).all()
    return batch_domain


def parse_input(precision=torch.float):
    # Parse the input spefications to this file.
    # Return network, domain, args

    # torch.manual_seed(43) #16, 18, 43 (big), 112 (huge) all yield worse bounds if pre-activation bounds
    # are not included in the form of the PLANET relaxation.
    torch.manual_seed(43)

    parser = argparse.ArgumentParser(description="Read a .rlv file"
                                                 "and prove its property.")
    parser.add_argument('--network_filename', type=str,
                        help='.rlv file to prove.')
    parser.add_argument('--reluify_maxpools', action='store_true')
    parser.add_argument('--random_net', type=str, choices=["cnn", "linear"],
                        help='whether to use a random network')
    parser.add_argument('--eta', type=float)
    parser.add_argument('--inner_iter', type=float)
    parser.add_argument('--prox_iter', type=float)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--use_preactivation', action='store_true',
                        help="whether to use pre-activation bounds")
    parser.add_argument('--initialize_ext', action='store_true',
                        help="whether to use bigm gurobi for initialisation")
    parser.add_argument('--bigm', type=str, choices=["only", "init"],
                        help="whether to use the bigm relaxation derivation for init/as a method")
    parser.add_argument('--bigm_algorithm', type=str, choices=["prox", "adam"],
                        help="which bigm algorithm to use, in case one does init or uses it alone")
    args = parser.parse_args()

    if args.random_net and args.network_filename:
        raise IOError("Test either on a random network or on a .rlv, not both.")

    if args.network_filename:
        # Test on .rlv file
        if args.network_filename.endswith('.rlv'):
            rlv_infile = open(args.network_filename, 'r')
            network, domain = load_and_simplify(rlv_infile, LinearizedNetwork)
            rlv_infile.close()
        else:
            network, domain = load_adversarial_problem(args.network_filename, LinearizedNetwork)
    else:
        if args.random_net == "cnn":
            # Test on small generated CNN.
            network, domain = generate_tiny_random_cnn()
        else:
            # Test on small generated fully connected network.
            network, domain = generate_tiny_random_linear(precision)

    if args.reluify_maxpools:
        network.remove_maxpools(domain)

    return network, domain, args


def dual_testing():
    # Test the proximal vs the dual implementation of the Anderson relaxation.

    precision = torch.float
    network, domain, args = parse_input(precision=precision)

    # dual_net = DualAndersonLinearizedNetwork([lay for lay in network.layers], proximal_iters=10,
    #                                          eta=1e-3, bca=10, use_preactivation=args.use_preactivation)  # on the safe side: proximal_iters=100000, eta=5e-5
    # dual_net_start = time.time()
    # dual_net.define_linear_approximation(domain)
    # dual_net_end = time.time()
    # dual_ubs = torch.Tensor(dual_net.upper_bounds[-1])
    # dual_lbs = torch.Tensor(dual_net.lower_bounds[-1])
    # print(f"Dual Gurobi LB: {dual_lbs}")
    # print(f"Dual Gurobi UB: {dual_ubs}")
    # print(f"Dual Gurobi Time: {dual_net_end - dual_net_start}")
    # import pdb; pdb.set_trace()

    # make the input domain a batch of domains
    # batch_domain = domain_split_batch_halves(domain)
    batch_domain = domain.unsqueeze(0).expand(3, *domain.shape).clone()

    gpu = args.gpu
    if gpu:
        exp_layers = [copy.deepcopy(lay).cuda() for lay in
                      network.layers]  # the copy is necessary as .cuda() acts in place for nn.Parameter
        exp_domain = batch_domain.cuda()
    else:
        exp_layers = network.layers
        exp_domain = batch_domain
    intermediate_net = SaddleLP(exp_layers)
    with torch.no_grad():
        intermediate_net.set_solution_optimizer('best_naive_kw', None)
        intermediate_net.define_linear_approximation(exp_domain, no_conv=False)
    intermediate_ubs = intermediate_net.upper_bounds
    intermediate_lbs = intermediate_net.lower_bounds
    explp_params = {
        "anderson_algorithm": "saddle",
        "nb_iter": args.inner_iter,
        'blockwise': True,
        "step_size_dict": {
            'type': 'fw',
            'fw_start': 100
        },
        "bigm": "init",
        "bigm_algorithm": "adam",  # either prox or adam
        "init_params": {
            "nb_outer_iter": 100,
            'initial_step_size': 1e-1,
            'final_step_size': 1e-3,
            'betas': (0.9, 0.999),
            'M_factor': 1.1  # constant factor of the max big-m variables to use for M
        },
        "primal_init_params": {
            'nb_bigm_iter': 600,
            'nb_anderson_iter': 100,
            'initial_step_size': 1e0,  # 1e-2,
            'final_step_size': 1e-2,  # 1e-3,
            'betas': (0.9, 0.999)
        }
    }
    # explp_params = {
    #     "anderson_algorithm": "prox",
    #     "initial_eta": args.eta,  # 1e-2 seems the way to go on CIFAR
    #     "nb_inner_iter": int(args.inner_iter),  # 5?
    #     "nb_outer_iter": args.prox_iter,
    #     "bigm": False,
    #     # "bigm_algorithm": "adam",  # either prox or adam
    #     "init_params": {
    #         "nb_outer_iter": 100,
    #         'initial_step_size': 1e-1,
    #         'final_step_size': 1e-3,
    #         'betas': (0.9, 0.999),
    #         'M_factor': 1000.0  # constant factor of the max big-m variables to use for M
    #     },
    #     "primal_init_params": {
    #         'nb_bigm_iter': 900,
    #         'nb_anderson_iter': 100,
    #         'initial_step_size': 1e-1,  # 1e-2,
    #         'final_step_size': 1e-4,  # 1e-3,
    #         'betas': (0.9, 0.999)
    #     }
    # }
    exp_net = ExpLP(exp_layers, params=explp_params, debug=False, precision=precision, fixed_M=True)
    exp_net_start = time.time()
    with torch.no_grad():
        # exp_net.define_linear_approximation(exp_domain,  no_conv=True)
        exp_net.build_model_using_bounds(exp_domain, (intermediate_lbs, intermediate_ubs))
        lb, ub = exp_net.compute_lower_bound()
    exp_net_end = time.time()
    # exp_lbs = exp_net.lower_bounds[-1].cpu()
    # exp_ubs = exp_net.upper_bounds[-1].cpu()
    exp_lbs = lb.cpu()
    exp_ubs = ub.cpu()
    print(f"ExpLP Time: {exp_net_end - exp_net_start}")
    print(f"ExpLP LB: {exp_lbs}")
    print(f"ExpLP UB: {exp_ubs}")

    with open("all-explp-trials-file.txt", "a") as file:
        file.write(f"LB: {exp_lbs},UB: {exp_ubs},Time: {exp_net_end - exp_net_start},eta: {args.eta},"
                   f"inner_iter: {args.inner_iter},prox_iter: {args.prox_iter}\n")

    # Test on ProxLP (w/ pre-acts) for comparison.
    # gpu = args.gpu
    # if gpu:
    #     prox_layers = [copy.deepcopy(lay).cuda() for lay in network.layers]
    #     prox_domain = batch_domain.cuda()
    # else:
    #     prox_layers = network.layers
    #     prox_domain = batch_domain
    # optprox_params = {
    #     'nb_total_steps': int(args.prox_iter) * int(args.inner_iter),
    #     'max_nb_inner_steps': int(args.inner_iter),
    #     'eta': args.eta,
    #     'log_values': False,
    #     'inner_cutoff': 0,
    #     'maintain_primal': True
    # }
    # optprox_net = SaddleLP(prox_layers)
    # optprox_start = time.time()
    # with torch.no_grad():
    #     optprox_net.set_decomposition('pairs', 'KW')
    #     optprox_net.set_solution_optimizer('optimized_prox', optprox_params)
    #     # optprox_net.define_linear_approximation(prox_domain, force_optim=True)
    #     optprox_net.build_model_using_bounds(prox_domain, (intermediate_lbs, intermediate_ubs))
    #     lb, ub = optprox_net.compute_lower_bound()
    # optprox_end = time.time()
    # # optprox_lbs = optprox_net.lower_bounds[-1].cpu()
    # # optprox_ubs = optprox_net.upper_bounds[-1].cpu()
    # optprox_lbs = lb.cpu()
    # optprox_ubs = ub.cpu()
    # print(f"ProxLP Time: {optprox_end - optprox_start}")
    # print(f"ProxLP LB: {optprox_lbs}")
    # print(f"ProxLP UB: {optprox_ubs}")

    # gpu = args.gpu
    # if gpu:
    #     prox_layers = [copy.deepcopy(lay).cuda() for lay in network.layers]
    #     prox_domain = domain.cuda()
    # else:
    #     prox_layers = network.layers
    #     prox_domain = domain
    # adam_params = {
    #     'nb_steps': int(args.prox_iter) * int(args.inner_iter),
    #     'initial_step_size': 1e-2,
    #     'final_step_size': 1e-4,
    #     'betas': (0.9, 0.999),
    #     'log_values': False,
    #     'to_decomposition': False,
    # }
    # djadam_net = DJRelaxationLP(prox_layers, params=adam_params)
    # djadam_start = time.time()
    # with torch.no_grad():
    #     djadam_net.build_model_using_bounds(prox_domain, (intermediate_lbs, intermediate_ubs))
    #     lb, ub = djadam_net.compute_lower_bound(all_optim=True)
    # djadam_end = time.time()
    # djadam_lbs = lb.cpu()
    # djadam_ubs = ub.cpu()
    # print(f"ProxLP Time: {djadam_end - djadam_start}")
    # print(f"ProxLP LB: {djadam_lbs}")
    # print(f"ProxLP UB: {djadam_ubs}")

    # Anderson Gurobi bounds, stopping at the root node (i.e., solving a LP with cutting planes)
    # n_cuts = 1e1  # n_cuts = 2e5
    # print(f"Executing LP-cut with {n_cuts} cuts")
    # lp_and_grb_net = AndersonLinearizedNetwork(
    #     network.layers, mode="lp-cut", n_cuts=n_cuts, use_preactivation=args.use_preactivation, cuts_per_neuron=True,
    #     store_bounds_progress=1
    # )
    # lp_and_grb_start = time.time()
    # lp_and_grb_net.build_model_using_bounds(domain, ([lbs.cpu() for lbs in intermediate_lbs],
    #                                                  [ubs.cpu() for ubs in intermediate_ubs]))
    # # lp_and_grb_net.define_linear_approximation(domain)
    # lb, ub = lp_and_grb_net.compute_lower_bound()
    # lp_and_grb_end = time.time()
    # lp_and_grb_time = lp_and_grb_end - lp_and_grb_start
    # # lp_and_grb_ubs = torch.Tensor(lp_and_grb_net.upper_bounds[-1])
    # # lp_and_grb_lbs = torch.Tensor(lp_and_grb_net.lower_bounds[-1])
    # lp_and_grb_ubs = torch.Tensor(ub)
    # lp_and_grb_lbs = torch.Tensor(lb)
    # print(f"Anderson cutLP Gurobi LB: {lp_and_grb_lbs}")
    # print(f"Anderson cutLP Gurobi UB: {lp_and_grb_ubs}")
    # print(f"Anderson cutLP Gurobi Time: {lp_and_grb_time}")
    # print(f"Ambiguous ReLUs: {lp_and_grb_net.ambiguous_relus}")



if __name__ == '__main__':

    dual_testing()
