#!/usr/bin/env python
import argparse
import time, os

from plnn.network_linear_approximation import LinearizedNetwork
from plnn.model import load_and_simplify, load_adversarial_problem
from plnn.modules import Flatten
from plnn.explp_solver.solver import ExpLP
from plnn.proxlp_solver.solver import SaddleLP

import torch, copy


from plnn.simplex_solver.solver import SimplexLP
from plnn.simplex_solver.baseline_solver import Baseline_SimplexLP
from plnn.simplex_solver import utils
from plnn.simplex_solver.baseline_gurobi_linear_approximation import Baseline_LinearizedNetwork
from plnn.simplex_solver.gurobi_linear_approximation import Simp_LinearizedNetwork
from plnn.simplex_solver.disjunctive_gurobi import DP_LinearizedNetwork

"""
    Comparison between the Big-M relaxation for a ReLU-based feedforward NN and the Simplex relaxation.
    Both are solved using Gurobi.
"""


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
    X = (in_lower + in_upper)/2

    # Generate layers: 2 convolutional (followed by one ReLU each), one final linear.
    out_size1 = 3
    lin1 = torch.nn.Linear(input_size, out_size1)
    lin1.weight = torch.nn.Parameter(torch.randn((out_size1, input_size)), requires_grad=False)
    lin1.bias = torch.nn.Parameter(torch.randn(out_size1), requires_grad=False)
    # lin1.weight = torch.nn.Parameter(torch.tensor([[2,1],[2,1],[2,1]]), requires_grad=False)
    # lin1.bias = torch.nn.Parameter(torch.tensor([-1.25,-1.25,-1.25]), requires_grad=False)
    relu1 = torch.nn.ReLU()
    
    # out_size2 = 3
    # lin2 = torch.nn.Linear(out_size1, out_size2)
    # lin2.weight = torch.nn.Parameter(torch.randn((out_size2, out_size1)), requires_grad=False)
    # lin2.bias = torch.nn.Parameter(torch.randn(out_size2), requires_grad=False)
    # relu2 = torch.nn.ReLU()

    final = torch.nn.Linear(out_size1, 1)
    final.weight = torch.nn.Parameter(torch.randn((1, out_size1)), requires_grad=False)
    final.bias = torch.nn.Parameter(torch.randn(1), requires_grad=False)

    input_domain = (input_domain).type(precision)

    lin1.weight = torch.nn.Parameter(lin1.weight.type(precision))
    lin1.bias = torch.nn.Parameter(lin1.bias.type(precision))
    # lin2.weight = torch.nn.Parameter(lin2.weight.type(precision))
    # lin2.bias = torch.nn.Parameter(lin2.bias.type(precision))
    final.weight = torch.nn.Parameter(final.weight.type(precision))
    final.bias = torch.nn.Parameter(final.bias.type(precision))

    # layers = [lin1, relu1, lin2, relu2, final]
    layers = [lin1, relu1, final]

    return LinearizedNetwork(layers), input_domain, X


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
    # torch.manual_seed(43)

    parser = argparse.ArgumentParser(description="Read a .rlv file"
                                                 "and prove its property.")
    parser.add_argument('--network_filename', type=str,
                        help='.rlv file to prove.')
    parser.add_argument('--reluify_maxpools', action='store_true')
    parser.add_argument('--random_net', type=str, choices=["cnn", "linear"],
                        help='whether to use a random network')
    parser.add_argument('--eta', type=float)
    parser.add_argument('--eps', type=float, default=10)
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
            network, domain, X = generate_tiny_random_linear(precision)

    if args.reluify_maxpools:
        network.remove_maxpools(domain)

    return network, domain, X.unsqueeze(0), args


def primal_testing():
    # Test the proximal vs the dual implementation of the Anderson relaxation.

    precision = torch.float
    network, domain, X, args = parse_input(precision=precision)
    #######################################
    ### FOR SIMPLEX CONDITIONED METHODS ###
    #######################################
    simp_layers = [copy.deepcopy(lay).cuda() for lay in network.layers]  # the copy is necessary as .cuda() acts in place for nn.Parameter
    cuda_domain = (X.cuda(), args.eps)
    intermediate_net = Baseline_SimplexLP(simp_layers)
    
    print('hello')
    domain = (X, args.eps)
    with torch.no_grad():
        intermediate_net.set_solution_optimizer('best_naive_kw', None)
        intermediate_net.define_linear_approximation(cuda_domain, no_conv=False,
                                                     override_numerical_errors=True)
    intermediate_ubs = intermediate_net.upper_bounds
    intermediate_lbs = intermediate_net.lower_bounds


    ## Gurobi planet-simplex Bounds
    grb_net = Baseline_LinearizedNetwork([lay for lay in network.layers])
    grb_start = time.time()

    grb_net.build_model_using_bounds(domain, ([lbs.cpu().squeeze(0) for lbs in intermediate_lbs], [ubs.cpu().squeeze(0) for ubs in intermediate_ubs]), intermediate_net.init_cut_coeffs, n_threads=4)
    lb, ub = grb_net.compute_lower_bound()

    grb_end = time.time()
    grb_time = grb_end - grb_start
    grb_lbs = lb.cpu()
    grb_ubs = ub.cpu()
    print(f"GurLP Time: {grb_time}")
    print(f"GurLP LB: {grb_lbs}")
    print(f"GurLP UB: {grb_ubs}")

    # with open("gurobi-planet-simplex-fixed.txt", "a") as file:
    #     file.write(f"LB: {grb_lbs},UB: {grb_ubs},Time: {grb_time},eta: {args.eta}\n")
    del grb_net

    ## Gurobi dp-simplex Bounds
    grb_net = DP_LinearizedNetwork([lay for lay in network.layers])
    grb_start = time.time()

    grb_net.build_model_using_bounds(domain, ([lbs.cpu().squeeze(0) for lbs in intermediate_lbs], [ubs.cpu().squeeze(0) for ubs in intermediate_ubs]), intermediate_net.init_cut_coeffs, n_threads=4)
    lb, ub = grb_net.compute_lower_bound()

    grb_end = time.time()
    grb_time = grb_end - grb_start
    grb_dp_lbs = lb.cpu()
    grb_dp_ubs = ub.cpu()
    print(f"GurLP Time: {grb_time}")
    print(f"GurLP LB: {grb_dp_lbs}")
    print(f"GurLP UB: {grb_dp_ubs}")

    with open("gurobi-dp-simplex-fixed.txt", "a") as file:
        file.write(f" Weight:{simp_layers[0].weight.data}, Bias:{simp_layers[0].bias.data}, Gurobi: LB: {grb_lbs},UB: {grb_ubs},Time: {grb_time}. DP: LB: {grb_dp_lbs},UB: {grb_dp_ubs},Time: {grb_time}\n")
    with open("gurobi-different-simplex-fixed.txt", "a") as file:
        if grb_dp_ubs!=grb_ubs:
            file.write(f" Weight:{simp_layers[0].weight.data}, Bias:{simp_layers[0].bias.data}, Gurobi: LB: {grb_lbs},UB: {grb_ubs},Time: {grb_time}. DP: LB: {grb_dp_lbs},UB: {grb_dp_ubs},Time: {grb_time}\n")
    with open("gurobi-big-diff-simplex-fixed.txt", "a") as file:
        if grb_ubs-grb_dp_ubs>0.1:
            file.write(f" Weight:{simp_layers[0].weight.data}, Bias:{simp_layers[0].bias.data}, Gurobi: LB: {grb_lbs},UB: {grb_ubs},Time: {grb_time}. DP: LB: {grb_dp_lbs},UB: {grb_dp_ubs},Time: {grb_time}\n")
    del grb_net

if __name__ == '__main__':

    primal_testing()

    # for i in range(100):
    #     try:
    #         primal_testing()
    #     except:
            # pass