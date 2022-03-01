import argparse
import torch
from tools.cifar_bound_comparison import load_network, make_elided_models, cifar_loaders, dump_bounds
from tools.bab_tools.model_utils import load_cifar_1to1_exp, load_mnist_1to1_exp, load_1to1_eth, load_cifar_l1_network
from plnn.proxlp_solver.solver import SaddleLP
from plnn.proxlp_solver.dj_relaxation import DJRelaxationLP
from plnn.network_linear_approximation import LinearizedNetwork
from plnn.anderson_linear_approximation import AndersonLinearizedNetwork
from plnn.explp_solver.solver import ExpLP

from plnn.simplex_solver.solver import SimplexLP
from plnn.simplex_solver.baseline_solver import Baseline_SimplexLP
from plnn.simplex_solver import utils
from plnn.simplex_solver.baseline_gurobi_linear_approximation import Baseline_LinearizedNetwork
from plnn.simplex_solver.gurobi_linear_approximation import Simp_LinearizedNetwork
from plnn.simplex_solver.disjunctive_gurobi import DP_LinearizedNetwork
from plnn.simplex_solver.simplex_gurobi_linear_approximation import SimplexLinearizedNetwork

import copy, csv
import time, os, math
import pandas as pd
from tools.plot_utils import custom_plot
import matplotlib.pyplot as plt
import matplotlib
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

import numpy as np
import sys

from tools.mmbt_utils import get_fargs
from mmbt.mmbt.data.helpers import get_data_loaders
from mmbt.mmbt.models.bow import GloveBowEncoder
from mmbt.mmbt.models.concat_bow_relu import MultimodalConcatBowClfRelu
from mmbt.mmbt.models.image import ImageEncoder
from tools.bab_tools.model_utils import MultimodalConcatBowOnlyClfRelu
from sklearn.metrics import f1_score, accuracy_score
import torch.nn as nn
import pickle

def run_lower_bounding():
    parser = argparse.ArgumentParser(description="Compute a bound and plot the results")

    # Argument option 1: pass network filename, epsilon, image index.
    parser.add_argument('--network_filename', type=str, help='Path of the network')
    parser.add_argument('--eps', type=float, help='Epsilon')
    parser.add_argument('--img_idx', type=int, default=0)

    # Argument option 2: pass jodie's rlv name, network, and an index i to use the i-th property in that file
    parser.add_argument('--pdprops', type=str, help='pandas table with all props we are interested in')
    parser.add_argument('--nn_name', type=str, help='network architecture name')
    parser.add_argument('--prop_idx', type=int, default=0)

    parser.add_argument('--data', type=str, default='cifar')
    parser.add_argument('--eta', type=float)
    parser.add_argument('--feta', type=float)
    parser.add_argument('--init_step', type=float, default=1e-2)
    parser.add_argument('--fin_step', type=float, default=1e-4)
    parser.add_argument('--out_iters', type=int, default=20)
    parser.add_argument('--prox_momentum', type=float, default=0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--n_cuts', type=float, default=100)

    parser.add_argument('--M_factor', type=float, default=1.1)
    parser.add_argument('--nb_primal_anderson', type=float, default=100)
    parser.add_argument('--anderson_step_type', type=str, choices=["fw", "grad"])
    parser.add_argument('--fw_step_init', type=float, default=100)
    parser.add_argument('--init_out_iters', type=int, default=500)
    parser.add_argument('--cut_frequency', type=int, default=450)
    parser.add_argument('--max_cuts', type=int, default=10)
    parser.add_argument('--cut_add', type=int, default=1)  # nubmer of cuts added at frequency
    parser.add_argument('--init_init_step', type=float, default=1e-2)
    parser.add_argument('--init_fin_step', type=float, default=1e-4)
    parser.add_argument('--prim_init_step', type=float, default=1e-1)
    parser.add_argument('--prim_fin_step', type=float, default=1e-3)
    parser.add_argument('--init_dual_type', type=str, choices=["bigm", "cuts"],
                        help="Dual algorithm for Anderson init.",
                        default="bigm")
    parser.add_argument('--init_inner_iters', type=int)

    parser.add_argument('--max_solver_batch', type=float, default=3000,
                        help='max batch size for bounding computations')
    parser.add_argument('--define_linear_approximation', action='store_true',
                        help="if this flag is true, compute all intermediate bounds w/ the selected algorithm")
    parser.add_argument('--algorithm', type=str, choices=["planet-adam", "proxlp", "saddle-explp", "baseline-bigm-adam-simplex", "simplex", "gurobi", "gurobi-anderson", "dj-adam", "cut", "bigm-adam", "bigm-adam-simplex", "dp-simplex", "gurobi-baseline-planet-simplex", "gurobi-planet-simplex", "gurobi-simplex", "gurobi-dp-simplex", "baseline-lirpa", "auto-lirpa-simplex", "auto-lirpa-dp", "baseline-cut-simplex"], help="which algorithm to use, in case one does init or uses it alone")
    parser.add_argument('--int_bounds', type=str, choices=["best_naive_kw", "best_naive_simplex", "best_naive_dp", "opt_dp"], default="best_naive_kw")

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=1)
    
    args = parser.parse_args()
    np.random.seed(args.seed)

    emb_layer=False
    if args.network_filename and args.data == 'cifar':#UAI
        # Load all the required data, setup the model
        if args.nn_name:
            model = load_cifar_l1_network(args.nn_name)
        else:
            model = load_network(args.network_filename)
        elided_models = make_elided_models(model)
        _, test_loader = cifar_loaders(1)
        for idx, (X, y) in enumerate(test_loader):
            if idx == args.img_idx:
                out = model(X)
                pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()
                print(idx, y.item(), pred[0])
                if y.item()!=pred[0]:
                    print("Incorrect prediction")
                    sys.exit()
                model = elided_models[y.item()]
                break
        domain = torch.stack([X.squeeze(0) - args.eps,
                              X.squeeze(0) + args.eps], dim=-1).cuda()

        # domain = torch.stack([torch.zeros_like(X).squeeze(0),
        #                       torch.ones_like(X).squeeze(0)], dim=-1).cuda()

        # print(model[-1].bias.shape)
        # input('')
    elif args.network_filename and args.data == 'food101':
        # Load all the required data, setup the model

        fargs = get_fargs()
        fargs.num_layers = args.num_layers

        _, _, test_loaders = get_data_loaders(fargs)

        full_model = MultimodalConcatBowClfRelu(fargs)
        full_model.cuda()
        checkpoint = torch.load(args.network_filename)
        full_model.load_state_dict(checkpoint["state_dict"])
        full_model.eval()

        txtenc=full_model.txtenc
        imgenc=full_model.imgenc
        model = nn.Sequential(*full_model.layers)
        elided_models = make_elided_models(model)

        with open ('./data/mmbt_models/list_10k.pkl', 'rb') as fp:
            itemlist = pickle.load(fp)

        selected_weights = txtenc.embed.weight[itemlist]
        selected_weights = selected_weights[:1000, :]

        test_loader = test_loaders['test']

        for idx, (txt, segment, mask, img, tgt) in enumerate(test_loader):
            # print(args.img_idx, idx)
            if idx == args.img_idx:
                txt, img = txt.cuda(), img.cuda()
                img_emb = imgenc(img)
                img_emb = torch.flatten(img_emb, start_dim=1)

                ### Predicting first way
                out=full_model(txt, img)
                pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()
                print(idx, tgt.item(), pred[0])
                if pred[0]!=tgt.item():
                    # continue
                    sys.exit()

               #  ## Predicting second way
               #  txt_emb = txtenc(txt)
               #  X = torch.cat([txt_emb, img_emb], -1)
               #  out1=model(X)
               #  pred1 = torch.nn.functional.softmax(out1, dim=1).argmax(dim=1).cpu().detach().numpy()

               #  ## Predicting third way:  embedding style
                # cat_new = torch.zeros(400005, dtype=torch.float).cuda()
                # for el in range(txt.shape[1]):
                #     cat_new[txt[0,el]] += 1.0
                #     # cat_new[el] += 1.0
                # cat_new=cat_new/txt.shape[1]
                # cat_new=cat_new*500
                # new_txt_emb = (txtenc.embed.weight.T@cat_new).unsqueeze(0)
                ## new_txt_emb = torch.zeros_like(new_txt_emb)
                # X = torch.cat([new_txt_emb, img_emb], -1)
                # out2=model(X)
                # print(torch.nn.functional.softmax(out2, dim=1))
                # pred2 = torch.nn.functional.softmax(out2, dim=1).argmax(dim=1).cpu().detach().numpy()
                # print(pred2)
                # print(tgt, pred, pred1, pred2)
               #  input('')

                X = (img_emb, selected_weights, txt.shape[1])
                emb_layer=True
                y=tgt.item()
                model = elided_models[y]
                break


    elif args.network_filename and args.data == 'mnist':#simplex


        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
        ])
        mnist_test = datasets.MNIST("./mnistdata/", train=False, download=True, transform =transform)
        print('Image id: ', args.img_idx)   

        # 'mnist_oval'
        X, model, test = load_mnist_1to1_exp(args.nn_name, int(args.img_idx) , int(args.prop_idx), mnist_test)
        # domain = torch.stack([X.squeeze(0) - args.eps, X.squeeze(0) + args.eps], dim=-1).cuda()


        domain = torch.stack([torch.clamp(X.squeeze(0) - args.eps, 0, None),
                                          torch.clamp(X.squeeze(0) + args.eps, None, 1.0)], -1).cuda()

        y=test # this is a dummy assignment. test is not the label. It is only used in emb_true case

    elif args.nn_name:
        if args.data in ['cifar', 'mnist_oval'] and args.pdprops != None:#cifar-jodie
            # load all properties
            path = './batch_verification_results/'
            gt_results = pd.read_pickle(path + args.pdprops)
            batch_ids = gt_results.index

            for new_idx, idx in enumerate(batch_ids):
                if idx != args.prop_idx:
                    continue
                print('Prop id: ', args.prop_idx)
                imag_idx = gt_results.loc[idx]["Idx"]
                prop_idx = gt_results.loc[idx]['prop']
                eps_temp = gt_results.loc[idx]["Eps"]
                # skip the nan prop_idx or eps_temp (happens in wide.pkl, jodie's mistake, I guess)
                if (math.isnan(imag_idx) or math.isnan(prop_idx) or math.isnan(eps_temp)):
                    continue

                if args.data == 'cifar':
                    x, model, test = load_cifar_1to1_exp(args.nn_name, int(imag_idx), int(prop_idx))
                    # since we normalise cifar data set, it is unbounded now
                    assert test == prop_idx
                    domain = torch.stack([x.squeeze(0) - eps_temp, x.squeeze(0) + eps_temp], dim=-1)
                else:
                    # 'mnist_oval'
                    x, model, test = load_mnist_1to1_exp(args.nn_name, int(imag_idx), int(prop_idx))
                    assert test == prop_idx
                    domain = torch.stack([torch.clamp(x.squeeze(0) - eps_temp, 0, None),
                                          torch.clamp(x.squeeze(0) + eps_temp, None, 1.0)], -1)


        elif args.pdprops == None: #ETH. mnist-eth and cifar-eth
            csvfile = open('././data/%s_test.csv'%(args.data), 'r')
            tests = list(csv.reader(csvfile, delimiter=','))

            try:#colt nets as used in workshop
                eps_temp = float(args.nn_name[6:]) if args.data == 'mnist' else float(args.nn_name.split('_')[1])/float(args.nn_name.split('_')[2])
            except:
                eps_temp = None

            x, model, test, domain = load_1to1_eth(args.data, args.nn_name, idx=args.prop_idx, test=tests, eps_temp=eps_temp,
                                           max_solver_batch=args.max_solver_batch)
            # since we normalise cifar data set, it is unbounded now
            prop_idx = test

        y=test # this is a dummy assignment. test is not the label

    lin_approx_string = "" if not args.define_linear_approximation else "-allbounds"
    image = args.prop_idx if args.nn_name else args.img_idx

    # compute intermediate bounds with KW. Use only these for every method to allow comparison on the last layer
    # and optimize only the last layer
    cuda_elided_model = copy.deepcopy(model).cuda() if args.network_filename and args.data == 'cifar' else \
        [copy.deepcopy(lay).cuda() for lay in model]
    if args.algorithm == "bigm-adam-simplex" or args.algorithm == "simplex" or args.algorithm == "dp-simplex" or args.algorithm == "gurobi-planet-simplex" or args.algorithm == "gurobi-simplex" or args.algorithm == "gurobi-dp-simplex" or args.algorithm=="auto-lirpa-simplex" or args.algorithm=="auto-lirpa-dp":
        intermediate_net = SimplexLP([lay for lay in cuda_elided_model], max_batch=args.max_solver_batch, seed=args.seed, tgt=y)
        domain = (X, args.eps)
        if not emb_layer:
            cuda_domain = (X.cuda(), args.eps)
        else:
            cuda_domain=domain

    elif args.algorithm == "baseline-bigm-adam-simplex" or args.algorithm == "gurobi-baseline-planet-simplex" or args.algorithm == "baseline-lirpa" or args.algorithm == "baseline-cut-simplex":
        intermediate_net = Baseline_SimplexLP([lay for lay in cuda_elided_model], max_batch=args.max_solver_batch, tgt=y)
        if not emb_layer:
            cuda_domain = (X.cuda(), args.eps)
        else:
            cuda_domain = (X, args.eps)
    else:
        cuda_domain = domain.unsqueeze(0).cuda()
        intermediate_net = SaddleLP([lay for lay in cuda_elided_model], max_batch=args.max_solver_batch)
    
    start_time = time.time()
    with torch.no_grad():
        intermediate_net.set_solution_optimizer(args.int_bounds, None)
        if emb_layer:
            intermediate_net.define_linear_approximation(cuda_domain, emb_layer=emb_layer, no_conv=False)
        else:
            intermediate_net.define_linear_approximation(cuda_domain, no_conv=False)
    intermediate_ubs = intermediate_net.upper_bounds
    intermediate_lbs = intermediate_net.lower_bounds
    print(intermediate_ubs[-1].cpu(), time.time()-start_time)
    print(intermediate_lbs[-1].cpu(), time.time()-start_time)
    # sys.exit()

    # if emb_layer:
    #     # min_softmax_prob = intermediate_net.min_softmax_prob()
    #     # print(min_softmax_prob)
    #     # with open("emb_results.txt", "a") as file:
    #     #     file.write(f"img_idx:{args.img_idx},network_filename:{args.network_filename},int_bounds:{args.int_bounds},min_softmax_prob:{min_softmax_prob},Time:{time.time()-start_time}\n")
    #     sys.exit()

    ##############
    ## pgd bounds
    # pgd_bounds = intermediate_net.pgd_upper_bound()
    # print('pgd_bounds', pgd_bounds[:9])
    # print('pgd_bounds', pgd_bounds[9:])
    if args.algorithm == "baseline-bigm-adam-simplex" or args.algorithm == "gurobi-baseline-planet-simplex" or args.algorithm == "baseline-lirpa":
        pgd_bounds_new = intermediate_net.advertorch_pgd_upper_bound()
        print('pgd_bounds', pgd_bounds_new[:9])
        print('pgd_bounds', pgd_bounds_new[9:])
        # input('')
        # sys.exit()

    folder = f"./timings_{args.data}/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    if args.nn_name:
        folder += f"{args.nn_name}_"

    if args.algorithm == "proxlp":
        # ProxLP
        acceleration_dict = {
            'momentum': args.prox_momentum,  # decent momentum: 0.6 w/ increasing eta
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
        optprox_net = SaddleLP(cuda_elided_model, store_bounds_progress=len(intermediate_net.weights),
                               max_batch=args.max_solver_batch)
        optprox_start = time.time()
        with torch.no_grad():
            optprox_net.set_decomposition('pairs', 'naive')
            optprox_net.set_solution_optimizer('optimized_prox', optprox_params)
            if not args.define_linear_approximation:
                optprox_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                lb, ub = optprox_net.compute_lower_bound()
            else:
                optprox_net.define_linear_approximation(cuda_domain)
                lb = optprox_net.lower_bounds[-1]
                ub = optprox_net.upper_bounds[-1]
        optprox_end = time.time()
        optprox_time = optprox_end - optprox_start
        optprox_lbs = lb.cpu().mean()
        optprox_ubs = ub.cpu().mean()
        print(f"ProxLP Time: {optprox_time}")
        print(f"ProxLP LB: {optprox_lbs}")
        print(f"ProxLP UB: {optprox_ubs}")
        with open("tunings.txt", "a") as file:
            file.write(folder + f"{args.algorithm},LB:{optprox_lbs},Time:{optprox_time},Eta{args.eta},Out-iters:{args.out_iters}\n")

        acceleration_string = f"-mom:{args.prox_momentum}"
        pickle_name = folder + f"timings-img{image}-{args.algorithm},eta:{args.eta}-feta:{args.feta}{acceleration_string}{lin_approx_string}.pickle"
        torch.save(optprox_net.logger, pickle_name)

    elif args.algorithm == "planet-adam":
        adam_params = {
            'nb_steps': args.out_iters,
            'initial_step_size': args.init_step,
            'final_step_size': args.fin_step,
            'betas': (args.beta1, 0.999),
            'log_values': False
        }
        adam_net = SaddleLP(cuda_elided_model, store_bounds_progress=len(intermediate_net.weights),
                            max_batch=args.max_solver_batch)
        adam_start = time.time()
        with torch.no_grad():
            adam_net.set_decomposition('pairs', 'naive')
            adam_net.set_solution_optimizer('adam', adam_params)
            if not args.define_linear_approximation:
                adam_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                lb, ub = adam_net.compute_lower_bound()
            else:
                adam_net.define_linear_approximation(cuda_domain)
                lb = adam_net.lower_bounds[-1]
                ub = adam_net.upper_bounds[-1]
        adam_end = time.time()
        adam_time = adam_end - adam_start
        adam_lbs = lb.cpu().mean()
        adam_ubs = ub.cpu().mean()
        print(f"Planet adam Time: {adam_time}")
        print(f"Planet adam LB: {adam_lbs}")
        print(f"Planet adam UB: {adam_ubs}")
        with open("tunings.txt", "a") as file:
            file.write(folder + f"{args.algorithm},LB:{adam_lbs},Time:{adam_time},Out-iters:{args.out_iters}\n")
        pickle_name = folder + f"timings-img{image}-{args.algorithm},istepsize:{args.init_step},fstepsize:{args.fin_step},beta1:{args.beta1}{lin_approx_string}.pickle"
        torch.save(adam_net.logger, pickle_name)
        planet_logger = torch.load(pickle_name, map_location=torch.device('cpu'))
        color_id = 0
        fig_idx = 0
        print('plotting')
        custom_plot(fig_idx, planet_logger.get_last_layer_time_trace(), planet_logger.get_last_layer_bounds_means_trace(first_half_only_as_ub=True), None, "Time [s]", "Upper Bound", "Upper bound vs time", errorbars=False, labelname=rf"stepsize \in$"  + f"[{args.init_step}, {args.fin_step}]", dotted="-", xlog=False, ylog=False, color=colors[color_id])

    elif args.algorithm == "dj-adam":
        adam_params = {
            'nb_steps': args.out_iters,
            'initial_step_size': args.init_step,
            'final_step_size': args.fin_step,
            'betas': (args.beta1, 0.999),
            'log_values': False
        }
        djadam_net = DJRelaxationLP(cuda_elided_model, params=adam_params,
                                    store_bounds_progress=len(intermediate_net.weights),
                                    max_batch=args.max_solver_batch)
        djadam_start = time.time()
        with torch.no_grad():
            if not args.define_linear_approximation:
                djadam_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                lb, ub = djadam_net.compute_lower_bound()
            else:
                djadam_net.define_linear_approximation(cuda_domain)
                lb = djadam_net.lower_bounds[-1]
                ub = djadam_net.upper_bounds[-1]
        djadam_end = time.time()
        djadam_time = djadam_end - djadam_start
        djadam_lbs = lb.cpu().mean()
        djadam_ubs = ub.cpu().mean()
        print(f"Planet adam Time: {djadam_time}")
        print(f"Planet adam LB: {djadam_lbs}")
        print(f"Planet adam UB: {djadam_ubs}")
        with open("tunings.txt", "a") as file:
            file.write(folder + f"{args.algorithm},LB:{djadam_lbs},Time:{djadam_time},Out-iters:{args.out_iters}\n")
        pickle_name = folder + f"timings-img{image}-{args.algorithm},istepsize:{args.init_step},fstepsize:{args.fin_step},beta1:{args.beta1}{lin_approx_string}.pickle"
        torch.save(djadam_net.logger, pickle_name)

    elif args.algorithm == "gurobi":
        grb_net = LinearizedNetwork([lay for lay in model])
        grb_start = time.time()
        if not args.define_linear_approximation:
            grb_net.build_model_using_bounds(domain, ([lbs.cpu().squeeze(0) for lbs in intermediate_lbs],
                                                      [ubs.cpu().squeeze(0) for ubs in intermediate_ubs]))
            print('model building done')
            lb, ub = grb_net.compute_lower_bound()
        else:
            grb_net.define_linear_approximation(domain, n_threads=4)
            lb = grb_net.lower_bounds[-1]
            ub = grb_net.upper_bounds[-1]
        grb_end = time.time()
        grb_time = grb_end - grb_start
        print(f"Gurobi Time: {grb_time}")
        print(f"Gurobi LB: {lb}")
        print(f"Gurobi UB: {ub}")
        with open("tunings.txt", "a") as file:
            file.write(f"{args.algorithm},LB:{lb},Time:{grb_time}\n")

    elif args.algorithm == "gurobi-anderson":
        bounds_net = AndersonLinearizedNetwork(
            [lay for lay in model], mode="lp-cut", n_cuts=args.n_cuts, cuts_per_neuron=True)
        grb_start = time.time()
        if not args.define_linear_approximation:
            bounds_net.build_model_using_bounds(domain, ([lbs.cpu().squeeze(0) for lbs in intermediate_lbs],
                                                      [ubs.cpu().squeeze(0) for ubs in intermediate_ubs]))
            lb, ub = bounds_net.compute_lower_bound()
        else:
            bounds_net.define_linear_approximation(domain, n_threads=4)
            lb = bounds_net.lower_bounds[-1]
            ub = bounds_net.upper_bounds[-1]
        grb_end = time.time()
        grb_time = grb_end - grb_start
        print(f"Gurobi Anderson Time: {grb_time}")
        print(f"Gurobi Anderson LB: {lb}")
        print(f"Gurobi Anderson UB: {ub}")
        with open("tunings.txt", "a") as file:
            file.write(f"{args.algorithm},LB:{lb},Time:{grb_time}\n")

    elif args.algorithm == "bigm-adam":
        bigm_adam_params = {
            "bigm_algorithm": "adam",
            "bigm": "only",
            "nb_outer_iter": args.out_iters,
            'initial_step_size': args.init_step,
            'final_step_size': args.fin_step,
            'betas': (args.beta1, 0.999)
        }
        bigmadam_net = ExpLP(cuda_elided_model, params=bigm_adam_params,
                             store_bounds_progress=len(intermediate_net.weights), debug=True)
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
        pickle_name = folder + f"timings-img{image}-{args.algorithm},istepsize:{args.init_step},fstepsize:{args.fin_step},beta1:{args.beta1}{lin_approx_string}.pickle"
        torch.save(bigmadam_net.logger, pickle_name)

    elif args.algorithm == "saddle-explp":

        if args.anderson_step_type == "grad":
            step_size_dict = {
                'type': 'grad',
                'initial_step_size': args.init_step,  # 1e-2,  # 5e-3, constant, also performs quite well
                'final_step_size': args.fin_step,  # 1e-3
            }
        else:
            step_size_dict = {
                'type': 'fw',
                'fw_start': args.fw_step_init
            }

        explp_params = {
            "anderson_algorithm": "saddle",
            "nb_iter": args.out_iters,
            'blockwise': False,
            "step_size_dict": step_size_dict,
            "init_params": {
                "nb_outer_iter": args.init_out_iters,  # 500 is fine
                'initial_step_size': args.init_init_step,
                'final_step_size': args.init_fin_step,
                'betas': (0.9, 0.999),
                'M_factor': args.M_factor  # constant factor of the big-m variables to use for M -- 1.1 works very well
            },
            "primal_init_params": {
                'nb_bigm_iter': 100,
                'nb_anderson_iter': args.nb_primal_anderson,  # 0 for bigm init, at least 1000 for cuts init
                'initial_step_size': args.prim_init_step,  # 1e-1,
                'final_step_size': args.prim_fin_step,  # 1e-3,
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
                'nb_iter': 1000,
                'initial_step_size': 1e-3,
                'final_step_size': 1e-6,
            }
            explp_params.update({"cut": "init", "cut_init_params": cut_init_params})

        exp_net = ExpLP(
            [lay for lay in cuda_elided_model], params=explp_params, use_preactivation=True,
            store_bounds_progress=len(intermediate_net.weights), fixed_M=(args.M_factor == 1.0))
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

    elif args.algorithm == "cut":
        color_id = 0
        fig_idx = 0
        # Big-M Prox
        explp_params = {
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
                'initial_step_size': args.init_init_step,
                'final_step_size': args.init_fin_step,
                'betas': (0.9, 0.999)
            },
        }
        exp_net = ExpLP([lay for lay in cuda_elided_model], params=explp_params, use_preactivation=True, store_bounds_progress=len(intermediate_net.weights))
        # exp_net = ExpLP([lay for lay in cuda_elided_model], params=explp_params, use_preactivation=True, store_bounds_progress=len(intermediate_net.weights), initialize_primal_externally=args.initialize_ext)
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
                         f"Out-iters:{args.out_iters}"
                         f"init:{args.init_out_iters}{args.init_step}{args.fin_step}\n")
        pickle_name = folder + f"timings-img{image}-{args.algorithm},outer:{args.out_iters},{args.init_step},{args.fin_step}" \
                               f"cut:{args.cut_frequency},{args.max_cuts},{args.cut_add}" \
                               f"init:{args.init_out_iters}{args.init_init_step}{args.init_fin_step},{lin_approx_string}.pickle"
        torch.save(exp_net.logger, pickle_name)
        cut_logger = torch.load(pickle_name, map_location=torch.device('cpu'))
        custom_plot(fig_idx, cut_logger.get_last_layer_time_trace(), cut_logger.get_last_layer_bounds_means_trace(first_half_only_as_ub=True), None, "Time [s]", "Upper Bound", "Upper bound vs time", errorbars=False, labelname=rf"cut $\alpha \in$"  + f"[{args.cut_frequency}, {args.max_cuts}]", dotted="-", xlog=False, ylog=False, color=colors[color_id])

    # if lb < 0 and args.nn_name:
    #     with open(f"undecided-{args.data}-eps{eps_temp}.txt", "a") as file:
    #         file.write(f"{args.prop_idx}, ")

    elif args.algorithm == "gurobi-baseline-planet-simplex":
        # gurobi_samebudget_target_file = "gurobi-baseline-planet-simplex.txt"
        grb_net = Baseline_LinearizedNetwork([lay.cpu() for lay in model])
        grb_start = time.time()
        if not args.define_linear_approximation:
            grb_net.build_model_using_bounds((X, args.eps), ([lbs.cpu().squeeze(0) for lbs in intermediate_lbs],
                                                      [ubs.cpu().squeeze(0) for ubs in intermediate_ubs]))
            lb, ub = grb_net.compute_lower_bound()
        else:
            grb_net.define_linear_approximation(cuda_domain, n_threads=4)
            lb = grb_net.lower_bounds[-1]
            ub = grb_net.upper_bounds[-1]
        grb_end = time.time()
        grb_time = grb_end - grb_start
        print(f"Gurobi Time: {grb_time}")
        print(f"Gurobi LB: {lb.mean()}")
        print(f"Gurobi UB: {ub.mean()}")
        # dump_bounds(gurobi_samebudget_target_file, ub.cpu())
        with open("tunings.txt", "a") as file:
            file.write(f"{args.algorithm},img:{args.img_idx},prop:{args.prop_idx},UB:{ub.mean()}\n")

        with open("mnist_tunings.txt", "a") as file:
            file.write(f"net:{args.nn_name}, eps:{args.eps}, img:{args.img_idx}, prop:{args.prop_idx}, {ub.mean()},")

    elif args.algorithm == "gurobi-planet-simplex":
        grb_net = Simp_LinearizedNetwork([lay for lay in model], intermediate_net.weights)
        grb_start = time.time()
        if not args.define_linear_approximation:
            # grb_net.build_model_using_bounds(cuda_domain, ([lbs.cpu().squeeze(0) for lbs in intermediate_lbs],
                                                      # [ubs.cpu().squeeze(0) for ubs in intermediate_ubs]), coeff_net.init_cut_coeffs)
            grb_net.build_model_using_bounds(cuda_domain, ([lbs.cpu().squeeze(0) for lbs in intermediate_lbs], [ubs.cpu().squeeze(0) for ubs in intermediate_ubs]), intermediate_net.init_cut_coeffs)
            lb, ub = grb_net.compute_lower_bound()
        else:
            grb_net.define_linear_approximation(cuda_domain, n_threads=4)
            lb = grb_net.lower_bounds[-1]
            ub = grb_net.upper_bounds[-1]
        grb_end = time.time()
        grb_time = grb_end - grb_start
        print(f"Gurobi Time: {grb_time}")
        print(f"Gurobi LB: {lb.mean(), intermediate_lbs[-1].cpu().mean()}")
        print(f"Gurobi UB: {ub.mean(), intermediate_ubs[-1].cpu().mean()}")
        with open("tunings.txt", "a") as file:
            file.write(f"{args.algorithm},LB:{lb},Time:{grb_time}\n")

        if args.nn_name:
            file_n = args.nn_name + ".txt"
            with open(file_n, "a") as file:
                file.write(f"seed:{args.seed}, algo:{args.algorithm}, int_bounds:{args.int_bounds}, net:{args.nn_name}, eps:{args.eps}, img:{args.img_idx}, prop:{args.prop_idx}, gb:{ub.mean().item()}, lirpa:{intermediate_ubs[-1].cpu().mean()}\n")
        else:
            if args.network_filename=="./data/cifar_sgd_8px.pth":
                file_n = "cifar_sgd_8px.txt"
            else:
                file_n = "cifar_madry_8px.txt"
            with open(file_n, "a") as file:
                file.write(f"eps:{args.eps}, img:{args.img_idx}, prop:{args.prop_idx}, algo:{args.algorithm}, seed:{args.seed}, int_bounds:{args.int_bounds}, gb:{ub.mean().item()}, lirpa:{intermediate_ubs[-1].cpu().mean()}\n")

    elif args.algorithm == "gurobi-dp-simplex":
        # gurobi_samebudget_target_file = "gurobi-dp-simplex.txt"
        grb_net = DP_LinearizedNetwork([lay for lay in model], intermediate_net.weights)
        grb_start = time.time()
        if not args.define_linear_approximation:
            # grb_net.build_model_using_bounds(cuda_domain, ([lbs.cpu().squeeze(0) for lbs in intermediate_lbs],
                                                      # [ubs.cpu().squeeze(0) for ubs in intermediate_ubs]), coeff_net.init_cut_coeffs)
            grb_net.build_model_using_bounds(cuda_domain, ([lbs.cpu().squeeze(0) for lbs in intermediate_lbs], [ubs.cpu().squeeze(0) for ubs in intermediate_ubs]), intermediate_net.init_cut_coeffs)
            lb, ub = grb_net.compute_lower_bound()
        else:
            grb_net.define_linear_approximation(cuda_domain, n_threads=4)
            lb = grb_net.lower_bounds[-1]
            ub = grb_net.upper_bounds[-1]
        grb_end = time.time()
        grb_time = grb_end - grb_start
        print(f"Gurobi Time: {grb_time}")
        print(f"Gurobi LB: {lb.mean()}")
        print(f"Gurobi UB: {ub.mean()}")
        # dump_bounds(gurobi_samebudget_target_file, grb_time, ub.cpu())
        with open("tunings.txt", "a") as file:
            file.write(f"{args.algorithm},img:{args.img_idx},prop:{args.prop_idx},UB:{ub.cpu().mean()}\n")

        if args.nn_name:
            file_n = args.nn_name + ".txt"
            with open(file_n, "a") as file:
                file.write(f"seed:{args.seed}, algo:{args.algorithm}, int_bounds:{args.int_bounds}, net:{args.nn_name}, eps:{args.eps}, img:{args.img_idx}, prop:{args.prop_idx}, gb:{ub.mean().item()}, lirpa:{intermediate_ubs[-1].cpu().mean()}\n")
        else:
            file_n = "cifar_madry_8px.txt"
            with open(file_n, "a") as file:
                file.write(f"eps:{args.eps}, img:{args.img_idx}, prop:{args.prop_idx}, algo:{args.algorithm}, seed:{args.seed}, int_bounds:{args.int_bounds}, gb:{ub.mean().item()}, lirpa:{intermediate_ubs[-1].cpu().mean()}\n")

    elif args.algorithm == "gurobi-simplex":
        grb_net = SimplexLinearizedNetwork([lay.cpu() for lay in model], mode="lp-cut", n_cuts=args.n_cuts, store_bounds_progress=len(intermediate_net.weights), intermediate_net_weights=intermediate_net.weights)
        grb_start = time.time()
        if not args.define_linear_approximation:
            # grb_net.build_model_using_bounds(cuda_domain, ([lbs.cpu().squeeze(0) for lbs in intermediate_lbs],
                                                      # [ubs.cpu().squeeze(0) for ubs in intermediate_ubs]), coeff_net.init_cut_coeffs)
            grb_net.build_model_using_bounds(domain, ([lbs.cpu().squeeze(0) for lbs in intermediate_lbs],
                                                      [ubs.cpu().squeeze(0) for ubs in intermediate_ubs]), intermediate_net.init_cut_coeffs, simplex_conditioning=True)
            lb, ub = grb_net.compute_lower_bound()
        else:
            grb_net.define_linear_approximation(cuda_domain, n_threads=4)
            lb = grb_net.lower_bounds[-1]
            ub = grb_net.upper_bounds[-1]
        grb_end = time.time()
        grb_time = grb_end - grb_start
        print(f"Gurobi Time: {grb_time}")
        print(f"Gurobi LB: {lb.mean()}")
        print(f"Gurobi UB: {ub.mean()}")
        with open("tunings.txt", "a") as file:
            file.write(f"{args.algorithm},LB:{lb},Time:{grb_time}\n")

        pickle_name = folder + f"timings-img{image}-{args.algorithm},istepsize:{args.init_step},fstepsize:{args.fin_step},beta1:{args.beta1}{lin_approx_string}.pickle"
        torch.save(grb_net.logger, pickle_name)

    elif args.algorithm == "baseline-bigm-adam-simplex":
        bigm_adam_params = {
                "bigm_algorithm": "adam",
                "bigm": "only",
                "nb_outer_iter": args.out_iters,
                'initial_step_size': args.init_step,
                'final_step_size': args.fin_step,
                'betas': (args.beta1, 0.999)
        }
        bigmadam_net = Baseline_SimplexLP(cuda_elided_model, params=bigm_adam_params,
                             store_bounds_progress=len(intermediate_net.weights), debug=True)
        bigmadam_start = time.time()
        with torch.no_grad():
            bigmadam_net.optimize = bigmadam_net.bigm_subgradient_optimizer
            bigmadam_net.logger = utils.OptimizationTrace()
            # bigmadam_net.set_solution_optimizer('bigm_subgradient_optimizer', None)
            if not args.define_linear_approximation:
                # bigmadam_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                bigmadam_net.build_model_using_intermediate_net(cuda_domain, (intermediate_lbs, intermediate_ubs), intermediate_net)
                lb, ub = bigmadam_net.compute_lower_bound()
            else:
                bigmadam_net.define_linear_approximation(cuda_domain)
                ub = bigmadam_net.upper_bounds[-1]
        bigmadam_end = time.time()
        bigmadam_time = bigmadam_end - bigmadam_start
        bigmadam_lbs = lb.cpu().mean()
        bigmadam_ubs = ub.cpu().mean()
        print(f"BigM adam Time: {bigmadam_time}")
        print(f"BigM adam LB: {bigmadam_lbs}")
        print(f"BigM adam UB: {bigmadam_ubs}")
        with open("tunings.txt", "a") as file:
            file.write(folder + f"{args.algorithm},UB:{bigmadam_ubs},Time:{bigmadam_time},Out-iters:{args.out_iters}\n")
        pickle_name = folder + f"timings-img{image}-{args.algorithm},istepsize:{args.init_step},fstepsize:{args.fin_step},beta1:{args.beta1}{lin_approx_string}.pickle"
        torch.save(bigmadam_net.logger, pickle_name)

    elif args.algorithm == "baseline-cut-simplex":
        bigm_adam_params = {
                "bigm_algorithm": "adam",
                "bigm": "only",
                "nb_outer_iter": args.out_iters,
                'initial_step_size': args.init_step,
                'final_step_size': args.fin_step,
                'betas': (args.beta1, 0.999)
        }
        bigmadam_net = Baseline_SimplexLP(cuda_elided_model, params=bigm_adam_params,
                             store_bounds_progress=len(intermediate_net.weights), debug=True)
        bigmadam_start = time.time()
        with torch.no_grad():
            bigmadam_net.optimize = bigmadam_net.cut_anderson_optimizer
            bigmadam_net.logger = utils.OptimizationTrace()
            if not args.define_linear_approximation:
                # bigmadam_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                bigmadam_net.build_model_using_intermediate_net(cuda_domain, (intermediate_lbs, intermediate_ubs), intermediate_net)
                lb, ub = bigmadam_net.compute_lower_bound()
            else:
                bigmadam_net.define_linear_approximation(cuda_domain)
                ub = bigmadam_net.upper_bounds[-1]
        bigmadam_end = time.time()
        bigmadam_time = bigmadam_end - bigmadam_start
        bigmadam_lbs = lb.cpu().mean()
        bigmadam_ubs = ub.cpu().mean()
        print(f"BigM adam Time: {bigmadam_time}")
        print(f"BigM adam LB: {bigmadam_lbs}")
        print(f"BigM adam UB: {bigmadam_ubs}")
        with open("tunings.txt", "a") as file:
            file.write(folder + f"{args.algorithm},UB:{bigmadam_ubs},Time:{bigmadam_time},Out-iters:{args.out_iters}\n")
        pickle_name = folder + f"timings-img{image}-{args.algorithm},istepsize:{args.init_step},fstepsize:{args.fin_step},beta1:{args.beta1}{lin_approx_string}.pickle"
        torch.save(bigmadam_net.logger, pickle_name)

    elif args.algorithm == "bigm-adam-simplex":
        bigm_adam_params = {
                "bigm_algorithm": "adam",
                "bigm": "only",
                "nb_outer_iter": args.out_iters,
                'initial_step_size': args.init_step,
                'final_step_size': args.fin_step,
                'betas': (args.beta1, 0.999)
        }
        bigmadam_net = SimplexLP(cuda_elided_model, params=bigm_adam_params,
                             store_bounds_progress=len(intermediate_net.weights), debug=True)
        bigmadam_start = time.time()
        with torch.no_grad():
            bigmadam_net.optimize = bigmadam_net.bigm_subgradient_optimizer
            bigmadam_net.logger = utils.OptimizationTrace()
            if not args.define_linear_approximation:
                # bigmadam_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs), intermediate_net)
                bigmadam_net.build_model_using_intermediate_net(cuda_domain, (intermediate_lbs, intermediate_ubs), intermediate_net)
                lb, ub = bigmadam_net.compute_lower_bound()
            else:
                bigmadam_net.define_linear_approximation(cuda_domain)
                ub = bigmadam_net.upper_bounds[-1]
        bigmadam_end = time.time()
        bigmadam_time = bigmadam_end - bigmadam_start
        bigmadam_lbs = lb.cpu().mean()
        bigmadam_ubs = ub.cpu().mean()
        print(f"BigM adam Time: {bigmadam_time}")
        print(f"BigM adam LB: {bigmadam_lbs}")
        print(f"BigM adam UB: {bigmadam_ubs}")
        # with open("tunings.txt", "a") as file:
        #     file.write(folder + f"{args.algorithm},UB:{bigmadam_ubs},Time:{bigmadam_time},Out-iters:{args.out_iters}\n")
        pickle_name = folder + f"timings-img{image}-{args.algorithm},istepsize:{args.init_step},fstepsize:{args.fin_step},beta1:{args.beta1}{lin_approx_string}.pickle"
        torch.save(bigmadam_net.logger, pickle_name)

        file_n = args.nn_name + ".txt"
        with open(file_n, "a") as file:
            file.write(f"algo:{args.algorithm}, net:{args.nn_name}, eps:{args.eps}, img:{args.img_idx}, prop:{args.prop_idx}, {bigmadam_ubs}\n")

    elif args.algorithm == "simplex":
        explp_params = {
            "nb_outer_iter": args.out_iters,
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
                'initial_step_size': args.init_init_step,
                'final_step_size': args.init_fin_step,
                'betas': (0.9, 0.999)
            },
        }
        simplex_net = SimplexLP(cuda_elided_model, params=explp_params,
                             store_bounds_progress=len(intermediate_net.weights), debug=True)
        simplex_start = time.time()
        with torch.no_grad():
            bigmadam_net.optimize = bigmadam_net.simplex_optimizer
            bigmadam_net.logger = utils.OptimizationTrace()
            if not args.define_linear_approximation:
                # simplex_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs), intermediate_net)
                simplex_net.build_model_using_intermediate_net(cuda_domain, (intermediate_lbs, intermediate_ubs), intermediate_net)
                _, ub = simplex_net.compute_lower_bound()
            else:
                simplex_net.define_linear_approximation(cuda_domain)
                ub = simplex_net.upper_bounds[-1]
        simplex_end = time.time()
        simplex_time = simplex_end - simplex_start
        simplex_ubs = ub.cpu().mean()
        print(f"Simplex Time: {simplex_time}")
        print(f"Simplex UB: {simplex_ubs}")
        with open("tunings.txt", "a") as file:
            file.write(folder + f"{args.algorithm},UB:{simplex_ubs},Time:{simplex_time},Out-iters:{args.out_iters}\n")
        pickle_name = folder + f"timings-img{image}-{args.algorithm},outer:{args.out_iters},{args.init_step},{args.fin_step}" \
                               f"cut:{args.cut_frequency},{args.max_cuts},{args.cut_add}" \
                               f"init:{args.init_out_iters}{args.init_init_step}{args.init_fin_step}.pickle"
        torch.save(simplex_net.logger, pickle_name)

    elif args.algorithm == "dp-simplex":
        explp_params = {
            "nb_outer_iter": args.out_iters,
            'bigm': "init",
            'cut': "only",
            "bigm_algorithm": "adam",# either prox or adam
            'betas': (0.9, 0.999),
            'initial_step_size': args.init_step,
            'final_step_size': args.fin_step,
            "init_params": {
                "nb_outer_iter": args.init_out_iters,
                'initial_step_size': args.init_init_step,
                'final_step_size': args.init_fin_step,
                'betas': (0.9, 0.999)
            },
        }
        dp_net = SimplexLP(cuda_elided_model, params=explp_params,
                             store_bounds_progress=len(intermediate_net.weights), debug=True)
        dp_start = time.time()
        with torch.no_grad():
            bigmadam_net.optimize = bigmadam_net.dp_optimizer
            bigmadam_net.logger = utils.OptimizationTrace()
            if not args.define_linear_approximation:
                # simplex_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs), intermediate_net)
                dp_net.build_model_using_intermediate_net(cuda_domain, (intermediate_lbs, intermediate_ubs), intermediate_net)
                _, ub = dp_net.compute_lower_bound()
            else:
                dp_net.define_linear_approximation(cuda_domain)
                ub = dp_net.upper_bounds[-1]
        dp_end = time.time()
        dp_time = dp_end - dp_start
        dp_ubs = ub.cpu().mean()
        print(f"Dp Time: {dp_time}")
        print(f"Dp UB: {dp_ubs}")
        # with open("tunings.txt", "a") as file:
        #     file.write(folder + f"{args.algorithm},UB:{dp_ubs},Time:{dp_time},Out-iters:{args.out_iters}\n")
        pickle_name = folder + f"timings-img{image}-{args.algorithm},istepsize:{args.init_step},fstepsize:{args.fin_step},beta1:{args.beta1}{lin_approx_string}.pickle"
        torch.save(dp_net.logger, pickle_name)

        file_n = args.nn_name + ".txt"
        with open(file_n, "a") as file:
            file.write(f"algo:{args.algorithm}, net:{args.nn_name}, eps:{args.eps}, img:{args.img_idx}, prop:{args.prop_idx}, {dp_ubs}\n")

    elif args.algorithm == "baseline-lirpa":
        lirpa_params = {
                "bigm_algorithm": "adam",
                "bigm": "only",
                "nb_outer_iter": args.out_iters,
                'initial_step_size': args.init_step,
                'final_step_size': args.fin_step,
                'betas': (args.beta1, 0.999)
        }
        lirpa_net = Baseline_SimplexLP(cuda_elided_model, params=lirpa_params,
                             store_bounds_progress=len(intermediate_net.weights), debug=True)
        lirpa_start = time.time()
        with torch.no_grad():
            lirpa_net.optimize = lirpa_net.auto_lirpa_optimizer
            lirpa_net.logger = utils.OptimizationTrace()
            if not args.define_linear_approximation:
                # lirpa_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                lirpa_net.build_model_using_intermediate_net(cuda_domain, (intermediate_lbs, intermediate_ubs), intermediate_net)
                lb, ub = lirpa_net.compute_lower_bound()
            else:
                lirpa_net.define_linear_approximation(cuda_domain)
                ub = lirpa_net.upper_bounds[-1]
        lirpa_end = time.time()
        lirpa_time = lirpa_end - lirpa_start
        print(lb)
        lirpa_lbs = lb.cpu().mean()
        lirpa_ubs = ub.cpu().mean()
        print(f"Lirpa Time: {lirpa_time}")
        print(f"Lirpa LB: {lirpa_lbs}")
        print(f"Lirpa UB: {lirpa_ubs}")
        with open("tunings.txt", "a") as file:
            file.write(folder + f"{args.algorithm},UB:{lirpa_ubs},Time:{lirpa_time},Out-iters:{args.out_iters}\n")
        pickle_name = folder + f"timings-img{image}-{args.algorithm},istepsize:{args.init_step},fstepsize:{args.fin_step},beta1:{args.beta1}{lin_approx_string}.pickle"
        torch.save(lirpa_net.logger, pickle_name)

    elif args.algorithm == "auto-lirpa-simplex":
        lirpa_params = {
                "bigm_algorithm": "adam",
                "bigm": "only",
                "nb_outer_iter": args.out_iters,
                'initial_step_size': args.init_step,
                'final_step_size': args.fin_step,
                'betas': (args.beta1, 0.999)
        }
        lirpa_net = SimplexLP(cuda_elided_model, params=lirpa_params,
                             store_bounds_progress=len(intermediate_net.weights), debug=True, dp=False)
        lirpa_start = time.time()
        with torch.no_grad():
            lirpa_net.optimize = lirpa_net.auto_lirpa_optimizer
            lirpa_net.logger = utils.OptimizationTrace()
            # lirpa_net.set_solution_optimizer('bigm_subgradient_optimizer', None)
            if not args.define_linear_approximation:
                # lirpa_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                lirpa_net.build_model_using_intermediate_net(cuda_domain, (intermediate_lbs, intermediate_ubs), intermediate_net)
                lb, ub = lirpa_net.compute_lower_bound()
            else:
                lirpa_net.define_linear_approximation(cuda_domain)
                ub = lirpa_net.upper_bounds[-1]
        lirpa_end = time.time()
        lirpa_time = lirpa_end - lirpa_start
        lirpa_lbs = lb.cpu().mean()
        lirpa_ubs = ub.cpu().mean()
        print(f"Lirpa Time: {lirpa_time}")
        print(f"Lirpa LB: {lirpa_lbs}")
        print(f"Lirpa UB: {lirpa_ubs}")
        with open("tunings.txt", "a") as file:
            file.write(folder + f"{args.algorithm},UB:{lirpa_ubs},Time:{lirpa_time},Out-iters:{args.out_iters}\n")
        pickle_name = folder + f"timings-img{image}-{args.algorithm},istepsize:{args.init_step},fstepsize:{args.fin_step},beta1:{args.beta1}{lin_approx_string}.pickle"
        torch.save(lirpa_net.logger, pickle_name)

    elif args.algorithm == "auto-lirpa-dp":
        lirpa_params = {
                "bigm_algorithm": "adam",
                "bigm": "only",
                "nb_outer_iter": args.out_iters,
                'initial_step_size': args.init_step,
                'final_step_size': args.fin_step,
                'betas': (args.beta1, 0.999)
        }
        lirpa_net = SimplexLP(cuda_elided_model, params=lirpa_params,
                             store_bounds_progress=len(intermediate_net.weights), debug=True, dp=True)
        lirpa_start = time.time()
        with torch.no_grad():
            lirpa_net.optimize = lirpa_net.auto_lirpa_optimizer
            lirpa_net.logger = utils.OptimizationTrace()
            # lirpa_net.set_solution_optimizer('bigm_subgradient_optimizer', None)
            if not args.define_linear_approximation:
                # lirpa_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                lirpa_net.build_model_using_intermediate_net(cuda_domain, (intermediate_lbs, intermediate_ubs), intermediate_net)
                lb, ub = lirpa_net.compute_lower_bound()
            else:
                lirpa_net.define_linear_approximation(cuda_domain)
                ub = lirpa_net.upper_bounds[-1]
        lirpa_end = time.time()
        lirpa_time = lirpa_end - lirpa_start
        print(lb)
        lirpa_lbs = lb.cpu().mean()
        lirpa_ubs = ub.cpu().mean()
        print(f"Lirpa Time: {lirpa_time}")
        print(f"Lirpa LB: {lirpa_lbs}")
        print(f"Lirpa UB: {lirpa_ubs}")
        with open("tunings.txt", "a") as file:
            file.write(folder + f"{args.algorithm},UB:{lirpa_ubs},Time:{lirpa_time},Out-iters:{args.out_iters}\n")
        pickle_name = folder + f"timings-img{image}-{args.algorithm},istepsize:{args.init_step},fstepsize:{args.fin_step},beta1:{args.beta1}{lin_approx_string}.pickle"
        torch.save(lirpa_net.logger, pickle_name)


if __name__ == '__main__':
    run_lower_bounding()
