import argparse
import os
import torch
import time
import copy
import sys
from plnn.network_linear_approximation import LinearizedNetwork
from plnn.anderson_linear_approximation import AndersonLinearizedNetwork
from tools.cifar_bound_comparison import load_network, make_elided_models, cifar_loaders, dump_bounds

from plnn.simplex_solver.solver import SimplexLP
from plnn.simplex_solver.baseline_solver import Baseline_SimplexLP
from plnn.simplex_solver import utils
from plnn.simplex_solver.baseline_gurobi_linear_approximation import Baseline_LinearizedNetwork
from plnn.simplex_solver.gurobi_linear_approximation import Simp_LinearizedNetwork
from plnn.simplex_solver.disjunctive_gurobi import DP_LinearizedNetwork

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

def main():
    parser = argparse.ArgumentParser(description="Compute and time a bunch of bounds.")
    parser.add_argument('--network_filename', type=str,
                        help='Path to the network')
    parser.add_argument('--eps', type=float,
                        help='Epsilon - default: 0.0347')
    parser.add_argument('--target_directory', type=str,
                        help='Where to store the results')
    parser.add_argument('--modulo', type=int,
                        help='Numbers of a job to split the dataset over.')
    parser.add_argument('--modulo_do', type=int,
                        help='Which job_id is this one.')
    parser.add_argument('--from_intermediate_bounds', action='store_true',
                        help="if this flag is true, intermediate bounds are computed w/ best of naive-KW")
    args = parser.parse_args()

    results_dir = args.target_directory
    os.makedirs(results_dir, exist_ok=True)
    np.random.seed(0)

    fargs = get_fargs()
    fargs.num_layers = 1

    _, _, test_loaders = get_data_loaders(fargs)

    full_model = MultimodalConcatBowClfRelu(fargs)
    full_model.cuda()
    checkpoint = torch.load(args.network_filename)
    full_model.load_state_dict(checkpoint["state_dict"])
    full_model.eval()

    txtenc=full_model.txtenc
    imgenc=full_model.imgenc
    model = nn.Sequential(*full_model.layers)

    with open ('./data/mmbt_models/list_10k.pkl', 'rb') as fp:
        itemlist = pickle.load(fp)

    selected_weights = txtenc.embed.weight[itemlist]
    selected_weights = selected_weights[:1000, :]

    test_loader = test_loaders['test']


    planet_correct_sum = 0
    dp_correct_sum = 0
    total_images = 0

    ft = open(os.path.join(results_dir, "ver_acc.txt"), "w")

    for idx, (txt, segment, mask, img, tgt) in enumerate(test_loader):
        if (args.modulo is not None) and (idx % args.modulo != args.modulo_do):
            continue

        # if idx%17!=0:
        #     continue

        # if idx<300:
        #     continue

        target_dir = os.path.join(results_dir, f"{idx}")
        os.makedirs(target_dir, exist_ok=True)

        txt, img = txt.cuda(), img.cuda()
        img_emb = imgenc(img)
        img_emb = torch.flatten(img_emb, start_dim=1)

        ### Predicting first way
        out=full_model(txt, img)
        pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()
        print(idx, tgt.item(), pred[0])
        if pred[0]!=tgt.item():
            continue
            # sys.exit()

        total_images +=1

        X = (img_emb, selected_weights, txt.shape[1])
        emb_layer=True
        y=tgt.item()

        elided_models = make_elided_models(model, True)

        elided_model = elided_models[y]
        to_ignore = y


        lin_approx_string = "" if not args.from_intermediate_bounds else "-fromintermediate"

        # compute intermediate bounds with KW. Use only these for every method to allow comparison on the last layer
        #######################################
        ###### FOR BASELINE METHODS FIRST #####
        #######################################
        
        # LIRPA SIMPLEX
        lirpa_target_file = os.path.join(target_dir, f"baseline-lirpa{lin_approx_string}-fixed.txt")
        lirpa_l_target_file = os.path.join(target_dir, f"l_baseline-lirpa{lin_approx_string}-fixed.txt")

        cuda_elided_model = copy.deepcopy(elided_model).cuda()
        intermediate_net = Baseline_SimplexLP([lay for lay in cuda_elided_model], max_batch=3000, tgt=y)
        domain = (X, args.eps)
        cuda_domain = domain

        lirpa_start = time.time()

        with torch.no_grad():
            intermediate_net.set_solution_optimizer("best_naive_simplex", None)
            intermediate_net.define_linear_approximation(cuda_domain, emb_layer=True, no_conv=False)
        intermediate_ubs = intermediate_net.upper_bounds
        intermediate_lbs = intermediate_net.lower_bounds

        ub = intermediate_ubs[-1]
        lb = intermediate_lbs[-1]
        lirpa_end = time.time()
        lirpa_time = lirpa_end - lirpa_start
        lirpa_ubs = torch.Tensor(ub.cpu()).detach()
        lirpa_lbs = torch.Tensor(lb.cpu()).detach()
        dump_bounds(lirpa_target_file, lirpa_time, lirpa_ubs)
        dump_bounds(lirpa_l_target_file, lirpa_time, lirpa_lbs)
        # print(ub, lb, lirpa_time)

        ###########################
        ## verified accuracy
        correct=1
        for bn in ub.cpu()[0]:
            if bn >0:
                correct=0
                break
        planet_correct_sum += correct
        ###########################

        ###########################
        #### pgd bounds
        pgd_target_file = os.path.join(target_dir, f"pgd{lin_approx_string}-fixed.txt")
        l_pgd_target_file = os.path.join(target_dir, f"l_pgd{lin_approx_string}-fixed.txt")

        pgd_bounds = intermediate_net.advertorch_pgd_upper_bound()
        pgd_time = 0
        pgd_lbs = pgd_bounds[:9]
        pgd_ubs = pgd_bounds[9:]
        dump_bounds(pgd_target_file, pgd_time, pgd_ubs)
        dump_bounds(l_pgd_target_file, pgd_time, pgd_lbs)
        ###########################


        del intermediate_net
        del cuda_elided_model
        # del elided_model
        # del intermediate_ubs, intermediate_lbs, grb_ubs
        # del img_emb, txt, img


        #######################################
        ### FOR SIMPLEX CONDITIONED METHODS ###
        #######################################

        # # LIRPA Simplex
        lirpa_target_file = os.path.join(target_dir, f"auto-lirpa-dp{lin_approx_string}-fixed.txt")
        lirpa_l_target_file = os.path.join(target_dir, f"l_auto-lirpa-dp{lin_approx_string}-fixed.txt")

        cuda_elided_model = copy.deepcopy(elided_model).cuda()
        intermediate_net = SimplexLP([lay for lay in cuda_elided_model], max_batch=3000)

        lirpa_start = time.time()

        with torch.no_grad():
            intermediate_net.set_solution_optimizer('best_naive_dp', None)
            intermediate_net.define_linear_approximation(cuda_domain, emb_layer=True, no_conv=False,
                                                         override_numerical_errors=True)
        intermediate_ubs = intermediate_net.upper_bounds
        intermediate_lbs = intermediate_net.lower_bounds

        ub = intermediate_ubs[-1]
        lb = intermediate_lbs[-1]
        lirpa_end = time.time()
        lirpa_time = lirpa_end - lirpa_start
        lirpa_ubs = torch.Tensor(ub.cpu()).detach()
        lirpa_lbs = torch.Tensor(lb.cpu()).detach()
        dump_bounds(lirpa_target_file, lirpa_time, lirpa_ubs)
        dump_bounds(lirpa_l_target_file, lirpa_time, lirpa_lbs)

        # print(ub, lirpa_time)

        ###########################
        ## verified accuracy
        correct=1
        for bn in ub.cpu()[0]:
            if bn >0:
                correct=0
                break
        dp_correct_sum += correct
        ###########################

        del intermediate_net
        del cuda_elided_model
        del elided_model
        # del domain
        # input('')

        print('Tot imges: ', total_images, 'Planet, dp acc: ', planet_correct_sum/float(total_images), dp_correct_sum/float(total_images))

        # dump_bounds(grb_target_file, grb_time, grb_ubs)
        # input('')


        ft.write(str(planet_correct_sum/float(total_images)))
        ft.write("\n")
        ft.write(str(dp_correct_sum/float(total_images)))
        ft.write("\n")
        ft.write(str(total_images))



if __name__ == '__main__':
    main()
