import copy
import math
import time
import torch
from torch import nn
from torch.nn import functional as F

from plnn.network_linear_approximation import LinearizedNetwork

from plnn.simplex_solver.utils import LinearOp, ConvOp, prod, OptimizationTrace, ProxOptimizationTrace, bdot
from plnn.simplex_solver.utils import BatchLinearOp, BatchConvOp, get_relu_mask, compute_output_padding, create_final_coeffs_slice
from plnn.simplex_solver.by_pairs import ByPairsDecomposition, DualVarSet
from plnn.branch_and_bound.utils import ParentInit
from plnn.simplex_solver import bigm_optimization, simplex_optimization, dp_optimization
from plnn.simplex_solver import utils
from tools.colt_layers import Flatten
import itertools
import numpy as np

from plnn.simplex_solver.simplex_lirpa_optimization import autolirpa_opt_dp, AutoLirpa
import sys

default_params = {
    "initial_eta": 100,
    "nb_inner_iter": 100,
    "nb_outer_iter": 10,
    "nb_iter": 500,
    "anderson_algorithm": "saddle",  # either saddle or prox
    "bigm": "init", # whether to use the specialized bigm relaxation solver. Alone: "only". As initializer: "init"
    "bigm_algorithm": "adam",  # either prox or adam
    "init_params": {
        'nb_outer_iter': 100,
        'initial_step_size': 1e-3,
        'final_step_size': 1e-6,
        'betas': (0.9, 0.999)
    }
}
eps_tol=0.01

class SimplexLP(LinearizedNetwork):
    '''
    The objects of this class are s.t: the input lies in l1.
    1. the first layer is conditioned s.t that the input lies in a probability simplex.
    2. Simplex Propagation: the layers(including first layer, excluding output linear layer) are conditioned s.t their output also lies in simplex, this is done using the simplex cut.
    3. better ib bounds are computed using these propagated simplices

    return: so the define_linear_approximation fn returns a net whose input lies in simplex and all other intermediate layers lie in a simplex too
    '''

    def __init__(self, 
        layers, 
        debug=False, 
        params=None, 
        view_tensorboard=False, 
        precision=torch.float, 
        store_bounds_progress=-1, 
        store_bounds_primal=False, 
        max_batch=20000, 
        seed=0,
        dp=True,
        tgt=1):
        """
        :param store_bounds_progress: whether to store bounds progress over time (-1=False 0=True)
        :param store_bounds_primal: whether to store the primal solution used to compute the final bounds
        :param max_batch: maximal batch size for parallel bounding çomputations over both output neurons and domains
        """
        self.optimizers = {
            'init': self.init_optimizer,
            'best_naive_kw': self.best_naive_kw_optimizer,
            'bigm_subgradient_optimizer': self.bigm_subgradient_optimizer,
            'simplex_optimizer': self.simplex_optimizer,
            'dp_optimizer': self.dp_optimizer,
            'best_naive_simplex': self.best_naive_simplex_optimizer,
            'best_naive_dp': self.best_naive_dp_optimizer,
            'opt_dp': self.opt_dp_optimizer,
            'auto_lirpa_optimizer': self.auto_lirpa_optimizer,
        }

        self.layers = layers
        self.net = nn.Sequential(*layers)

        for param in self.net.parameters():
            param.requires_grad = False

        self.decomposition = ByPairsDecomposition('KW')
        self.optimize, _ = self.init_optimizer(None)

        self.store_bounds_progress = store_bounds_progress
        self.store_bounds_primal = store_bounds_primal

        # store which relus are ambiguous. 1=passing, 0=blocking, -1=ambiguous. Shape: dom_batch_size x layer_width
        self.relu_mask = []
        self.max_batch = max_batch

        self.debug = debug
        self.view_tensorboard = view_tensorboard
        if self.view_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(comment="bigm")

        self.params = dict(default_params, **params) if params is not None else default_params
        # self.bigm_init = self.params["bigm"] and (self.params["bigm"] == "init")
        # self.bigm_only = self.params["bigm"] and (self.params["bigm"] == "only")
        # self.cut_init = ("cut" in self.params) and self.params["cut"] and (self.params["cut"] == "init")
        # self.cut_only = ("cut" in self.params) and self.params["cut"] and (self.params["cut"] == "only")

        self.precision = precision
        self.init_cut_coeffs = []

        self.seed = seed
        self.dp = dp

        self.tgt = tgt #target label

    def set_decomposition(self, decomp_style, decomp_args, ext_init=None):
        decompositions = {
            'pairs': ByPairsDecomposition
        }
        assert decomp_style in decompositions
        self.decomposition = decompositions[decomp_style](decomp_args)

        if ext_init is not None:
            self.decomposition.set_external_initial_solution(ext_init)

    def set_solution_optimizer(self, method, method_args=None):
        assert method in self.optimizers
        self.optimize, _ = self.optimizers[method](method_args)
        self.logger = utils.OptimizationTrace()

    @staticmethod
    def get_preact_bounds_first_layer(X, eps, layer, no_conv=False, conditioning=True, seed=0):
        '''
        This function does 4 jobs:
        1st is to condition the first layer from l1 to simplex
        2nd 2a: it gets the init_cut_coeff. this is the b in lambda x leq b kind of cuts, for the special init in which the lamba is 1
        2b: then it conditions the layer with this coeff, so now the output also lies in simplex
        3rd it constructs the batch conv or batch conv linear layer.
        4th is to get the preact bounds of the first layer
        '''
        w_1 = layer.weight
        b_1 = layer.bias
        X_vector = X.view(X.shape[0], -1)
        dim = X_vector.shape[1]
        E = torch.zeros(dim, 2*dim).cuda(X.get_device())
        for row_idx in range(0,dim):
            E[row_idx, 2*row_idx] = 1
            E[row_idx, 2*row_idx+1] = -1

        init_cut_coeff = None
        if isinstance(layer, nn.Linear):
            ###############
            # STEP-1- Conditioning this layer so that input lies in simplex
            ###############
            cond_w_1 = eps * w_1 @ E
            if X_vector.dim() ==2:
                X_vector = X_vector.squeeze(0)
            cond_b_1 = b_1 + X_vector @ w_1.t()
            ###############
            ###############

            ###############
            # STEP-2- construct BatchLinearOp
            ###############
            cond_w_1_unsq = cond_w_1.unsqueeze(0)
            cond_layer = BatchLinearOp(cond_w_1_unsq, cond_b_1)
            ###############

            if conditioning:
                ###############
                # Getting the lambda
                ###############
                if seed!=0:
                    a = np.random.random(b_1.shape[0])
                    a /= a.sum()
                    lmbd=torch.from_numpy(a).float().to(cond_w_1.device)
                else:### default version
                    lmbd=torch.ones(b_1.shape[0]).float().to(cond_w_1.device)
                ###############

                init_cut_coeff = cond_layer.simplex_conditioning(lmbd, conditioning=True)
            
            ###############
            # STEP-4- calculating pre-activation bounds of conditioned layer
            ###############
            # W_min = torch.stack([min(min(row),0) for row in cond_w_1])
            # W_max = torch.stack([max(max(row),0) for row in cond_w_1])
            W_min, _ = torch.min(cond_layer.weights.squeeze(0), 1)
            W_min = torch.clamp(W_min, None, 0)
            W_max, _ = torch.max(cond_layer.weights.squeeze(0), 1)
            W_max = torch.clamp(W_max, 0, None)
            l_1 = W_min + cond_layer.bias
            u_1 = W_max + cond_layer.bias
            l_1 = l_1.unsqueeze(0)
            u_1 = u_1.unsqueeze(0)
            assert (u_1 - l_1).min() >= 0, "Incompatible bounds"
            ###############

            if isinstance(cond_layer, LinearOp) and (X.dim() > 2):
                # This is the first LinearOp, so we need to include the flattening
                cond_layer.flatten_from((X.shape[1], X.shape[2], 2*X.shape[3]))

        elif isinstance(layer, nn.Conv2d):
            ###############
            # STEP-1- Conditioning this layer so that input lies in simplex
            ###############
            out_bias = F.conv2d(X, w_1, b_1,
                                layer.stride, layer.padding,
                                layer.dilation, layer.groups)

            output_padding = compute_output_padding(X, layer) #can comment this to recover old behaviour

            cond_layer = BatchConvOp(w_1, out_bias, b_1,
                                    layer.stride, layer.padding,
                                    layer.dilation, layer.groups, output_padding)
            cond_layer.add_prerescaling(eps*E)# this is vector prescaling 
            ###############
            ###############

            if conditioning:
                X_in = torch.zeros(X.shape[0],X.shape[1],X.shape[2],2*X.shape[3]).cuda(X.get_device())
                cond_layer_linear = cond_layer.equivalent_linear(X_in.squeeze(0))

                ###############
                # Getting the lambda
                ###############
                if seed!=0:
                    a = np.random.random(cond_layer_linear.bias.shape[0])
                    a /= a.sum()
                    lmbd=torch.from_numpy(a).float().to(w_1.device)
                else:### default version
                    lmbd=torch.ones(cond_layer_linear.bias.shape[0], dtype=torch.float, device=w_1.device)
                ###############

                init_cut_coeff = cond_layer.simplex_conditioning(X, lmbd, conditioning=True)

            ###############
            # STEP-3- calculating pre-activation bounds.
            ###############
            weight_matrix= cond_layer.weights.view(cond_layer.weights.shape[0], -1)
            a_min, _ = torch.min(weight_matrix, 1)
            a_max, _ = torch.max(weight_matrix, 1)
            W_min = eps*torch.min(a_min, -a_max)
            W_max = eps*torch.max(a_max, -a_min)
            # W_min = eps*torch.stack([min(min(row), min(-row)) for row in weight_matrix])
            # W_max = eps*torch.stack([max(max(row),max(-row)) for row in weight_matrix])
            
            out_shape = cond_layer.get_output_shape(X_in.unsqueeze(0).shape)
            l_1 = torch.zeros(out_shape[0]*out_shape[1],out_shape[2],out_shape[3],out_shape[4]).cuda(X.get_device())
            u_1 = torch.zeros_like(l_1)
            for j in range(l_1.shape[0]):
                for i in range(l_1.shape[1]):
                    l_1[j,i,:,:] = W_min[i]
                for i in range(u_1.shape[1]):
                    u_1[j,i,:,:] = W_max[i]
            l_1 = l_1 + out_bias / init_cut_coeff
            u_1 = u_1 + out_bias / init_cut_coeff

            ## lambda conditioning
            # since l_1 is an element-wise min and lambda >=0, so max will occur at max of l_1
            # l_1 = l_1 * lmbd.unsqueeze(0).view_as(out_bias)
            # u_1 = u_1 * lmbd.unsqueeze(0).view_as(out_bias)
            ##

            ###############
            ###############

            if no_conv:
                cond_layer = cond_layer_linear.equivalent_linear(X)

        assert (u_1 - l_1).min() >= 0, "Incompatible bounds"

        return l_1, u_1, init_cut_coeff, cond_layer, lmbd

    @staticmethod
    def get_preact_bounds_first_emb_layer(X, eps, layer, no_conv=False, conditioning=True, seed=0):
        '''
        This function does 4 jobs:
        1st is to condition the first layer from l1 to simplex
        2nd 2a: it gets the init_cut_coeff. this is the b in lambda x leq b kind of cuts, for the special init in which the lamba is 1
        2b: then it conditions the layer with this coeff, so now the output also lies in simplex
        3rd it constructs the batch conv or batch conv linear layer.
        4th is to get the preact bounds of the first layer
        '''
        w_1 = layer.weight
        b_1 = layer.bias

        img_emb, selected_weights, num_words = X 

        # X_vector = X.view(X.shape[0], -1)
        # dim = X_vector.shape[1]
        # E = torch.zeros(dim, 2*dim).cuda(X.get_device())
        # for row_idx in range(0,dim):
        #     E[row_idx, 2*row_idx] = 1
        #     E[row_idx, 2*row_idx+1] = -1

        init_cut_coeff = None
        if isinstance(layer, nn.Linear):
            ###############
            # STEP-1- Conditioning this layer so that input lies in simplex
            ###############
            new_txt_emb = (torch.zeros(300, dtype=torch.float).cuda()).unsqueeze(0)
            concat_emb = torch.cat([new_txt_emb, img_emb], -1).squeeze(0)
            cond_b_1 = b_1 + concat_emb @ w_1.t()

            cond_w_1 = num_words*w_1[:, :300] @ selected_weights.t() # multiplied with 500 because network takes input that is not conditioned to sum to 1. but the simplex will sum to 1.
            # cond_w_1 = w_1[:, :300] @ selected_weights.t() # multiplied with 500 because network takes input that is not conditioned to sum to 1. but the simplex will sum to 1.
            # cond_w_1 = w_1[:, :300] @ selected_weights.t() 
            ###############
            ###############

            ###############
            # STEP-2- construct BatchLinearOp
            ###############
            cond_w_1_unsq = cond_w_1.unsqueeze(0)
            cond_layer = BatchLinearOp(cond_w_1_unsq, cond_b_1)
            ###############

            if conditioning:
                ###############
                # Getting the lambda
                ###############
                if seed!=0:
                    a = np.random.random(b_1.shape[0])
                    a /= a.sum()
                    lmbd=torch.from_numpy(a).float().to(cond_w_1.device)
                else:### default version
                    lmbd=torch.ones(b_1.shape[0]).float().to(cond_w_1.device)
                ###############

                init_cut_coeff = cond_layer.simplex_conditioning(lmbd, conditioning=True)
            
            ###############
            # STEP-4- calculating pre-activation bounds of conditioned layer
            ###############
            # W_min = torch.stack([min(min(row),0) for row in cond_w_1])
            # W_max = torch.stack([max(max(row),0) for row in cond_w_1])
            W_min, _ = torch.min(cond_layer.weights.squeeze(0), 1)
            W_min = torch.clamp(W_min, None, 0)
            W_max, _ = torch.max(cond_layer.weights.squeeze(0), 1)
            W_max = torch.clamp(W_max, 0, None)
            l_1 = W_min + cond_layer.bias
            u_1 = W_max + cond_layer.bias
            l_1 = l_1.unsqueeze(0)
            u_1 = u_1.unsqueeze(0)
            assert (u_1 - l_1).min() >= 0, "Incompatible bounds"
            ###############


        return l_1, u_1, init_cut_coeff, cond_layer, lmbd

    @staticmethod
    def build_simp_layer(prev_ub, prev_lb, layer, init_cut_coeffs, lmbd_prev, no_conv=False, orig_shape_prev_ub=None, conditioning=True, seed=0, prop_simp_bounds=False):
        '''
        This function also does the conditioning using init_cut_coeff
        This function also calculates the init_cut_coeff which is the b coefficient for lambda x leq b cut
        This function return a ConvOp or LinearOp object depending on the layer type. 
        This function also computes the pre-activation bounds
        '''
        w_kp1 = layer.weight
        b_kp1 = layer.bias

        obj_layer_orig = None
        init_cut_coeff = None
        lmbd = None

        l_kp1 = None
        u_kp1 = None

        if isinstance(layer, nn.Conv2d):
            
            #################################
            ## PREVIOUS LAYER CONDITIONING ##
            #################################
            w_kp1 = (w_kp1) * init_cut_coeffs[-1]
            #################################
            #################################

            output_padding = compute_output_padding(prev_ub, layer)
            obj_layer = ConvOp(w_kp1, b_kp1,
                               layer.stride, layer.padding,
                               layer.dilation, layer.groups, output_padding)

            if conditioning:
                prev_ub_unsq = prev_ub.squeeze(0)
                cond_layer_linear = obj_layer.equivalent_linear(prev_ub_unsq)

                ###############
                # Getting the lambda
                ###############
                if seed!=0:
                    a = np.random.random(cond_layer_linear.bias.shape[0])
                    a /= a.sum()
                    lmbd=torch.from_numpy(a).float().to(w_kp1.device)
                else:### default version
                    lmbd=torch.ones(cond_layer_linear.bias.shape[0], dtype=torch.float, device=w_kp1.device)
                ###############

                ###############
                # STEP-2a- finding the cut
                ###############
                cond_layer_linear_wb = cond_layer_linear.weights + cond_layer_linear.bias[:,None]
                wb_clamped = torch.clamp(cond_layer_linear_wb, 0, None)
                lambda_wb_clamped = (wb_clamped.T*lmbd).T
                wb_col_sum = torch.sum(lambda_wb_clamped, 0)
                # for origin point
                b_clamped = torch.clamp(cond_layer_linear.bias, 0, None)
                lambda_b_clamped = b_clamped*lmbd
                b_sum = torch.sum(lambda_b_clamped)
                #
                init_cut_coeff = max(max(wb_col_sum),b_sum)
                if init_cut_coeff==0:
                    init_cut_coeff = torch.ones_like(init_cut_coeff)
                ###############

                ###############
                # STEP-2b- Conditioning this layer. now output also lies in simplex
                ###############
                # Needs conditioning both weights and bias
                # 1. has alpha scaling
                # 2. has lambda scaling
                w_kp1 = w_kp1 / init_cut_coeff
                b_kp1 = b_kp1 / init_cut_coeff
                # cond_layer.postscale = lmbd.unsqueeze(0).view_as(cond_layer.bias)

            # STEP-4- calculating pre-activation bounds.
            if prop_simp_bounds:
                weight_matrix= w_kp1.view(w_kp1.shape[0], -1)

                W_min, _ = torch.min(weight_matrix, 1)
                W_min_cl = torch.clamp(W_min, None, 0)

                W_max, _ = torch.max(weight_matrix, 1)
                W_max_cl = torch.clamp(W_max, 0, None)

                # W_min = torch.stack([min(row) for row in weight_matrix])
                # W_min_cl = torch.clamp(W_min,None,0)
                # W_max = torch.stack([max(row) for row in weight_matrix])
                # W_max_cl = torch.clamp(W_max,0,None)
                 ### slower option to find size is to do a forward pass
                l_1_box = F.conv2d(prev_ub, w_kp1, b_kp1, layer.stride, layer.padding, layer.dilation, layer.groups)
                l_kp1 = torch.zeros_like(l_1_box)
                u_kp1 = torch.zeros_like(l_1_box)
                ###
                ### faster option is to get the output shape. works perfectly. switch on later
                # h_out_0 = math.floor((l_0.shape[2] + 2*layer.padding[0] - layer.dilation[0]*(w_1.shape[2]-1) -1) / layer.stride[0] + 1)
                # h_out_1 = math.floor((l_0.shape[3] + 2*layer.padding[1] - layer.dilation[1]*(w_1.shape[3]-1) -1) / layer.stride[1] + 1)
                # l_kp1 = torch.ones(l_0.shape[0] ,w_1.shape[0] ,h_out_0 ,h_out_1).to(b_1.device)
                # u_kp1 = torch.ones(l_0.shape[0] ,w_1.shape[0] ,h_out_0 ,h_out_1).to(b_1.device)
                ###
                for j in range(l_kp1.shape[0]):
                    for i in range(l_kp1.shape[1]):
                        l_kp1[j,i,:,:] = W_min_cl[i] + b_kp1[i]
                    for i in range(u_kp1.shape[1]):
                        u_kp1[j,i,:,:] = W_max_cl[i] + b_kp1[i]
                ##

            output_padding = compute_output_padding(prev_ub, layer) #can comment this to recover old behaviour
            obj_layer = ConvOp(w_kp1, b_kp1,
                               layer.stride, layer.padding,
                               layer.dilation, layer.groups, output_padding)

        else:
            #################################
            ## PREVIOUS LAYER CONDITIONING ##
            #################################
            # Only needs conditioning the weights
            # 1. has alpha scaling
            # 2. has lambda scaling
            w_kp1 = (w_kp1) * init_cut_coeffs[-1]
            # if lmbd_prev is None:
            #     w_kp1 = (w_kp1) * init_cut_coeffs[-1]
            # else:
            #     w_kp1 = (w_kp1*torch.reciprocal(lmbd_prev)) * init_cut_coeffs[-1]
            #################################
            #################################

            ###############
            # STEP-2- construct LinearOp
            ###############
            obj_layer = LinearOp(w_kp1, b_kp1)
            ###############

            if conditioning:
                ###############
                # Getting the lambda
                ###############
                if seed!=0:
                    a = np.random.random(b_kp1.shape[0])
                    a /= a.sum()
                    lmbd=torch.from_numpy(a).float().to(w_kp1.device)
                else:### default version
                    lmbd=torch.ones(b_kp1.shape[0]).float().to(w_kp1.device)
                ###############

                init_cut_coeff = obj_layer.simplex_conditioning(lmbd, conditioning=True)

            # STEP-4- calculating pre-activation bounds of conditioned layer
            if prop_simp_bounds:
                weight_matrix= obj_layer.weights.view(obj_layer.weights.shape[0], -1)
                W_min, _ = torch.min(weight_matrix, 1)
                W_min = torch.clamp(W_min, None, 0)
                W_max, _ = torch.max(weight_matrix, 1)
                W_max = torch.clamp(W_max, 0, None)

                l_kp1 = W_min + obj_layer.bias
                u_kp1 = W_max + obj_layer.bias
           

        if isinstance(obj_layer, LinearOp) and (prev_ub.dim() > 2):
            # This is the first LinearOp,
            # We need to include the flattening
            obj_layer.flatten_from(prev_ub.shape[1:])

        if prop_simp_bounds:
            ini_lbs, ini_ubs = obj_layer.interval_forward(torch.clamp(prev_lb, 0, None),
                                                        torch.clamp(prev_ub, 0, None))
            batch_size = ini_lbs.shape[0]
            out_shape = ini_lbs.shape[1:]#this is the actual output shape
            l_kp1 = l_kp1.view(batch_size, *out_shape)
            u_kp1 = u_kp1.view(batch_size, *out_shape)

            assert (u_kp1 - l_kp1).min() >= 0, "Incompatible bounds"


        return l_kp1, u_kp1, init_cut_coeff, obj_layer, obj_layer_orig, lmbd

    def compute_lower_bound(self, node=(-1, None), upper_bound=False, counterexample_verification=False):
        '''
        Compute a lower bound of the function for the given node

        node: (optional) Index (as a tuple) in the list of gurobi variables of the node to optimize
              First index is the layer, second index is the neuron.
              For the second index, None is a special value that indicates to optimize all of them,
              both upper and lower bounds.
        upper_bound: (optional) Compute an upper bound instead of a lower bound
        '''
        additional_coeffs = {}
        current_lbs = self.lower_bounds[node[0]]
        if current_lbs.dim() == 0:
            current_lbs = current_lbs.unsqueeze(0)
        node_layer_shape = current_lbs.shape[1:]
        batch_size = current_lbs.shape[0]
        self.opt_time_per_layer = []

        lay_to_opt = len(self.lower_bounds) + node[0] if node[0] < 0 else node[0]
        is_batch = (node[1] is None)
        # with batchification, we need to optimize over all layers in any case, as otherwise the tensors of
        # different sizes should be kept as a list (slow)
        # Optimize all the bounds
        nb_out = prod(node_layer_shape)

        start_opt_time = time.time()
        # if the resulting batch size from parallelizing over the output neurons boundings is too large, we need
        # to divide into sub-batches
        neuron_batch_size = nb_out * 2 if is_batch else 1
        c_batch_size = int(math.floor(self.max_batch / batch_size))
        n_batches = int(math.ceil(neuron_batch_size / float(c_batch_size)))
        print(f"----------------> {c_batch_size} * {n_batches}; total {neuron_batch_size}*{batch_size}")
        bound = None
        for sub_batch_idx in range(n_batches):
            # compute intermediate bounds on sub-batch
            start_batch_index = sub_batch_idx * c_batch_size
            end_batch_index = min((sub_batch_idx + 1) * c_batch_size, neuron_batch_size)

            slice_coeffs = create_final_coeffs_slice(
                start_batch_index, end_batch_index, batch_size, nb_out, current_lbs, node_layer_shape, node,
                upper_bound=upper_bound)
            additional_coeffs[lay_to_opt] = slice_coeffs

            c_bound = self.optimize(self.weights, additional_coeffs, self.lower_bounds, self.upper_bounds)
            bound = c_bound if bound is None else torch.cat([bound, c_bound], 1)
        end_opt_time = time.time()

        self.opt_time_per_layer.append(end_opt_time - start_opt_time)
        if is_batch:
            opted_ubs = -bound[:, :nb_out]
            opted_lbs = bound[:, nb_out:]
            ubs = opted_ubs.view(batch_size, *node_layer_shape)
            lbs = opted_lbs.view(batch_size, *node_layer_shape)

            # this is a bit of a hack for use in the context of standard counter-example verification problems
            if counterexample_verification:
                # if the bounds are not actual lower/upper bounds, then the subdomain for counter-example verification
                # is infeasible
                if lay_to_opt == len(self.weights):
                    # signal infeasible domains with infinity at the last layer bounds
                    lbs = torch.where(lbs > ubs, float('inf') * torch.ones_like(lbs), lbs)
                    ubs = torch.where(lbs > ubs, float('inf') * torch.ones_like(ubs), ubs)
                # otherwise, ignore the problem: it will be caught by the last layer
                return lbs, ubs

            assert (ubs - lbs).min() >= 0, "Incompatible bounds"

            return lbs, ubs
        else:
            if upper_bound:
                bound = -bound
            return bound

    def define_linear_approximation(self, input_domain, emb_layer=False, no_conv=False, override_numerical_errors=False):
        '''
        this function computes intermediate bounds and stores them into self.lower_bounds and self.upper_bounds.
        It also stores the network weights into self.weights.
        Now this function will compute these bounds and then condition the layers such that the input constraints are simplex.

        no_conv is an option to operate only on linear layers, by transforming all
        the convolutional layers into equivalent linear layers.
        lower_bounds [input_bounds,1st_layer_output,2nd_layeroutput ....]
        '''

        # store which relus are ambiguous. 1=passing, 0=blocking, -1=ambiguous. Shape: dom_batch_size x layer_width
        self.relu_mask = []
        self.no_conv = no_conv
        # Setup the bounds on the inputs
        self.input_domain = input_domain
        self.opt_time_per_layer = []
        X, eps = input_domain

        next_is_linear = True
        conditioning = True
        prop_simp_bounds = False

        ################################
        ## This checks for the scenario when the first layer is a flatten layer
        ################################
        first_layer_flatten = False
        if not (isinstance(self.layers[0], nn.Conv2d) or isinstance(self.layers[0], nn.Linear)):
            first_layer_flatten = True
            self.layers_copy = copy.deepcopy(self.layers)
            self.layers = self.layers[1:]
        ################################
        ################################

        for lay_idx, layer in enumerate(self.layers):
            print(lay_idx, layer)
            if lay_idx == len(self.layers)-1:
                print('not conditioning')
                conditioning=False

            if lay_idx == 0:
                assert next_is_linear
                next_is_linear = False
                layer_opt_start_time = time.time()
                if not emb_layer:
                    l_1, u_1, init_cut_coeff, cond_first_linear, lmbd = self.get_preact_bounds_first_layer(X, eps, layer, no_conv, conditioning = conditioning, seed=self.seed)
                else:
                    l_1, u_1, init_cut_coeff, cond_first_linear, lmbd = self.get_preact_bounds_first_emb_layer(X, eps, layer, no_conv, conditioning = conditioning, seed=self.seed)
                print('layer weight shape', cond_first_linear.weights.shape)
                print('init_cut_coeff', init_cut_coeff)
                layer_opt_end_time = time.time()
                time_used = layer_opt_end_time - layer_opt_start_time
                print(f"Time used for layer {lay_idx}: {time_used}")
                if init_cut_coeff is not None:
                    self.init_cut_coeffs.append(init_cut_coeff)

                if no_conv:
                    # when linearizing conv layers, we need to keep track of the original shape of the bounds
                    self.original_shape_lbs = [-torch.ones_like(X), l_1]
                    self.original_shape_ubs = [torch.ones_like(X), u_1]
                    X = X.view(X.shape[0], -1)
                    X = X.view(X.shape[0], -1)
                    X = X.view(X.shape[0], -1)
                    X = X.view(X.shape[0], -1)
                
                if not emb_layer:
                    self.lower_bounds = [-torch.ones(*X.shape[:-1], 2*X.shape[-1]).cuda(X.get_device()), l_1]
                    self.upper_bounds = [torch.ones(*X.shape[:-1], 2*X.shape[-1]).cuda(X.get_device()), u_1]
                else:
                    self.lower_bounds = [-torch.ones(1, X[1].shape[0]).cuda(X[1].get_device()), l_1]
                    self.upper_bounds = [torch.ones(1, X[1].shape[0]).cuda(X[1].get_device()), u_1]
                
                weights = [cond_first_linear]
                self.relu_mask.append(get_relu_mask(l_1, u_1))


            elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                assert next_is_linear
                next_is_linear = False

                orig_shape_prev_ub = self.original_shape_ubs[-1] if no_conv else None
                layer_opt_start_time = time.time()
                # print('layer input shape', self.upper_bounds[-1].shape)
                l_kp1, u_kp1, init_cut_coeff, obj_layer, obj_layer_orig, lmbd = self.build_simp_layer(self.upper_bounds[-1], self.lower_bounds[-1], layer, self.init_cut_coeffs, lmbd, no_conv, orig_shape_prev_ub=orig_shape_prev_ub, conditioning = conditioning, seed=self.seed, prop_simp_bounds=prop_simp_bounds)
                print('Conditioning time: ', time.time()-layer_opt_start_time)
                # print('layer weight shape', obj_layer.weights.shape)
                # print('init_cut_coeff', init_cut_coeff)

                weights.append(obj_layer)

                layer_opt_start_time = time.time()
                l_kp1_lirpa , u_kp1_lirpa = self.solve_problem(weights, self.lower_bounds, self.upper_bounds, override_numerical_errors=override_numerical_errors)

                if prop_simp_bounds:
                    l_kp1=torch.max(l_kp1, l_kp1_lirpa)
                    u_kp1=torch.min(u_kp1, u_kp1_lirpa)
                else:
                    l_kp1=l_kp1_lirpa
                    u_kp1=u_kp1_lirpa


                assert (u_kp1 - l_kp1).min() >= 0, "Incompatible bounds"
                layer_opt_end_time = time.time()
                time_used = layer_opt_end_time - layer_opt_start_time
                print(f"Time used for layer {lay_idx}: {time_used}")
                self.opt_time_per_layer.append(layer_opt_end_time - layer_opt_start_time)

                if lay_idx != len(self.layers)-1:
                    self.init_cut_coeffs.append(init_cut_coeff)

                if no_conv:
                    if isinstance(layer, nn.Conv2d):
                        self.original_shape_lbs.append(
                            l_kp1.view(obj_layer_orig.get_output_shape(self.original_shape_lbs[-1].unsqueeze(1).shape)).
                            squeeze(1)
                        )
                        self.original_shape_ubs.append(
                            u_kp1.view(obj_layer_orig.get_output_shape(self.original_shape_ubs[-1].unsqueeze(1).shape)).
                            squeeze(1)
                        )
                    else:
                        self.original_shape_lbs.append(l_kp1)
                        self.original_shape_ubs.append(u_kp1)
                self.lower_bounds.append(l_kp1)
                self.upper_bounds.append(u_kp1)
                if lay_idx < (len(self.layers)-1):
                    # the relu mask doesn't make sense on the final layer
                    self.relu_mask.append(get_relu_mask(l_kp1, u_kp1))
            elif isinstance(layer, nn.ReLU):
                assert not next_is_linear
                next_is_linear = True
            else:
                pass

        # if first_layer_flatten:
        #     self.layers = self.layers_copy
            
        self.weights = weights


    def build_model_using_intermediate_net(self, domain, intermediate_bounds, intermediate_net, no_conv=False):
        """
        Build the model from the provided intermediate bounds.
        If no_conv is true, convolutional layers are treated as their equivalent linear layers. In that case,
        provided intermediate bounds should retain the convolutional structure.
        """
        self.no_conv = no_conv
        self.input_domain = domain
        ref_lbs, ref_ubs = copy.deepcopy(intermediate_bounds)
        int_weights = copy.deepcopy(intermediate_net.weights)

        # Bounds on the inputs
        X, eps = domain

        ################################
        ## This checks for the scenario when the first layer is a flatten layer
        ################################
        first_layer_flatten = False
        if not (isinstance(self.layers[0], nn.Conv2d) or isinstance(self.layers[0], nn.Linear)):
            first_layer_flatten = True
            self.layers_copy = copy.deepcopy(self.layers)
            self.layers = self.layers[1:]
        ################################
        ################################

        # Add the first layer, appropriately rescaled.
        self.weights = [int_weights[0]]
        # Change the lower bounds and upper bounds corresponding to the inputs
        if not no_conv:
            self.lower_bounds = ref_lbs.copy()
            self.upper_bounds = ref_ubs.copy()
            # self.lower_bounds[0] = -torch.ones_like(X)
            # self.upper_bounds[0] = torch.ones_like(X)

        next_is_linear = False
        lay_idx = 1
        for layer in self.layers[1:]:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                assert next_is_linear
                next_is_linear = False

                self.weights.append(int_weights[lay_idx])
                lay_idx += 1
            elif isinstance(layer, nn.ReLU):
                assert not next_is_linear
                next_is_linear = True
            else:
                pass

    def solve_problem(self, weights, lower_bounds, upper_bounds, override_numerical_errors=False):
        '''
        Compute bounds on the last layer of the problem. (it will compute 2*number of output neurons bounds.)
        With batchification, we need to optimize over all layers in any case, as otherwise the tensors of different
         sizes should be kept as a list (slow)
        '''
        ini_lbs, ini_ubs = weights[-1].interval_forward(torch.clamp(lower_bounds[-1], 0, None),
                                                        torch.clamp(upper_bounds[-1], 0, None))
        out_shape = ini_lbs.shape[1:]#this is the actual output shape
        nb_out = prod(out_shape)#number of output neurons of that layer.
        batch_size = ini_lbs.shape[0]

        # if the resulting batch size from parallelizing over the output neurons boundings is too large, we need
        # to divide into sub-batches
        neuron_batch_size = nb_out * 2
        c_batch_size = int(math.floor(self.max_batch / batch_size))
        n_batches = int(math.ceil(neuron_batch_size / float(c_batch_size)))
        bound = None
        for sub_batch_idx in range(n_batches):
            # compute intermediate bounds on sub-batch
            start_batch_index = sub_batch_idx * c_batch_size
            end_batch_index = min((sub_batch_idx + 1) * c_batch_size, neuron_batch_size)
            subbatch_coeffs = create_final_coeffs_slice(
                start_batch_index, end_batch_index, batch_size, nb_out, ini_lbs, out_shape)
            #subbatch_coeffs is of size (batch_size*output size), each rows stores 1 or -1 for which
            #output neuron this batch corresponds to.
            additional_coeffs = {len(lower_bounds): subbatch_coeffs}
            c_bound = self.optimize(weights, additional_coeffs, lower_bounds, upper_bounds)
            bound = c_bound if bound is None else torch.cat([bound, c_bound], 1)

        ubs = -bound[:, :nb_out]
        lbs = bound[:, nb_out:]
        lbs = lbs.view(batch_size, *out_shape)
        ubs = ubs.view(batch_size, *out_shape)

        if not override_numerical_errors:
            assert (ubs - lbs).min() >= 0, "Incompatible bounds"
        else:
            ubs = torch.where((ubs - lbs <= 0) & (ubs - lbs >= -1e-5), lbs + 1e-5, ubs)
            assert (ubs - lbs).min() >= 0, "Incompatible bounds"

        return lbs, ubs

    def init_optimizer(self, method_args):
        return self.init_optimize, None

    def best_naive_kw_optimizer(self, method_args):
        # best bounds out of kw and naive interval propagation
        kw_fun, kw_logger = self.optimizers['init'](None)
        naive_fun, naive_logger = self.optimizers['init'](None)

        def optimize(*args, **kwargs):
            self.set_decomposition('pairs', 'KW')
            bounds_kw = kw_fun(*args, **kwargs)

            self.set_decomposition('pairs', 'naive')
            bounds_naive = naive_fun(*args, **kwargs)
            bounds = torch.max(bounds_kw, bounds_naive)

            return bounds

        return optimize, [kw_logger, naive_logger]

    def best_naive_simplex_optimizer(self, method_args):
        # best bounds out of kw and naive interval propagation
        kw_fun, kw_logger = self.optimizers['init'](None)
        naive_fun, naive_logger = self.optimizers['init'](None)

        def optimize(*args, **kwargs):
            # self.set_decomposition('pairs', 'KW')
            # bounds_kw = kw_fun(*args, **kwargs)

            # self.set_decomposition('pairs', 'naive')
            # bounds_naive = naive_fun(*args, **kwargs)
            # bounds = torch.max(bounds_kw, bounds_naive)

            # ### optimal for cifar_sgd
            # opt_args = {
            #     'nb_iter': 20,
            #     'lower_initial_step_size': 0.00001,
            #     'lower_final_step_size': 10,
            #     'upper_initial_step_size': 1e2,
            #     'upper_final_step_size': 1e3,
            #     'betas': (0.9, 0.999)
            # }

            # optimal for cifar_madry
            # opt_args = {
            #     'nb_iter': 20,
            #     'lower_initial_step_size': 10,
            #     'lower_final_step_size': 0.01,
            #     'upper_initial_step_size': 1e2,
            #     'upper_final_step_size': 1e3,
            #     'betas': (0.9, 0.999)
            # }
            
            # # optimal for cifar_l1 nets (havent tuned much)
            opt_args = {
                'nb_iter': 20,
                'lower_initial_step_size': 0.00001,
                'lower_final_step_size': 10,
                'upper_initial_step_size': 1e2,
                'upper_final_step_size': 1e3,
                'betas': (0.9, 0.999)
            }


            auto_lirpa_object = AutoLirpa(*args, **kwargs)
            auto_lirpa_object.crown_initialization(*args, **kwargs)

            # for crown
            # bounds_auto_lirpa = auto_lirpa_object.get_bound_lirpa_backward(*args, **kwargs)

            # for auto_lirpa
            bounds_auto_lirpa = auto_lirpa_object.auto_lirpa_optimizer(*args, **kwargs, dp=False, opt_args=opt_args)
            # bounds = torch.max(bounds, bounds_auto_lirpa)

            return bounds_auto_lirpa

        return optimize, [kw_logger, naive_logger]

    def best_naive_dp_optimizer(self, method_args):
        # best bounds out of kw and naive interval propagation
        kw_fun, kw_logger = self.optimizers['init'](None)
        naive_fun, naive_logger = self.optimizers['init'](None)

        def optimize(*args, **kwargs):
            # self.set_decomposition('pairs', 'KW')
            # bounds_kw = kw_fun(*args, **kwargs)

            # self.set_decomposition('pairs', 'naive')
            # bounds_naive = naive_fun(*args, **kwargs)
            # bounds = torch.max(bounds_kw, bounds_naive)

            # for cifar
            # opt_args = {
            #     'nb_iter': 20,
            #     'lower_initial_step_size': 0.0001,
            #     'lower_final_step_size': 1,
            #     'upper_initial_step_size': 1e2,
            #     'upper_final_step_size': 1e3,
            #     'betas': (0.9, 0.999)
            # }

            ## for food101 old__1
            # opt_args = {
            #         'nb_iter': 20,
            #         'lower_initial_step_size': 10,
            #         'lower_final_step_size': 0.01,
            #         'upper_initial_step_size': 0.001,
            #         'upper_final_step_size': 1e3,
            #         'betas': (0.9, 0.999)
            #     }

            # ## for food101 newer
            opt_args = {
                    'nb_iter': 20,
                    'lower_initial_step_size': 0.00001,
                    'lower_final_step_size': 1,
                    'upper_initial_step_size': 1,
                    'upper_final_step_size': 100,
                    'betas': (0.9, 0.999)
                }

            auto_lirpa_object = AutoLirpa(*args, **kwargs)
            auto_lirpa_object.crown_initialization(*args, **kwargs)

            bounds_auto_lirpa = auto_lirpa_object.auto_lirpa_optimizer(*args, **kwargs, dp=True, opt_args=opt_args)
            # bounds = torch.max(bounds, bounds_auto_lirpa)

            return bounds_auto_lirpa

        return optimize, [kw_logger, naive_logger]

    def opt_dp_optimizer(self, method_args):
        # best bounds out of kw and naive interval propagation
        kw_fun, kw_logger = self.optimizers['init'](None)
        naive_fun, naive_logger = self.optimizers['init'](None)

        def optimize(*args, **kwargs):
            # self.set_decomposition('pairs', 'KW')
            # bounds_kw = kw_fun(*args, **kwargs)

            # self.set_decomposition('pairs', 'naive')
            # bounds_naive = naive_fun(*args, **kwargs)
            # bounds = torch.max(bounds_kw, bounds_naive)

            bounds_auto_lirpa = autolirpa_opt_dp(*args, **kwargs)
            # bounds = torch.max(bounds, bounds_auto_lirpa)

            return bounds_auto_lirpa

        return optimize, [kw_logger, naive_logger]

    def auto_lirpa_optimizer(self, weights, final_coeffs, lower_bounds, upper_bounds):
        '''
        This is simplex lirpa optimization.
        '''

        # # optimal for cifar_sgd, cifar_l1 and mnist nets 
        opt_args = {
            'nb_iter': self.params["nb_outer_iter"],
            'lower_initial_step_size': 0.00001,
            'lower_final_step_size': 10,
            'upper_initial_step_size': 1e2,
            'upper_final_step_size': 1e3,
            'betas': (0.9, 0.999)
        }


        auto_lirpa_object = AutoLirpa(weights, final_coeffs, lower_bounds, upper_bounds)
        auto_lirpa_object.crown_initialization(weights, final_coeffs, lower_bounds, upper_bounds)

        bounds_auto_lirpa = auto_lirpa_object.auto_lirpa_optimizer(weights, final_coeffs, lower_bounds, upper_bounds, dp=self.dp, opt_args=opt_args)
            
        return bounds_auto_lirpa


    def init_optimize(self, weights, final_coeffs, lower_bounds, upper_bounds):
        '''
        Simply use the values that it has been initialized to.
        '''
        dual_vars = self.decomposition.initial_dual_solution(weights, final_coeffs,
                                                             lower_bounds, upper_bounds)
        #corresponding to each output neuron, the dual_vars is of size of one set of input neurons
        matching_primal_vars = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                   lower_bounds, upper_bounds,
                                                                   dual_vars)
        if self.store_bounds_primal:
            self.bounds_primal = matching_primal_vars
        bound = self.decomposition.compute_objective(dual_vars, matching_primal_vars, final_coeffs)
        return bound

    def min_softmax_prob(self):
        final_layer_weights = self.weights[-1].weights
        final_layer_bias = self.weights[-1].bias

        target_class_weights = final_layer_weights[self.tgt, :]
        target_class_bias = final_layer_bias[self.tgt]

        print(final_layer_bias)
        print(target_class_weights)
        overall_max = -10000
        for l in range(final_layer_weights.shape[0]):
            if l==self.tgt:
                continue
            current_class_weights = final_layer_weights[l, :]
            current_class_bias = final_layer_bias[l]

            weights_diff = current_class_weights - target_class_weights
            bias_diff = current_class_bias - target_class_bias

            pos_w1 = torch.clamp(weights_diff, 0, None)
            neg_w1 = torch.clamp(weights_diff, None, 0)

            current_l_bound = bias_diff + pos_w1@(self.upper_bounds[-2].squeeze(0)) + neg_w1@(self.lower_bounds[-2].squeeze(0))
            if current_l_bound>overall_max:
                overall_max = current_l_bound
        print(pos_w1)
        print(self.upper_bounds[-2].squeeze(0))
        print(-overall_max, - math.log(final_layer_weights.shape[0]))
        return math.exp(-overall_max - math.log(final_layer_weights.shape[0]))


    def dump_instance(self, path_to_file):
        to_save = {
            'layers': self.layers,
            'lbs': self.lower_bounds,
            'ubs': self.upper_bounds,
            'input_domain': self.input_domain
        }
        torch.save(to_save, path_to_file)

    @classmethod
    def load_instance(cls, path_to_file):
        saved = torch.load(path_to_file)

        intermediate_bounds = (saved['lbs'], saved['ubs'])

        inst = cls(saved['layers'])
        inst.build_model_using_bounds(saved['input_domain'],
                                      intermediate_bounds)

        return inst

    def get_lower_bound_network_input(self):
        """
        Return the input of the network that was used in the last bounds computation.
        Converts back from the conditioned input domain to the original one.
        Assumes that the last layer is a single neuron.
        """
        assert self.store_bounds_primal
        assert self.bounds_primal.z0.shape[1] in [1, 2], "the last layer must have a single neuron"
        l_0 = self.input_domain.select(-1, 0)
        u_0 = self.input_domain.select(-1, 1)
        net_input = (1/2) * (u_0 - l_0) * self.bounds_primal.z0.select(1, self.bounds_primal.z0.shape[1]-1) +\
                    (1/2) * (u_0 + l_0)
        return net_input

    def update_relu_mask(self):
        # update all the relu masks of the given network
        for x_idx in range(1, len(self.lower_bounds)-1):
            self.relu_mask[x_idx-1] = get_relu_mask(
                self.lower_bounds[x_idx], self.upper_bounds[x_idx])

    def initialize_from(self, external_init):
        # setter to have the optimizer initialized from an external list of dual variables (as list of tensors)
        self.set_decomposition('pairs', 'external', ext_init=external_init)

    def bigm_subgradient_optimizer(self, weights, additional_coeffs, lower_bounds, upper_bounds):
        # ADAM subgradient ascent for a specific big-M solver which operates directly in the dual space

        # hard-code default parameters, which are overridden by self.params
        opt_args = {
            'nb_outer_iter': 100,
            'initial_step_size': 1e-3,
            'final_step_size': 1e-6,
            'betas': (0.9, 0.999)
        }
        if self.bigm_init:
            opt_args.update(self.params['init_params'])
        else:
            opt_args.update(self.params)

        # TODO: at the moment we don't support optimization with objective
        # function on something else than the last layer (although it could be
        # adapted) similarly to the way it was done in the proxlp_solver. For
        # now, just assert that we're in the expected case.
        assert len(additional_coeffs) == 1
        assert len(weights) in additional_coeffs

        # if self.store_bounds_progress >= 0 and self.bigm_only:
        if self.store_bounds_progress >= 0:
            self.logger.start_timing()

        device = lower_bounds[-1].device
        # Build the clamped bounds
        clbs = [lower_bounds[0]] + [torch.clamp(bound, 0, None) for bound in lower_bounds[1:]]  # 0 to n-1
        cubs = [upper_bounds[0]] + [torch.clamp(bound, 0, None) for bound in upper_bounds[1:]]  # 0 to n-1

        pinit = False
        dual_vars = bigm_optimization.DualVars.naive_initialization(weights, additional_coeffs, device,
                                                                        lower_bounds[0].shape[1:])

        # Adam-related quantities.
        adam_stats = bigm_optimization.DualADAMStats(dual_vars.beta_0, beta1=opt_args['betas'][0], beta2=opt_args['betas'][1])
        init_step_size = opt_args['initial_step_size'] if not pinit else opt_args['initial_step_size_pinit']
        final_step_size = opt_args['final_step_size']

        add_coeff = next(iter(additional_coeffs.values()))
        batch_size = add_coeff.shape[:2]
        best_bound = -float("inf") * torch.ones(batch_size, device=device, dtype=self.precision)


        if self.store_bounds_progress >=0:
            start_logging_time = time.time()
            bound = bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
            # torch.max(best_bound, bound, out=best_bound)
            logging_time = time.time() - start_logging_time
            self.logger.add_point(len(weights), bound.clone(), logging_time=logging_time)

        if self.debug:
            obj_val = bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
            print(f"Average bound (and objective, they concide) at naive initialisation: {obj_val.mean().item()}")
            torch.max(best_bound, obj_val, out=best_bound)
            if self.view_tensorboard:
                self.writer.add_scalar('Average best bound', -best_bound.mean().item(), 0)
                self.writer.add_scalar('Average bound', -obj_val.mean().item(), 0)

        n_outer_iters = opt_args["nb_outer_iter"]
        for outer_it in itertools.count():
            if outer_it >= n_outer_iters:
                break

            if self.debug:
                obj_val = bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)

            dual_vars_subg = bigm_optimization.compute_dual_subgradient(
                weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)

            step_size = init_step_size + ((outer_it + 1) / n_outer_iters) * (final_step_size - init_step_size)

            # normal subgradient ascent
            # dual_vars.projected_linear_combination(
            #     step_size, dual_vars_subg, weights)

            # do adam for subgradient ascent
            adam_stats.update_moments_take_projected_step(
                weights, step_size, outer_it, dual_vars, dual_vars_subg)

            dual_vars.update_f_g(lower_bounds, upper_bounds)

            if self.debug:
                # This is the value "at convergence" of the dual problem
                obj_val = bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
                print(f"Average obj at the end of adam iteration {outer_it}: {obj_val.mean().item()}")
                torch.max(best_bound, obj_val, out=best_bound)
                print(f"{outer_it} Average best bound: {best_bound.mean().item()}")
                if self.view_tensorboard:
                    self.writer.add_scalar('Average best bound', -best_bound.mean().item(), outer_it + 1)
                    self.writer.add_scalar('Average bound', -obj_val.mean().item(), outer_it + 1)

            if self.store_bounds_progress >= 0 and len(weights) == self.store_bounds_progress and self.bigm_only:
                if outer_it % 1 == 0:
                    start_logging_time = time.time()
                    bound = bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
                    torch.max(best_bound, bound, out=best_bound)
                    logging_time = time.time() - start_logging_time
                    self.logger.add_point(len(weights), bound.clone(), logging_time=logging_time)

        # store the dual vars and primal vars for possible future usage
        self.bigm_dual_vars = dual_vars
        self.bigm_primal_vars = None
        self.bigm_adam_stats = adam_stats

        if self.bigm_only:
            self.children_init = bigm_optimization.BigMPInit(dual_vars)
            if self.store_bounds_primal:
                # Compute last matching primal (could make it a function...)
                nb_relu_layers = len(dual_vars.beta_0)
                xkm1, _ = bigm_optimization.layer_primal_linear_minimization(0, dual_vars.fs[0], None, clbs[0], cubs[0])
                zt = []
                xt = [xkm1]
                for lay_idx in range(1, nb_relu_layers):
                    # solve the inner problems.
                    xk, zk = bigm_optimization.layer_primal_linear_minimization(
                        lay_idx, dual_vars.fs[lay_idx], dual_vars.gs[lay_idx - 1], clbs[lay_idx], cubs[lay_idx])
                    xt.append(xk)
                    zt.append(zk)
                self.bounds_primal = anderson_optimization.PrimalVars(xt, zt)

        bound = bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
        return bound

    def simplex_optimizer(self, weights, additional_coeffs, lower_bounds, upper_bounds):
        # ADAM subgradient ascent for a specific big-M solver which operates directly in the dual space

        # hard-code default parameters, which are overridden by self.params
        opt_args = {
            "nb_inner_iter": 100,
            "nb_iter": 100,
            "alpha_M": 1e-3,
            "beta_M": 1e-3,
            "bigm_algorithm": "adam",
            ###
            'cut_frequency': 50,
            'max_cuts': 10,
            'cut_add': 2,
            ###
            'betas': (0.9, 0.999),
            'initial_step_size': 1e-3,
            'final_step_size': 1e-6,
            "init_params": {
                'nb_outer_iter': 100,
                'initial_step_size': 1e-2,
                'final_step_size': 1e-4,
                'betas': (0.9, 0.999),
                'M_factor': 1.0
            },
            'nb_outer_iter': 100,
            'bigm': 'init',
        }
        n_outer_iters = opt_args["nb_outer_iter"]
        print(n_outer_iters)
        if self.cut_only:
            opt_args.update(self.params)

        # TODO: at the moment we don't support optimization with objective
        # function on something else than the last layer (although it could be
        # adapted) similarly to the way it was done in the proxlp_solver. For
        # now, just assert that we're in the expected case.
        assert len(additional_coeffs) == 1
        assert len(weights) in additional_coeffs

        if self.store_bounds_progress >= 0 and self.bigm_only:
            self.logger.start_timing()

        device = lower_bounds[-1].device
        # Build the clamped bounds
        clbs = [lower_bounds[0]] + [torch.clamp(bound, 0, None) for bound in lower_bounds[1:]]  # 0 to n-1
        cubs = [upper_bounds[0]] + [torch.clamp(bound, 0, None) for bound in upper_bounds[1:]]  # 0 to n-1

        pinit = False
        dual_vars = bigm_optimization.DualVars.naive_initialization(weights, additional_coeffs, device,
                                                                        lower_bounds[0].shape[1:])

        # Adam-related quantities.
        if self.bigm_init:
            # Initialize alpha/beta with the output of the chosen big-m solver.
            bounds = self.bigm_subgradient_optimizer(weights, additional_coeffs, lower_bounds, upper_bounds)
            print(f"Average bounds after init with Bigm adam: {bounds.mean().item()}")

            dual_vars = simplex_optimization.SimplexDualVars.bigm_initialization(
                self.bigm_dual_vars, weights, additional_coeffs, device, clbs, cubs, lower_bounds,
                upper_bounds, opt_args)

            ## Adam-related quantities.
            adam_stats = simplex_optimization.DualADAMStats(dual_vars.beta_0, beta1=opt_args['betas'][0],
                                                                 beta2=opt_args['betas'][1])
            # initializes adam stats(momentum 1 and momentum 2 for dual variables(alpha,beta_0,beta_1))
            adam_stats.bigm_adam_initialization(dual_vars.beta_0, self.bigm_adam_stats, beta1=opt_args['betas'][0],
                                                beta2=opt_args['betas'][1])
            # initializes adam stats(momentum 1 and momentum 2 for dual variables(alpha,beta_0,beta_1))
        else:
            # Initialize dual variables to all 0s, primals to mid-boxes.
            print('it is doing naive initialisation')
            dual_vars = simplex_optimization.SimplexDualVars.naive_initialization(weights, additional_coeffs, device,
                                                                                   input_size)

            ## Adam-related quantities.
            adam_stats = simplex_optimization.DualADAMStats(dual_vars.beta_0, beta1=opt_args['betas'][0],
                                                                 beta2=opt_args['betas'][1])

        init_step_size = opt_args['initial_step_size'] if not pinit else opt_args['initial_step_size_pinit']
        final_step_size = opt_args['final_step_size']

        add_coeff = next(iter(additional_coeffs.values()))
        batch_size = add_coeff.shape[:2]
        best_bound = -float("inf") * torch.ones(batch_size, device=device, dtype=self.precision)

        #### CUT adam ######
        ## Adam-related quantities.
        cut_stats = simplex_optimization.CutADAMStats(dual_vars.beta_0, 0.1, 0.01, beta1=opt_args['betas'][0], beta2=opt_args['betas'][1])
        ##########

        if self.store_bounds_progress >=0:
            start_logging_time = time.time()
            bound = simplex_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
            # torch.max(best_bound, bound, out=best_bound)
            logging_time = time.time() - start_logging_time
            self.logger.add_point(len(weights), bound.clone(), logging_time=logging_time)
            
        if self.debug:
            obj_val = simplex_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
            print(f"Average bound (and objective, they concide) at naive initialisation: {obj_val.mean().item()}")
            torch.max(best_bound, obj_val, out=best_bound)
            if self.view_tensorboard:
                self.writer.add_scalar('Average best bound', -best_bound.mean().item(), 0)
                self.writer.add_scalar('Average bound', -obj_val.mean().item(), 0)

        n_outer_iters = opt_args["nb_outer_iter"]
        for outer_it in itertools.count():
            if outer_it >= n_outer_iters:
                break

            if self.debug:
                obj_val = simplex_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)

            dual_vars_subg = simplex_optimization.compute_dual_subgradient(
                weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds, outer_it, opt_args, cut_stats)

            step_size = init_step_size + ((outer_it + 1) / n_outer_iters) * (final_step_size - init_step_size)

            # normal subgradient ascent
            # dual_vars.projected_linear_combination(
            #     step_size, dual_vars_subg, weights)

            # do adam for subgradient ascent
            adam_stats.update_moments_take_projected_step(
                weights, step_size, outer_it, dual_vars, dual_vars_subg, opt_args)

            dual_vars.update_f_g(lower_bounds, upper_bounds)

            if self.debug:
                # This is the value "at convergence" of the dual problem
                obj_val = simplex_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
                print(f"Average obj at the end of adam iteration {outer_it}: {obj_val.mean().item()}")
                torch.max(best_bound, obj_val, out=best_bound)
                print(f"{outer_it} Average best bound: {best_bound.mean().item()}")
                if self.view_tensorboard:
                    self.writer.add_scalar('Average best bound', -best_bound.mean().item(), outer_it + 1)
                    self.writer.add_scalar('Average bound', -obj_val.mean().item(), outer_it + 1)

            if self.store_bounds_progress >= 0 and len(weights) == self.store_bounds_progress:
                # if outer_it % 1 == 0:
                start_logging_time = time.time()
                bound = simplex_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
                # torch.max(best_bound, bound, out=best_bound)
                logging_time = time.time() - start_logging_time
                self.logger.add_point(len(weights), bound.clone(), logging_time=logging_time)


        bound = simplex_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
        return bound

    def dp_optimizer(self, weights, additional_coeffs, lower_bounds, upper_bounds):
        # ADAM subgradient ascent for a dp+big-m solver which operates directly in the dual space

        # hard-code default parameters, which are overridden by self.params
        opt_args = {
            "nb_inner_iter": 100,
            "nb_iter": 100,
            "alpha_M": 1e-3,
            "beta_M": 1e-3,
            "bigm_algorithm": "adam",
            ###
            'cut_frequency': 50,
            'max_cuts': 10,
            'cut_add': 2,
            ###
            'betas': (0.9, 0.999),
            'initial_step_size': 1e-3,
            'final_step_size': 1e-6,
            "init_params": {
                'nb_outer_iter': 100,
                'initial_step_size': 1e-2,
                'final_step_size': 1e-4,
                'betas': (0.9, 0.999),
                'M_factor': 1.0
            },
            'nb_outer_iter': 100,
            'bigm': 'init',
        }
        n_outer_iters = opt_args["nb_outer_iter"]
        if self.cut_only:
            opt_args.update(self.params)

        # TODO: at the moment we don't support optimization with objective
        # function on something else than the last layer (although it could be
        # adapted) similarly to the way it was done in the proxlp_solver. For
        # now, just assert that we're in the expected case.
        assert len(additional_coeffs) == 1
        assert len(weights) in additional_coeffs

        # if self.store_bounds_progress >= 0 and self.bigm_only:
        if self.store_bounds_progress >= 0:
            self.logger.start_timing()

        device = lower_bounds[-1].device
        # Build the clamped bounds
        clbs = [lower_bounds[0]] + [torch.clamp(bound, 0, None) for bound in lower_bounds[1:]]  # 0 to n-1
        cubs = [upper_bounds[0]] + [torch.clamp(bound, 0, None) for bound in upper_bounds[1:]]  # 0 to n-1

        pinit = False
        dual_vars = bigm_optimization.DualVars.naive_initialization(weights, additional_coeffs, device,
                                                                        lower_bounds[0].shape[1:])

        # Adam-related quantities.
        if 0:#self.bigm_init:
            # Initialize alpha/beta with the output of the chosen big-m solver.
            bounds = self.bigm_subgradient_optimizer(weights, additional_coeffs, lower_bounds, upper_bounds)
            print(f"Average bounds after init with Bigm adam: {bounds.mean().item()}")

            dual_vars = dp_optimization.DPDualVars.bigm_initialization(
                self.bigm_dual_vars, weights, additional_coeffs, device, clbs, cubs, lower_bounds,
                upper_bounds, opt_args)

            ## Adam-related quantities.
            adam_stats = dp_optimization.DualADAMStats(dual_vars.beta_0, beta1=opt_args['betas'][0],
                                                                 beta2=opt_args['betas'][1])
            # initializes adam stats(momentum 1 and momentum 2 for dual variables(alpha,beta_0,beta_1))
            adam_stats.bigm_adam_initialization(dual_vars.beta_0, self.bigm_adam_stats, beta1=opt_args['betas'][0], beta2=opt_args['betas'][1])
            # initializes adam stats(momentum 1 and momentum 2 for dual variables(alpha,beta_0,beta_1))
        else:
            # Initialize dual variables to all 0s, primals to mid-boxes.
            print('it is doing naive initialisation')
            dual_vars = dp_optimization.DPDualVars.naive_initialization(weights, additional_coeffs, device,
                                                                                   lower_bounds[0].shape[1:])

            ## Adam-related quantities.
            adam_stats = dp_optimization.DualADAMStats(dual_vars.beta_0, beta1=opt_args['betas'][0],
                                                                 beta2=opt_args['betas'][1])

        init_step_size = opt_args['initial_step_size'] if not pinit else opt_args['initial_step_size_pinit']
        final_step_size = opt_args['final_step_size']

        add_coeff = next(iter(additional_coeffs.values()))
        batch_size = add_coeff.shape[:2]
        best_bound = -float("inf") * torch.ones(batch_size, device=device, dtype=self.precision)

        if self.store_bounds_progress >=0:
            start_logging_time = time.time()
            bound = dp_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
            # torch.max(best_bound, bound, out=best_bound)
            logging_time = time.time() - start_logging_time
            self.logger.add_point(len(weights), bound.clone(), logging_time=logging_time)
            
        if self.debug:
            obj_val = dp_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
            print(f"Average bound (and objective, they concide) at naive initialisation: {obj_val.mean().item()}")
            torch.max(best_bound, obj_val, out=best_bound)
            if self.view_tensorboard:
                self.writer.add_scalar('Average best bound', -best_bound.mean().item(), 0)
                self.writer.add_scalar('Average bound', -obj_val.mean().item(), 0)


        n_outer_iters = opt_args["nb_outer_iter"]
        for outer_it in itertools.count():
            if outer_it >= n_outer_iters:
                break

            if self.debug:
                obj_val = dp_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)

            dual_vars_subg = dp_optimization.compute_dual_subgradient(
                weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)

            step_size = init_step_size + ((outer_it + 1) / n_outer_iters) * (final_step_size - init_step_size)

            # normal subgradient ascent
            # dual_vars.projected_linear_combination(
            #     step_size, dual_vars_subg, weights)

            # do adam for subgradient ascent
            adam_stats.update_moments_take_projected_step(
                weights, step_size, outer_it, dual_vars, dual_vars_subg)

            dual_vars.update_f_g(lower_bounds, upper_bounds)

            if self.debug:
                # This is the value "at convergence" of the dual problem
                obj_val = dp_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
                print(f"Average obj at the end of adam iteration {outer_it}: {obj_val.mean().item()}")
                torch.max(best_bound, obj_val, out=best_bound)
                print(f"{outer_it} Average best bound: {best_bound.mean().item()}")
                if self.view_tensorboard:
                    self.writer.add_scalar('Average best bound', -best_bound.mean().item(), outer_it + 1)
                    self.writer.add_scalar('Average bound', -obj_val.mean().item(), outer_it + 1)

            if self.store_bounds_progress >= 0 and len(weights) == self.store_bounds_progress:
                # if outer_it % 1 == 0:
                start_logging_time = time.time()
                bound = dp_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
                # torch.max(best_bound, bound, out=best_bound)
                logging_time = time.time() - start_logging_time
                self.logger.add_point(len(weights), bound.clone(), logging_time=logging_time)


        bound = dp_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
        return bound

        
class DecompositionPInit(ParentInit):
    """
    Parent Init class for Lagrangian Decomposition on PLANET (the prox and supergradient solvers of this file).
    """
    def __init__(self, parent_rhos):
        # parent_rhos are the rhos values (list of tensors, dual values for ByPairsDecomposition) at parent termination
        self.rhos = parent_rhos

    def to_cpu(self):
        # Move content to cpu.
        self.rhos = [crho.cpu() for crho in self.rhos]

    def to_device(self, device):
        # Move content to device "device"
        self.rhos = [crho.to(device) for crho in self.rhos]

    def as_stack(self, stack_size):
        # Repeat (copies) the content of this parent init to form a stack of size "stack_size"
        stacked_rhos = [pinits[0].unsqueeze(0).repeat(((stack_size,) + (1,) * (pinits.dim() - 1)))
                        for pinits in self.rhos]
        return DecompositionPInit(stacked_rhos)

    def set_stack_parent_entries(self, parent_solution, batch_idx):
        # Given a solution for the parent problem (at batch_idx), set the corresponding entries of the stack.
        for x_idx in range(len(self.rhos)):
            self.rhos[x_idx][2 * batch_idx] = parent_solution.rhos[x_idx].clone()
            self.rhos[x_idx][2 * batch_idx + 1] = parent_solution.rhos[x_idx].clone()

    def get_stack_entry(self, batch_idx):
        # Return the stack entry at batch_idx as a new ParentInit instance.
        return DecompositionPInit([csol[batch_idx].unsqueeze(0) for csol in self.rhos])

    def get_lb_init_only(self):
        # Get instance of this class with only entries relative to LBs.
        # this operation makes sense only in the BaB context (single output neuron), when both lb and ub where computed.
        assert self.rhos[0].shape[1] == 2
        return DecompositionPInit([c_init[:, -1].unsqueeze(1) for c_init in self.rhos])