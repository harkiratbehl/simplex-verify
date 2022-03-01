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
from plnn.simplex_solver import baseline_bigm_optimization, baseline_cut_anderson_optimization, baseline_anderson_optimization
from plnn.simplex_solver import utils
import itertools

from plnn.simplex_solver.simplex_lirpa_optimization import AutoLirpa
from plnn.simplex_solver.utils import simplex_projection_sort, l1_projection_sort
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

class Baseline_SimplexLP(LinearizedNetwork):
    '''
    The objects of this class are s.t: the input lies in l1.
    1. the first layer is conditioned s.t that the input lies in a probability simplex.
    2. The simplex is not propagated at all. This is like Dj's EMNLP paper, here the input simplex is used to get
    better box constraints on first layer output, after that box constraints are usually propagated with best(ibp , kw).

    return: so the define_linear_approximation fn returns a net whose input lies in simplex and all other intermediate layers lie in a box
    '''

    def __init__(
            self,
            layers,
            debug=False,
            params=None,
            view_tensorboard=False,
            precision=torch.float,
            store_bounds_progress=-1,
            store_bounds_primal=False,
            max_batch=20000,
            tgt=1
    ):
        """
        :param store_bounds_progress: whether to store bounds progress over time (-1=False 0=True)
        :param store_bounds_primal: whether to store the primal solution used to compute the final bounds
        :param max_batch: maximal batch size for parallel bounding çomputations over both output neurons and domains
        """
        self.optimizers = {
            'init': self.init_optimizer,
            'best_naive_kw': self.best_naive_kw_optimizer,
            'bigm_subgradient_optimizer': self.bigm_subgradient_optimizer,
            'best_naive_simplex': self.best_naive_simplex_optimizer,
            'autolirpa': self.auto_lirpa_optimizer,
            'cut_anderson_optimizer': self.cut_anderson_optimizer
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
        # self.bigm_only = self.params["bigm"] and (self.params["bigm"] == "only")
        # self.cut_init = ("cut" in self.params) and self.params["cut"] and (self.params["cut"] == "init")
        # self.cut_only = ("cut" in self.params) and self.params["cut"] and (self.params["cut"] == "only")

        self.precision = precision
        self.init_cut_coeffs = []

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
        self.optimize, self.logger = self.optimizers[method](method_args)

    @staticmethod
    def build_first_conditioned_layer(X, eps, layer, no_conv=False):
        '''
        assum: the input domain to network lies in l1 with eps around X.
        This function does 2 main things:
        1. condition the input layer such that input now lies in simplex.
        2. compute the bounds on output of first layer.
        '''
        w_1 = layer.weight
        b_1 = layer.bias

        pos_w1 = torch.clamp(w_1, 0, None)
        neg_w1 = torch.clamp(w_1, None, 0) 

        X_vector = X.view(X.shape[0], -1)
        dim = X_vector.shape[1]
        E = torch.zeros(dim, 2*dim).cuda(X.get_device())
        for row_idx in range(0,dim):
            E[row_idx, 2*row_idx] = 1
            E[row_idx, 2*row_idx+1] = -1

        if isinstance(layer, nn.Linear):
            ###############
            # STEP-1- Conditioning this layer so that input lies in simplex
            ###############
            cond_w_1 = eps * w_1 @ E
            if X_vector.dim() ==2:
                X_vector = X_vector.squeeze(0)
            cond_w_1_unsq = cond_w_1.unsqueeze(0)
            cond_b_1 = b_1 + X_vector @ w_1.t()
            cond_layer = BatchLinearOp(cond_w_1_unsq, cond_b_1)
            ###############
            ###############

            ###############
            # STEP-2- calculating pre-activation bounds.
            ###############
            W_min, _ = torch.min(cond_w_1, 1)
            W_min = torch.clamp(W_min, None, 0)
            # W_max = torch.stack([max(max(row),0) for row in cond_w_1])
            W_max, _ = torch.max(cond_w_1, 1)
            W_max = torch.clamp(W_max, 0, None)
            l_1 = W_min + cond_b_1
            u_1 = W_max + cond_b_1
            l_1 = l_1.unsqueeze(0)
            u_1 = u_1.unsqueeze(0)
            assert (u_1 - l_1).min() >= 0, "Incompatible bounds"
            ###############
            ###############

            if isinstance(cond_layer, LinearOp) and (X.dim() > 2):
                # This is the first LinearOp, so we need to include the flattening
                cond_layer.flatten_from((X.shape[1], X.shape[2], 2*X.shape[3]))
                # cond_layer.flatten_from(X.shape[1:])

        elif isinstance(layer, nn.Conv2d):
            ###############
            # STEP-1- Conditioning this layer
            ###############
            out_bias = F.conv2d(X, w_1, b_1,
                                layer.stride, layer.padding,
                                layer.dilation, layer.groups)

            output_padding = compute_output_padding(X, layer) #can comment this to recover old behaviour

            cond_layer = BatchConvOp(w_1, out_bias, b_1,
                                    layer.stride, layer.padding,
                                    layer.dilation, layer.groups, output_padding)
            cond_layer.add_prerescaling(eps*E)
            ###############
            ###############

            ###############
            # STEP-2- calculating pre-activation bounds.
            ###############
            weight_matrix= w_1.view(w_1.shape[0], -1)
            W_min = eps*torch.stack([min(min(row), min(-row)) for row in weight_matrix])
            W_max = eps*torch.stack([max(max(row),max(-row)) for row in weight_matrix])
            ### slower option to find size is to do a forward pass
            l_1_box = F.conv2d(X, w_1, b_1, layer.stride, layer.padding, layer.dilation, layer.groups)
            l_1 = torch.ones_like(l_1_box)
            u_1 = torch.ones_like(l_1_box)
            ###
            ### faster option is to get the output shape. works perfectly. switch on later
            # h_out_0 = math.floor((l_0.shape[2] + 2*layer.padding[0] - layer.dilation[0]*(w_1.shape[2]-1) -1) / layer.stride[0] + 1)
            # h_out_1 = math.floor((l_0.shape[3] + 2*layer.padding[1] - layer.dilation[1]*(w_1.shape[3]-1) -1) / layer.stride[1] + 1)
            # l_1 = torch.ones(l_0.shape[0] ,w_1.shape[0] ,h_out_0 ,h_out_1).to(b_1.device)
            # u_1 = torch.ones(l_0.shape[0] ,w_1.shape[0] ,h_out_0 ,h_out_1).to(b_1.device)
            ###
            for j in range(l_1.shape[0]):
                for i in range(l_1.shape[1]):
                    l_1[j,i,:,:] = W_min[i]
                for i in range(u_1.shape[1]):
                    u_1[j,i,:,:] = W_max[i]
            l_1 = l_1 + out_bias
            u_1 = u_1 + out_bias
            assert (u_1 - l_1).min() >= 0, "Incompatible bounds"
            ###############
            ###############

            if no_conv:
                cond_layer = cond_layer.equivalent_linear(X)
        return l_1, u_1, cond_layer

    @staticmethod
    def build_first_embedding_layer(X, eps, layer, no_conv=False):
        '''
        assum: the input domain to network lies in l1 with eps around X.
        This function does 2 main things:
        1. condition the input layer such that input now lies in simplex.
        2. compute the bounds on output of first layer.
        '''
        w_1 = layer.weight
        b_1 = layer.bias

        pos_w1 = torch.clamp(w_1, 0, None)
        neg_w1 = torch.clamp(w_1, None, 0) 

        img_emb, selected_weights, num_words = X 

        # X_vector = X.view(X.shape[0], -1)
        # dim = X_vector.shape[1]

        if isinstance(layer, nn.Linear):
            ###############
            # STEP-1- Conditioning this layer so that input lies in simplex
            ###############
            # the output for image embedding becomes the bias
            new_txt_emb = (torch.zeros(300, dtype=torch.float).cuda()).unsqueeze(0)
            concat_emb = torch.cat([new_txt_emb, img_emb], -1).squeeze(0)
            cond_b_1 = b_1 + concat_emb @ w_1.t()

            cond_w_1 = num_words*w_1[:, :300] @ selected_weights.t() # multiplied with 500 because network takes input that is not conditioned to sum to 1. but the simplex will sum to 1.
            cond_w_1_unsq = cond_w_1.unsqueeze(0)
            cond_layer = BatchLinearOp(cond_w_1_unsq, cond_b_1)
            ###############
            ###############

            ###############
            # STEP-2- calculating pre-activation bounds.
            ###############
            W_min, _ = torch.min(cond_w_1, 1)
            W_min = torch.clamp(W_min, None, 0)
            # W_max = torch.stack([max(max(row),0) for row in cond_w_1])
            W_max, _ = torch.max(cond_w_1, 1)
            W_max = torch.clamp(W_max, 0, None)
            l_1 = W_min + cond_b_1
            u_1 = W_max + cond_b_1
            l_1 = l_1.unsqueeze(0)
            u_1 = u_1.unsqueeze(0)
            assert (u_1 - l_1).min() >= 0, "Incompatible bounds"
            ###############
            ###############

        return l_1, u_1, cond_layer

    @staticmethod
    def build_obj_layer(prev_ub, layer, no_conv=False, orig_shape_prev_ub=None):
        '''
        This function return a ConvOp or LinearOp object depending on the layer type.
        '''
        w_kp1 = layer.weight
        b_kp1 = layer.bias

        obj_layer_orig = None
        
        if isinstance(layer, nn.Conv2d):

            output_padding = compute_output_padding(prev_ub, layer) #can comment this to recover old behaviour
            obj_layer = ConvOp(w_kp1, b_kp1,
                               layer.stride, layer.padding,
                               layer.dilation, layer.groups, output_padding)
            if no_conv:
                obj_layer_orig = obj_layer
                obj_layer = obj_layer.equivalent_linear(orig_shape_prev_ub)
        else:
            obj_layer = LinearOp(w_kp1, b_kp1)

        if isinstance(obj_layer, LinearOp) and (prev_ub.dim() > 2):
            # This is the first LinearOp,
            # We need to include the flattening
            obj_layer.flatten_from(prev_ub.shape[1:])

        return obj_layer, obj_layer_orig

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
            print('returning ubs and lbs')
            return lbs, ubs
        else:
            if upper_bound:
                bound = -bound
            return bound

    def define_linear_approximation(self, input_domain, emb_layer=False, no_conv=False, override_numerical_errors=False):
        '''
        this function computes intermediate bounds and stores them into self.lower_bounds and self.upper_bounds.
        It also stores the network weights into self.weights.
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
        self.emb_layer = emb_layer

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
            if lay_idx == 0:
                assert next_is_linear
                next_is_linear = False
                layer_opt_start_time = time.time()
                if not emb_layer:
                    l_1, u_1, cond_first_linear = self.build_first_conditioned_layer(
                    X, eps, layer, no_conv)
                else:
                    l_1, u_1, cond_first_linear = self.build_first_embedding_layer(
                    X, eps, layer, no_conv)

                layer_opt_end_time = time.time()
                time_used = layer_opt_end_time - layer_opt_start_time
                print(f"Time used for layer {lay_idx}: {time_used}")

                #####
                # there is a 2* in the last dimension because we converted from l1 to simplex
                # and the number of coordinates needs to be doubled because of the transformation
                if not emb_layer:
                    self.lower_bounds = [torch.zeros(*X.shape[:-1], 2*X.shape[-1]).cuda(X.get_device()), l_1]
                    self.upper_bounds = [torch.ones(*X.shape[:-1], 2*X.shape[-1]).cuda(X.get_device()), u_1]
                else:
                    self.lower_bounds = [-torch.ones(1, X[1].shape[0]).cuda(X[1].get_device()), l_1]
                    self.upper_bounds = [torch.ones(1, X[1].shape[0]).cuda(X[1].get_device()), u_1]
                weights = [cond_first_linear]
                self.relu_mask.append(get_relu_mask(l_1, u_1))

            elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                assert next_is_linear
                next_is_linear = False

                layer_opt_start_time = time.time()
                orig_shape_prev_ub = self.original_shape_ubs[-1] if no_conv else None
                obj_layer, obj_layer_orig = self.build_obj_layer(self.upper_bounds[-1], layer, no_conv, orig_shape_prev_ub=orig_shape_prev_ub)
                print('Conditioning time: ', time.time()-layer_opt_start_time)

                weights.append(obj_layer)
                layer_opt_start_time = time.time()
                l_kp1, u_kp1 = self.solve_problem(weights, self.lower_bounds, self.upper_bounds,
                                                  override_numerical_errors=override_numerical_errors)
                layer_opt_end_time = time.time()
                time_used = layer_opt_end_time - layer_opt_start_time
                print(f"[PROX] Time used for layer {lay_idx}: {time_used}")
                self.opt_time_per_layer.append(layer_opt_end_time - layer_opt_start_time)

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

    def build_model_using_bounds(self, domain, intermediate_bounds, no_conv=False):
        """
        Build the model from the provided intermediate bounds.
        If no_conv is true, convolutional layers are treated as their equivalent linear layers. In that case,
        provided intermediate bounds should retain the convolutional structure.
        """
        self.no_conv = no_conv
        self.input_domain = domain
        ref_lbs, ref_ubs = copy.deepcopy(intermediate_bounds)

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
        
        _, _, cond_first_linear = self.build_first_conditioned_layer(
            X, eps, self.layers[0], no_conv=no_conv)
        # Add the first layer, appropriately rescaled.
        self.weights = [cond_first_linear]
        # Change the lower bounds and upper bounds corresponding to the inputs
        if not no_conv:
            self.lower_bounds = ref_lbs.copy()
            self.upper_bounds = ref_ubs.copy()
        else:
            self.original_shape_lbs = ref_lbs.copy()
            self.original_shape_ubs = ref_ubs.copy()
            self.original_shape_lbs[0] = -torch.ones_like(X)
            self.original_shape_ubs[0] = torch.ones_like(X)
            self.lower_bounds = [-torch.ones_like(X.view(-1))]
            self.upper_bounds = [torch.ones_like(X.view(-1))]
            for lay_idx in range(1, len(ref_lbs)):
                self.lower_bounds.append(ref_lbs[lay_idx].view(-1).clone())
                self.upper_bounds.append(ref_ubs[lay_idx].view(-1).clone())

        next_is_linear = False
        lay_idx = 1
        for layer in self.layers[1:]:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                assert next_is_linear
                next_is_linear = False
                orig_shape_prev_ub = self.original_shape_ubs[lay_idx] if no_conv else None
                new_layer, _ = self.build_obj_layer(
                    self.upper_bounds[lay_idx], layer, no_conv=no_conv, orig_shape_prev_ub=orig_shape_prev_ub)
                self.weights.append(new_layer)
                lay_idx += 1
            elif isinstance(layer, nn.ReLU):
                assert not next_is_linear
                next_is_linear = True
            else:
                pass

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
            # self.set_decomposition('pairs', 'KW')
            # bounds_kw = kw_fun(*args, **kwargs)

            # self.set_decomposition('pairs', 'naive')
            # bounds_naive = naive_fun(*args, **kwargs)
            # bounds = torch.max(bounds_kw, bounds_naive)

            auto_lirpa_object = AutoLirpa(*args, **kwargs)
            auto_lirpa_object.crown_initialization(*args, **kwargs)

            bounds_auto_lirpa = auto_lirpa_object.auto_lirpa_optimizer(*args, **kwargs, dp=False)

            # bounds = torch.max(bounds, bounds_auto_lirpa)
            
            return bounds_auto_lirpa

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

            ### optimal for cifar_sgd
            opt_args = {
                'nb_iter': 20,
                'lower_initial_step_size': 0.00001,
                'lower_final_step_size': 1,
                'upper_initial_step_size': 1e2,
                'upper_final_step_size': 1e3,
                'betas': (0.9, 0.999)
            }

            # # optimal for cifar_madry
            # opt_args = {
            #     'nb_iter': 20,
            #     'lower_initial_step_size': 10,
            #     'lower_final_step_size': 0.01,
            #     'upper_initial_step_size': 1e2,
            #     'upper_final_step_size': 1e3,
            #     'betas': (0.9, 0.999)
            # }

            ## optimal for mmbt nets
            # opt_args = {
            #     'nb_iter': 37,
            #     'lower_initial_step_size': 0.00001,
            #     'lower_final_step_size': 1,
            #     'upper_initial_step_size': 1e2,
            #     'upper_final_step_size': 1e3,
            #     'betas': (0.9, 0.999)
            # }

            auto_lirpa_object = AutoLirpa(*args, **kwargs)
            auto_lirpa_object.crown_initialization(*args, **kwargs)

            bounds_auto_lirpa = auto_lirpa_object.auto_lirpa_optimizer(*args, **kwargs, dp=False, opt_args=opt_args)

            # bounds = torch.max(bounds, bounds_auto_lirpa)
            
            return bounds_auto_lirpa

        return optimize, [kw_logger, naive_logger]

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

    def auto_lirpa_optimizer(self, weights, final_coeffs, lower_bounds, upper_bounds):
        '''
        This is simplex lirpa optimization.
        '''

        opt_args = {
            'nb_iter': self.params["nb_outer_iter"],
            'lower_initial_step_size': 0.00001,
            'lower_final_step_size': 1,
            'upper_initial_step_size': 1e2,
            'upper_final_step_size': 1e3,
            'betas': (0.9, 0.999)
        }

        auto_lirpa_object = AutoLirpa(weights, final_coeffs, lower_bounds, upper_bounds)
        auto_lirpa_object.crown_initialization(weights, final_coeffs, lower_bounds, upper_bounds)

        bounds_auto_lirpa = auto_lirpa_object.auto_lirpa_optimizer(weights, final_coeffs, lower_bounds, upper_bounds, dp=False, opt_args=opt_args)
            
        return bounds_auto_lirpa

    def min_softmax_prob(self):
        final_layer_weights = self.weights[-1].weights
        final_layer_bias = self.weights[-1].bias

        target_class_weights = final_layer_weights[self.tgt, :]
        target_class_bias = final_layer_bias[self.tgt]

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

        # return math.exp(-overall_max - math.log(final_layer_weights.shape[0]))
        return -overall_max - math.log(final_layer_weights.shape[0])

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

    def pgd_upper_bound(self, num_steps=50, step_size=0.01):

        # cond_first_linear = self.weights[0]
        # cond_first_linear_weights = self.weights[0].weights
        # new_first_layer = nn.linear(cond_first_linear_weights.shape[1], cond_first_linear_weights.shape[0]).cuda()
        # new_first_layer.weights = cond_first_linear_weights
        # new_first_layer.bias = cond_first_linear.bias

        # new_layers = [new_first_layer] + self.layers[1:]
        # cond_net = nn.Sequential(*new_layers)

        num_idx = self.layers[-1].bias.shape[0]
        final_bounds = []
        X = self.input_domain[0]

        for param in self.net.parameters():
            param.requires_grad = True
        

        for idx in range(num_idx):
            adv_loss = lambda x: self.net(x)[0][idx]  # pylint: disable=cell-var-from-loop

            x = X.clone().requires_grad_(True).cuda()
            for _ in range(num_steps):

                x=x.detach().requires_grad_(True)
                logit = adv_loss(x)

                # if idx==0:
                #     print(logit)

                logit.backward()
            
                # Get gradient
                grad_x = x.grad.data

                # Compute Adversary
                x.data -= step_size * torch.sign(grad_x)
                
                # Clamp data between valid ranges
                x_orig_shape = x.shape
                x_send = torch.flatten(x)-torch.flatten(X)

                x.data = X.data + l1_projection_sort(x_send.data, self.input_domain[1]).view(x_orig_shape)
                # print(X.data + self.input_domain[1])
                # x.data = torch.clamp(x.data, X.data - self.input_domain[1], X.data + self.input_domain[1])
                
                x.data = torch.max(x.data, X.data - self.input_domain[1])
                x.data = torch.min(x.data, X.data + self.input_domain[1])

                x.grad.zero_()

            x_ub = x
            # print(idx, logit)
            fgsm_ub = logit
            final_bounds.append(fgsm_ub.detach().cpu())



        ########################
        for idx in range(num_idx):
            adv_loss = lambda x: -self.net(x)[0][idx]  # pylint: disable=cell-var-from-loop

            x = X.clone().requires_grad_(True).cuda()
            for _ in range(num_steps):

                x=x.detach().requires_grad_(True)
                logit = adv_loss(x)

                # if idx==0:
                #     print(logit)

                logit.backward()
            
                # Get gradient
                grad_x = x.grad.data

                # Compute Adversary
                x.data -= step_size * torch.sign(grad_x)
                
                # Clamp data between valid ranges
                x_orig_shape = x.shape
                x_send = torch.flatten(x)-torch.flatten(X)

                x.data = X.data + l1_projection_sort(x_send.data, self.input_domain[1]).view(x_orig_shape)
                # print(X.data + self.input_domain[1])
                # x.data = torch.clamp(x.data, X.data - self.input_domain[1], X.data + self.input_domain[1])
                
                x.data = torch.max(x.data, X.data - self.input_domain[1])
                x.data = torch.min(x.data, X.data + self.input_domain[1])

                x.grad.zero_()

            x_ub = x
            # print(idx, logit)
            fgsm_ub = -logit
            final_bounds.append(fgsm_ub.detach().cpu())

        for param in self.net.parameters():
            param.requires_grad = False

        return torch.stack(final_bounds)

    def advertorch_pgd_upper_bound(self, num_steps=50, step_size=0.01):

        if not self.emb_layer:
            print(self.input_domain[1])
            num_idx = self.layers[-1].bias.shape[0]
            final_bounds = []
            X = self.input_domain[0]

            current_model = self.net

            for param in current_model.parameters():
                param.requires_grad = True
            

            ##### lower bound
            last_layer_logits_loss_fn = lambda x, y: -x[0][y]
            from advertorch.attacks import L1PGDAttack
            adversary = L1PGDAttack(
                current_model, loss_fn=last_layer_logits_loss_fn, eps=self.input_domain[1],
                nb_iter=40, eps_iter=0.01, rand_init=False, clip_min=-1.0,
                clip_max=1.0, targeted=False, l1_sparsity=0.3)


            for idx in range(num_idx):

                x = X.detach().clone().requires_grad_(True).cuda()
                
                data = adversary.perturb(x, torch.tensor(idx))
                logit = current_model(data)[0][idx]

                # print(idx, logit)
                # input('')
                final_bounds.append(logit.detach().cpu())
            ###################

            ##### upper bound
            last_layer_logits_loss_fn = lambda x, y: x[0][y]
            adversary = L1PGDAttack(
                current_model, loss_fn=last_layer_logits_loss_fn, eps=self.input_domain[1],
                nb_iter=40, eps_iter=0.01, rand_init=False, clip_min=-1.0,
                clip_max=1.0, targeted=False, l1_sparsity=0.3)

            for idx in range(num_idx):

                x = X.detach().clone().requires_grad_(True).cuda()
                
                data = adversary.perturb(x, torch.tensor(idx))
                logit = current_model(data)[0][idx]

                # print(idx, logit)
                # input('')
                final_bounds.append(logit.detach().cpu())
            ###################

            for param in current_model.parameters():
                param.requires_grad = False

            return torch.stack(final_bounds)

        else:
            num_idx = self.layers[-1].bias.shape[0]
            final_bounds = []

            ##
            # current model
            cond_first_linear = self.weights[0]
            cond_first_linear_weights = self.weights[0].weights.squeeze(0)
            new_first_layer = nn.Linear(cond_first_linear_weights.shape[1], cond_first_linear_weights.shape[0]).cuda()
            new_first_layer.weights = nn.Parameter(cond_first_linear_weights)
            new_first_layer.bias = nn.Parameter(cond_first_linear.bias)

            new_layers = [new_first_layer] + self.layers[1:]
            current_model = nn.Sequential(*new_layers)
            ##

            for param in current_model.parameters():
                param.requires_grad = True

            ##### lower bound
            last_layer_logits_loss_fn = lambda x, y: -x[0][y]
            from mmbt.mmbt.models.concat_bow_relu import fast_simplex_pgd

            for idx in range(num_idx):

                data = fast_simplex_pgd(current_model, idx, last_layer_logits_loss_fn)
                logit = current_model(data)[0][idx]

                # print(idx, logit)
                # input('')
                final_bounds.append(logit.detach().cpu())
            ###################

            ##### upper bound
            last_layer_logits_loss_fn = lambda x, y: x[0][y]

            for idx in range(num_idx):

                
                data = fast_simplex_pgd(current_model, idx, last_layer_logits_loss_fn)
                logit = current_model(data)[0][idx]

                # print(idx, logit)
                # input('')
                final_bounds.append(logit.detach().cpu())
            ###################

            for param in current_model.parameters():
                param.requires_grad = False

            return torch.stack(final_bounds)



    # TODO: in case we need a better upper bounding strategy, this needs to be implemented.
    def get_upper_bound_pgd(self, domain, init_point):
        '''
        Compute an upper bound of the minimum of the network on `domain`. Adapted from naive_approximation.
        init_point is a possible initialization point (along the random samples)

        Any feasible point is a valid upper bound on the minimum so we will
        perform some random testing.
        '''

        # Not adapted to the batched case yet.
        raise NotImplementedError

        nb_samples = 2056
        batch_size = init_point.shape[0]
        nb_inp = init_point.shape
        nb_inp = (nb_samples, *nb_inp)

        # Not a great way of sampling but this will be good enough
        # We want to get rows that are >= 0
        # rand_samples = torch.randn(nb_inp)
        rand_samples = torch.rand(nb_inp)

        best_ub = float('inf')
        best_ub_inp = None

        domain_lb = domain.select(-1, 0).contiguous()
        domain_ub = domain.select(-1, 1).contiguous()
        domain_lb = domain_lb
        domain_ub = domain_ub
        domain_width = domain_ub - domain_lb
        domain_width = domain_width.expand(nb_inp)
        inps = domain_lb.expand(nb_inp) + domain_width * rand_samples
        inps[0] = init_point.clone()  # substitute one of the random samples with the provided input point
        inps = inps.view(nb_inp[0] * nb_inp[1], *nb_inp[2:])  # fold the domain batch dimensionality into the other

        with torch.enable_grad():
            batch_ub = float('inf')
            for i in range(1000):
                prev_batch_best = batch_ub

                self.net.zero_grad()
                if inps.grad is not None:
                    inps.grad.zero_()
                inps = inps.detach().requires_grad_()
                out = self.net(inps)

                folded_out = out.view(nb_inp[0], nb_inp[1])
                batch_ub, _ = folded_out.min(dim=0)
                if i == 0:
                    idx = torch.ones_like(batch_ub).type(torch.long)
                    best_ub = float('inf') * torch.ones_like(batch_ub)
                best_ub = torch.min(best_ub, batch_ub)
                _, new_idx = out.min(dim=0)
                idx = torch.where(batch_ub < best_ub, new_idx, idx)
                best_ub_inp = inps[idx[0]]  # TODO: this is most certainly wrong, after I am done with using batches, I need to debug it
                # TODO: try gather (see scatter in anderson_optimization w/o last argument)

                if (batch_ub >= prev_batch_best).any():
                    break

                all_samp_sum = out.sum() / nb_samples
                all_samp_sum.backward()
                grad = inps.grad

                max_grad, _ = grad.max(dim=0)
                min_grad, _ = grad.min(dim=0)
                grad_diff = max_grad - min_grad

                lr = 1e-2 * domain_width / grad_diff
                min_lr = lr.min()

                step = -min_lr*grad
                inps = inps + step

                inps = torch.max(inps, domain_lb)
                inps = torch.min(inps, domain_ub)

        return best_ub_inp, best_ub

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


        self.bigm_only = self.params["bigm"] and (self.params["bigm"] == "only")
        self.cut_init = ("cut" in self.params) and self.params["cut"] and (self.params["cut"] == "init")
        self.cut_only = ("cut" in self.params) and self.params["cut"] and (self.params["cut"] == "only")

        # hard-code default parameters, which are overridden by self.params
        opt_args = {
            'nb_outer_iter': 100,
            'initial_step_size': 1e-2,
            'final_step_size': 1e-4,
            'betas': (0.9, 0.999)
        }
        if self.cut_init:
            opt_args.update(self.params['init_params'])
        else:
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
        dual_vars = baseline_bigm_optimization.DualVars.naive_initialization(weights, additional_coeffs, device, lower_bounds[0].shape[1:])

        # Adam-related quantities.
        adam_stats = baseline_bigm_optimization.DualADAMStats(dual_vars.beta_0, beta1=opt_args['betas'][0], beta2=opt_args['betas'][1])
        init_step_size = opt_args['initial_step_size'] if not pinit else opt_args['initial_step_size_pinit']
        final_step_size = opt_args['final_step_size']

        add_coeff = next(iter(additional_coeffs.values()))
        batch_size = add_coeff.shape[:2]
        best_bound = -float("inf") * torch.ones(batch_size, device=device, dtype=self.precision)

        if self.store_bounds_progress >=0:
            start_logging_time = time.time()
            bound = baseline_bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
            # torch.max(best_bound, bound, out=best_bound)
            logging_time = time.time() - start_logging_time
            self.logger.add_point(len(weights), bound.clone(), logging_time=logging_time)

        if self.debug:
            obj_val = baseline_bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
            print(f"Average bound (and objective, they concide) at naive initialisation: {obj_val.mean().item()}")
            torch.max(best_bound, obj_val, out=best_bound)
            if self.view_tensorboard:
                self.writer.add_scalar('Average best bound', -best_bound.mean().item(), 0)
                self.writer.add_scalar('Average bound', -obj_val.mean().item(), 0)

        n_outer_iters = opt_args["nb_outer_iter"]
        for outer_it in itertools.count():
            if outer_it > n_outer_iters:
                break

            if self.debug:
                obj_val = baseline_bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)

            dual_vars_subg = baseline_bigm_optimization.compute_dual_subgradient(
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
                obj_val = baseline_bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
                torch.max(best_bound, obj_val, out=best_bound)
                print(f"Average obj at the end of adam iteration {outer_it}: {obj_val.mean().item()}")
                print(f"{outer_it} Average best bound: {best_bound.mean().item()}")
                if self.view_tensorboard:
                    self.writer.add_scalar('Average best bound', -best_bound.mean().item(), outer_it + 1)
                    self.writer.add_scalar('Average bound', -obj_val.mean().item(), outer_it + 1)

            if self.store_bounds_progress >= 0 and len(weights) == self.store_bounds_progress and self.bigm_only:
                if outer_it % 1 == 0:
                    start_logging_time = time.time()
                    bound = baseline_bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
                    # torch.max(best_bound, bound, out=best_bound)
                    logging_time = time.time() - start_logging_time
                    self.logger.add_point(len(weights), bound.clone(), logging_time=logging_time)

        # store the dual vars and primal vars for possible future usage
        self.bigm_dual_vars = dual_vars
        self.bigm_primal_vars = None
        self.bigm_adam_stats = adam_stats

        if self.bigm_only:
            self.children_init = baseline_bigm_optimization.BigMPInit(dual_vars)
            if self.store_bounds_primal:
                # Compute last matching primal (could make it a function...)
                nb_relu_layers = len(dual_vars.beta_0)
                xkm1, _ = baseline_bigm_optimization.first_layer_primal_linear_minimization(0, dual_vars.fs[0], None, clbs[0], cubs[0])
                zt = []
                xt = [xkm1]
                for lay_idx in range(1, nb_relu_layers):
                    # solve the inner problems.
                    xk, zk = baseline_bigm_optimization.layer_primal_linear_minimization(
                        lay_idx, dual_vars.fs[lay_idx], dual_vars.gs[lay_idx - 1], clbs[lay_idx], cubs[lay_idx])
                    xt.append(xk)
                    zt.append(zk)
                self.bounds_primal = baseline_anderson_optimization.PrimalVars(xt, zt)

        bound = baseline_bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
        return bound

    def cut_anderson_optimizer(self, weights, additional_coeffs, lower_bounds, upper_bounds):
        # ADAM subgradient ascent for a specific big-M solver which operates directly in the dual space
        # hard-code default parameters, which are overridden by self.params
        opt_args = {
            "nb_inner_iter": 100,
            "nb_iter": 100,
            "alpha_M": 1e-3,
            "beta_M": 1e-3,
            "bigm_algorithm": "adam",
            ###
            'random_cuts': False,
            'cut_frequency': 400,
            'max_cuts': 10,
            'cut_add': 2,
            'eta': 0,
            'volume': 0,
            'tau': 0,
            ###
            'betas': (0.9, 0.999),
            'initial_step_size': 1e-2,
            'final_step_size': 1e-3,
            "init_params": {
                'nb_outer_iter': 100,
                'initial_step_size': 1e-2,
                'final_step_size': 1e-4,
                'betas': (0.9, 0.999),
                'M_factor': 1.0
            },
            'nb_outer_iter': 10,
            'bigm': 'init',
        }
        # if self.cut_only:
        #     opt_args.update(self.params)
        # else:
        #     opt_args.update(self.params["cut_init_params"])

        # TODO: at the moment we don't support optimization with objective
        # function on something else than the last layer (although it could be
        # adapted) similarly to the way it was done in the proxlp_solver. For
        # now, just assert that we're in the expected case.
        assert len(additional_coeffs) == 1
        assert len(weights) in additional_coeffs  # if the last layer's coefficients are present
        # if self.store_bounds_progress >= 0 and self.bigm_only:
        self.logger.start_timing()

        device = lower_bounds[-1].device
        input_size = tuple(lower_bounds[0].shape[1:])
        add_coeff = next(iter(additional_coeffs.values()))
        batch_size = add_coeff.shape[:2]

        # Build the clamped bounds
        clbs = [lower_bounds[0]] + [torch.clamp(bound, 0, None) for bound in lower_bounds[1:]]  # 0 to n-1
        cubs = [upper_bounds[0]] + [torch.clamp(bound, 0, None) for bound in upper_bounds[1:]]  # 0 to n-1

        # Build the naive bounds
        nubs = [lin_k.interval_forward(cl_k, cu_k)[1] for (lin_k, cl_k, cu_k) in zip(weights, clbs, cubs)]  # 1 to n

        alpha_M = [opt_args["alpha_M"]] * len(clbs)
        beta_M = [opt_args["beta_M"]] * len(clbs)

        # Initialize alpha/beta with the output of the chosen big-m solver.
        bounds = self.bigm_subgradient_optimizer(weights, additional_coeffs, lower_bounds, upper_bounds)
        print(f"Average bounds after init with Bigm adam: {bounds.mean().item()}")

        dual_vars, primal_vars = baseline_cut_anderson_optimization.CutDualVars.bigm_initialization(
            self.bigm_dual_vars, weights, additional_coeffs, device, input_size, clbs, cubs, lower_bounds,
            upper_bounds, opt_args, alpha_M=alpha_M, beta_M=beta_M)

        ## Adam-related quantities.
        adam_stats = baseline_cut_anderson_optimization.DualADAMStats(dual_vars.sum_beta, beta1=opt_args['betas'][0],
                                                             beta2=opt_args['betas'][1])
        # initializes adam stats(momentum 1 and momentum 2 for dual variables(alpha,beta_0,beta_1))
        adam_stats.bigm_adam_initialization(dual_vars.sum_beta, self.bigm_adam_stats, beta1=opt_args['betas'][0],
                                            beta2=opt_args['betas'][1])
        # initializes adam stats(momentum 1 and momentum 2 for dual variables(alpha,beta_0,beta_1))



        best_bound = -float("inf") * torch.ones(batch_size, device=device, dtype=self.precision)

        bound_init = baseline_anderson_optimization.compute_bounds(dual_vars, weights, clbs, cubs)
        print(f"Average bound at initialisation: {bound_init.mean().item()}")
        torch.max(best_bound, bound_init, out=best_bound)

        self.alpha_time = 0
        self.beta_time = 0
        self.primals_time = 0

        if self.debug:
            if self.view_tensorboard:
                self.writer.add_scalar('Average best bound', -best_bound.mean().item(),
                                       self.params["init_params"]["nb_outer_iter"])
                self.writer.add_scalar('Average bound', -bound_init.mean().item(), self.params["init_params"]["nb_outer_iter"])

        init_step_size = opt_args['initial_step_size']
        final_step_size = opt_args['final_step_size']

        for steps in itertools.count():
            if steps >= opt_args["nb_iter"]:
                break

            dual_vars_subg = baseline_cut_anderson_optimization.compute_dual_subgradient_adam(
                weights, clbs, cubs, nubs, dual_vars, primal_vars, steps, precision=torch.float, opt_args=opt_args)
            step_size = init_step_size + ((steps + 1) / opt_args["nb_iter"]) * (final_step_size - init_step_size)
            # normal subgradient ascent
            # dual_vars.projected_linear_combination(
            #     step_size, dual_vars_subg)

            # do adam for subgradient ascent
            dual_vars_subg_updated = adam_stats.update_moments_take_projected_step(
                weights, step_size, steps, dual_vars, dual_vars_subg, primal_vars, clbs, cubs, nubs, lower_bounds,
                upper_bounds, opt_args['cut_frequency'], opt_args['max_cuts'], precision=torch.float, opt_args=opt_args)
            dual_vars.update_from_step(weights, dual_vars_subg_updated)

            if self.debug:
                bound = baseline_anderson_optimization.compute_bounds(dual_vars, weights, clbs, cubs)
                torch.max(best_bound, bound, out=best_bound)
                print(f"{steps} Average best bound: {best_bound.mean().item()}")
                print(f"{steps} Average bound: {bound.mean().item()}")
                if self.view_tensorboard:
                    self.writer.add_scalar('Average best bound', -best_bound.mean().item(),
                                           self.params["init_params"]["nb_outer_iter"] + steps + 1)
                    self.writer.add_scalar('Average bound', -bound.mean().item(),
                                           self.params["init_params"]["nb_outer_iter"] + steps + 1)

            if self.store_bounds_progress >= 0 and len(weights) == self.store_bounds_progress:
                if steps % 10 == 0:
                    start_logging_time = time.time()
                    bound = baseline_anderson_optimization.compute_bounds(dual_vars, weights, clbs, cubs)
                    torch.max(best_bound, bound, out=best_bound)
                    logging_time = time.time() - start_logging_time
                    self.logger.add_point(len(weights), best_bound.clone(), logging_time=logging_time)

            # if not self.store_bounds_primal:
            #     bound = baseline_anderson_optimization.compute_bounds(dual_vars, weights, clbs, cubs)
            #     torch.max(best_bound, bound, out=best_bound)
            #     print(f"Average best bound: {best_bound.mean().item()}")
            #     print(f"Average bound: {bound.mean().item()}")

        if self.cut_init:
            # store the dual vars and primal vars for possible future usage
            self.cut_dual_vars = dual_vars
            self.cut_primal_vars = primal_vars

        self.children_init = baseline_cut_anderson_optimization.CutInit(self.bigm_dual_vars, primal_vars)
        
        if self.store_bounds_primal:
            self.bounds_primal = primal_vars  # store the last matching primal
            bound = baseline_anderson_optimization.compute_bounds(dual_vars, weights, clbs, cubs)
            torch.max(best_bound, bound, out=best_bound)
            nb_neurons = int(best_bound.shape[1]/2)
            print(f"Average LB improvement: {(best_bound - bound_init)[:, nb_neurons:].mean()}")

        bound = baseline_anderson_optimization.compute_bounds(dual_vars, weights, clbs, cubs)
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