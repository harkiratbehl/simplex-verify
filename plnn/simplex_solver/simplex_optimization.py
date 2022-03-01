import itertools
import torch
from plnn.simplex_solver import utils
from plnn.branch_and_bound.utils import ParentInit
from plnn.simplex_solver import bigm_optimization
import torch
import math
import copy

def simplex_projection_sort(V, z=1):
    '''
    This function takes multiple input vectors and projects them onto simplexes.
    this function has been debugged and tested, it is correct!
    algo reference is https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
    numpy reference is https://gist.github.com/mblondel/c99e575a5207c76a99d714e8c6e08e89
    '''
    n_features = V.shape[1]
    U = torch.sort(V, axis=1, descending=True)[0]
    z = torch.ones(V.shape[0], device=V.device)*z
    cssv = torch.cumsum(U, dim=1) - z[:, None]
    ind = torch.arange(n_features, device=V.device) + 1
    cond = U - cssv / ind > 0
    rho = n_features - (cond == 0).sum(dim=1)# substitute for rho = np.count_nonzero(cond, axis=1)
    theta = cssv[torch.arange(V.shape[0], device=V.device), rho - 1] / rho
    return torch.clamp(V - theta[:, None], 0)


def compute_dual_subgradient(weights, dual_vars, lbs, ubs, l_preacts, u_preacts, outer_it, opt_args, cut_stats):
    """
    Given the network layers, post- and pre-activation bounds as lists of
    tensors, and dual variables (and functions thereof) as DualVars, compute the subgradient of the dual objective.
    :return: DualVars instance representing the subgradient for the dual variables (does not contain fs and gs)
    """

    # The step needs to be taken for all layers at once, as coordinate ascent seems to be problematic,
    # see https://en.wikipedia.org/wiki/Coordinate_descent

    nb_relu_layers = len(dual_vars.beta_0)

    alpha_subg = [torch.zeros_like(dual_vars.alpha[0])]
    beta_0_subg = [torch.zeros_like(dual_vars.beta_0[0])]
    beta_1_subg = [torch.zeros_like(dual_vars.beta_1[0])]
    old_gamma_subgs = []
    gamma_subgs = []
    rhos = []
    lambdas = []
    # layer 0
    xkm1, _ = bigm_optimization.layer_primal_linear_minimization(0, dual_vars.fs[0], None, lbs[0], ubs[0])

    for lay_idx in range(1, nb_relu_layers):
        # For each layer, we will do one step of subgradient descent on all dual variables at once.
        lin_k = weights[lay_idx - 1]
        # solve the inner problems.
        xk, zk = bigm_optimization.layer_primal_linear_minimization(lay_idx, dual_vars.fs[lay_idx], dual_vars.gs[lay_idx - 1],
                                                  lbs[lay_idx], ubs[lay_idx])

        # compute and store the subgradients.
        xk_hat = lin_k.forward(xkm1)
        alpha_subg.append(xk_hat - xk)
        beta_0_subg.append(xk - zk * u_preacts[lay_idx].unsqueeze(1))
        beta_1_subg.append(xk + (1 - zk) * l_preacts[lay_idx].unsqueeze(1) - xk_hat)

        ###########################################
        old_gamma_subgs.append([])

        unfolded_in = xk.view(xk.shape[0]*xk.shape[1], *xk.shape[2:])
        unfolded_in_flat = unfolded_in.view(unfolded_in.shape[0],-1)# batch_size * output_shape
        unfolded_in_flat_trans = unfolded_in_flat.t()
        for il in range(len(dual_vars.gamma_list[lay_idx-1])):
            # getting old_gamma_subg = lambda^T x* - rho
            gamma_subg = torch.zeros(unfolded_in.shape[0], device=xk.device)
            for lm in range(unfolded_in.shape[0]):
                    gamma_subg[lm] = dual_vars.lambda_list[lay_idx-1][il][:,lm]@unfolded_in_flat_trans[:, lm] - dual_vars.rho_list[lay_idx-1][il][lm]
            old_gamma_subgs[-1].append(gamma_subg)

        xkm1 = xk
        #############################################
        # compute and store the subgradients for lmo
        new_gamma_subg, new_rho, new_lambda = dual_vars.gammak_grad_lmo(lay_idx, weights, lbs, ubs, xk, outer_it, opt_args, cut_stats)
        gamma_subgs.append(new_gamma_subg)
        rhos.append(new_rho)
        lambdas.append(new_lambda)

    if not (len(dual_vars.gamma_list[lay_idx-1]) <= opt_args['max_cuts'] and outer_it%opt_args['cut_frequency'] < opt_args['cut_add']):
        gamma_subgs = []
        rhos = []
        lambdas = []

    return SimplexDualVars(alpha_subg, beta_0_subg, beta_1_subg, old_gamma_subgs, None, None, None, gamma_subgs, rhos, lambdas)

def compute_bounds(weights, dual_vars, clbs, cubs, l_preacts, u_preacts, prox=False):
    """
    Given the network layers, post- and pre-activation bounds  as lists of tensors, and dual variables
    (and functions thereof) as DualVars. compute the value of the (batch of) network bounds.
    If we are solving the prox problem (by default, no), check for non-negativity of dual vars. If they are negative,
    the bounds need to be -inf. This is because these non-negativity constraints have been relaxed, in that case.
    :return: a tensor of bounds, of size 2 x n_neurons of the layer to optimize. The first half is the negative of the
    upper bound of each neuron, the second the lower bound.
    """

    if prox:
        # in the prox case the dual variables might be negative (the constraint has been dualized). We therefore need
        # to clamp them to obtain a valid bound.
        c_dual_vars = dual_vars.get_nonnegative_copy(weights, l_preacts, u_preacts)
    else:
        c_dual_vars = dual_vars

    bounds = 0
    for lin_k, alpha_k_1 in zip(weights, c_dual_vars.alpha[1:]):
        b_k = lin_k.get_bias()
        bounds += utils.bdot(alpha_k_1, b_k)

    for f_k, cl_k, cu_k in zip(c_dual_vars.fs, clbs, cubs):
        f_k_matrix = f_k.view(f_k.shape[0],f_k.shape[1],-1)
        (a,b) = f_k_matrix.max(2)
        bounds -= a
        
        # bounds -= utils.bdot(torch.clamp(f_k, 0, None), cu_k.unsqueeze(1))
        # bounds -= utils.bdot(torch.clamp(f_k, None, 0), cl_k.unsqueeze(1))

    for g_k in c_dual_vars.gs:
        bounds -= torch.clamp(g_k, 0, None).view(*g_k.shape[:2], -1).sum(dim=-1)  # z to 1

    for beta_k_1, l_preact, lin_k in zip(c_dual_vars.beta_1[1:], l_preacts[1:], weights):
        bounds += utils.bdot(beta_k_1, (l_preact.unsqueeze(1) - lin_k.get_bias()))

    for gamma_k, rho_k in zip(c_dual_vars.gamma_list, c_dual_vars.rho_list):
        for gam_idx in range(len(gamma_k)):
            bounds -= gamma_k[gam_idx]*rho_k[gam_idx]
        # bounds -= utils.bdot(gamma_k, rho_k)

    return bounds

class SimplexDualVars(bigm_optimization.DualVars):
    """
    Class representing the dual variables alpha, beta_0, and beta_1, and their functions f and g.
    They are stored as lists of tensors, for ReLU indices from 0 to n-1 for beta_0, for indices 0 to n for
    the others.
    """
    def __init__(self, alpha, beta_0, beta_1, fs, gs, alpha_back, beta_1_back, gamma_list=None, rho_list=None, lambda_list=None):
        """
        Given the dual vars as lists of tensors (of correct length) along with their computed functions, initialize the
        class with these.
        alpha_back and beta_1_back are lists of the backward passes of alpha and beta_1. Useful to avoid
        re-computing them.
        """
        self.alpha = alpha
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.fs = fs
        self.gs = gs
        self.alpha_back = alpha_back
        self.beta_1_back = beta_1_back
        #
        self.gamma_list = gamma_list
        self.rho_list = rho_list
        self.lambda_list = lambda_list

    @staticmethod
    def from_super_class(super_instance, gamma_list=None, rho_list=None, lambda_list=None):
        """
        Return an instance of this class from an instance of the super class.
        """
        return SimplexDualVars(super_instance.alpha, super_instance.beta_0, super_instance.beta_1,
                           super_instance.fs, super_instance.gs, super_instance.alpha_back,
                           super_instance.beta_1_back, gamma_list=gamma_list, rho_list=rho_list, lambda_list=lambda_list)

    @staticmethod
    def naive_initialization(weights, additional_coeffs, device, input_size):
        """
        Given parameters from the optimize function, initialize the dual vairables and their functions as all 0s except
        some special corner cases. This is equivalent to initialising with naive interval propagation bounds.
        """
        add_coeff = next(iter(additional_coeffs.values()))
        batch_size = add_coeff.shape[:2]

        alpha = []  # Indexed from 0 to n, the last is constrained to the cost function, first is zero
        beta_0 = []  # Indexed from 0 to n-1, the first is always zero
        beta_1 = []  # Indexed from 0 to n, the first and last are always zero
        alpha_back = []  # Indexed from 1 to n,
        beta_1_back = []  # Indexed from 1 to n, last always 0

        # Build also the shortcut terms f and g
        fs = []  # Indexed from 0 to n-1
        gs = []  # Indexed from 1 to n-1

        # Fill in the variable holders with variables, all initiated to zero
        zero_tensor = lambda size: torch.zeros((*batch_size, *size), device=device)
        # Insert the dual variables for the box bound
        fs.append(zero_tensor(input_size))
        fixed_0_inpsize = zero_tensor(input_size)
        beta_0.append(fixed_0_inpsize)
        beta_1.append(fixed_0_inpsize)
        alpha.append(fixed_0_inpsize)
        for lay_idx, layer in enumerate(weights[:-1]):
            nb_outputs = layer.get_output_shape(beta_0[-1].shape)[2:]

            # Initialize the dual variables
            alpha.append(zero_tensor(nb_outputs))
            beta_0.append(zero_tensor(nb_outputs))
            beta_1.append(zero_tensor(nb_outputs))

            # Initialize the shortcut terms
            fs.append(zero_tensor(nb_outputs))
            gs.append(zero_tensor(nb_outputs))

        # Add the fixed values that can't be changed that comes from above
        alpha.append(additional_coeffs[len(weights)])
        beta_1.append(torch.zeros_like(alpha[-1]))

        for lay_idx in range(1, len(alpha)):
            alpha_back.append(weights[lay_idx-1].backward(alpha[lay_idx]))
            beta_1_back.append(weights[lay_idx-1].backward(beta_1[lay_idx]))

        # Adjust the fact that the last term for the f shorcut is not zero,
        # because it depends on alpha.
        fs[-1] = -weights[-1].backward(additional_coeffs[len(weights)])

        return SimplexDualVars(alpha, beta_0, beta_1, fs, gs, alpha_back, beta_1_back)

    @staticmethod
    def bigm_initialization(bigm_duals, weights, additional_coeffs, device, clbs, cubs, lower_bounds, upper_bounds, opt_args):
        """
        Given bigm dual variables, network weights, post/pre-activation lower and upper bounds,
        initialize the Anderson dual variables and their functions to the corresponding values of the bigm duals.
        Additionally, it returns the primal variables corresponding to the inner bigm minimization with those dual
        variables.
        """
        alpha, beta_0, beta_1, fs, gs, alpha_back, beta_1_back, gamma_list, rho_list, lambda_list = \
            bigm_duals.as_simplex_initialization(weights, clbs, cubs, lower_bounds, upper_bounds)

        base_duals = SimplexDualVars(alpha, beta_0, beta_1, fs, gs, alpha_back, beta_1_back, gamma_list, rho_list, lambda_list)

        return base_duals

    def update_f_g(self, l_preacts, u_preacts, lay_idx="all"):
        """
        Given the network pre-activation bounds as lists of tensors, update f_k and g_k in place.
        lay_idx are the layers (int or list) for which to perform the update. "all" means update all
        """
        if lay_idx == "all":
            lay_to_iter = range(len(self.beta_0))
        else:
            lay_to_iter = [lay_idx] if type(lay_idx) is int else list(lay_idx)

        for lay_idx in lay_to_iter:
            self.fs[lay_idx] = (
                    self.alpha[lay_idx] - self.alpha_back[lay_idx] -
                    (self.beta_0[lay_idx] + self.beta_1[lay_idx]) + self.beta_1_back[lay_idx])
            for gam_idx in range(len(self.gamma_list[lay_idx-1])):
                cond = self.lambda_list[lay_idx-1][gam_idx] *self.gamma_list[lay_idx-1][gam_idx]
                cond = cond.t()
                cond = cond.view(cond.shape[0], *self.fs[lay_idx].shape[2:])
                cond = cond.unsqueeze(0)
                self.fs[lay_idx] -= cond
            if lay_idx > 0:
                self.gs[lay_idx - 1] = (self.beta_0[lay_idx] * u_preacts[lay_idx].unsqueeze(1) +
                                        self.beta_1[lay_idx] * l_preacts[lay_idx].unsqueeze(1))

    def gammak_grad_lmo(self, lay_idx, weights, clbs, cubs, xk, outer_it, opt_args, cut_stats):
        gammak_list = self.gamma_list[lay_idx-1]
        new_gamma_subg, new_rho, new_lambda = [], [], []
        if (len(gammak_list) <= opt_args['max_cuts'] and outer_it % opt_args['cut_frequency'] < opt_args['cut_add']):
            print('adding cut number', len(gammak_list)+1)
            new_gamma_subg, new_rho, new_lambda = self.simplex_oracle(lay_idx, weights, xk, clbs, cut_stats)

        return new_gamma_subg, new_rho, new_lambda

    def simplex_oracle(self, lay_idx, weights, xk, clbs, cut_stats):
        """
        This function finds the most violated cutting plane as given by the simplex cutting plane equation.
        """
        lin_k = weights[lay_idx - 1]
        W_k = lin_k.weights
        n_iters=50

        if type(lin_k) in [utils.ConvOp, utils.BatchConvOp]:
            if xk.dim() == 5:
            # batch over multiple domains
                domain_batch_size = xk.shape[0]
                batch_size = xk.shape[1]
                unfolded_in = xk.view(domain_batch_size * batch_size, *xk.shape[2:])
                fold_back = True
            else:
                unfolded_in = xk
                fold_back = False

            unfolded_in_flat = unfolded_in.view(unfolded_in.shape[0],-1)# batch_size * output_shape
            unfolded_in_flat_trans = unfolded_in_flat.t()# output_shape * batch_size

            lambda_k_j = torch.zeros_like(unfolded_in_flat)
            lambda_k_j_trans = lambda_k_j.t() # output_shape * batch_size

            equ_layer_linear = lin_k.equivalent_linear(clbs[lay_idx-1].squeeze(0))

            batch_size = unfolded_in.shape[0]
            output_len = equ_layer_linear.weights.shape[0]
            input_len = equ_layer_linear.weights.shape[1]
            lambda_wb_clamped = torch.zeros(batch_size, output_len, input_len+1, device=xk.device)

            for it in range(n_iters):
                print('simplex oracle iteration: ', it)
                # STEP-1- Finding j*
                b_concatenated_weights = torch.cat((equ_layer_linear.weights, torch.zeros_like(equ_layer_linear.bias)[:,None]),1)# for origin point
                equ_layer_linear_wb = b_concatenated_weights + equ_layer_linear.bias[:,None]
                wb_clamped = torch.clamp(equ_layer_linear_wb, 0, None)
                for lm in range(batch_size):
                    lambda_wb_clamped[lm,:,:] = wb_clamped * lambda_k_j_trans[:, lm][:,None]
                lambda_wb_col_sum = torch.sum(lambda_wb_clamped, 1)#size is batch_size*input_len
                max_wb_col_sum, indices_wb_col_sum = torch.max(lambda_wb_col_sum, dim=1)
                h_ejstar = torch.index_select(wb_clamped,1,indices_wb_col_sum)# output_shape * batch_size
                # STEP-2- 
                # a) getting subgradient wrt lambda: g_t = h(e^{j*})-x*
                subgradient_k_j = h_ejstar - unfolded_in_flat_trans

                ### NORMAL SUBGRADIENT METHOD #######
                # b) y_{t+1} = \lambda - \eta g_t
                # yk = lambda_k_j_trans - 0.1*subgradient_k_j
                # # STEP-3- Projection onto Simplex
                # lambda_k_j_trans_new = simplex_projection_sort(yk.t()).t()
                ###############

                ##### ADAM FOR SUBGRADIENT DESCENT ####
                step_size = cut_stats.init_step_size + ((it + 1) / n_iters) * (cut_stats.final_step_size - cut_stats.init_step_size)
                lambda_k_j_trans_new = cut_stats.update_moments_take_projected_step(lay_idx, step_size, it, lambda_k_j_trans, subgradient_k_j)
                ##############

                error = torch.dist(lambda_k_j_trans, lambda_k_j_trans_new, 2)                
                lambda_k_j_trans = lambda_k_j_trans_new
                print(error)
                if error < 0.001:
                    break

            # lambda_k_j = lambda_k_j_trans.t().view(batch_size, *xk.shape[2:])
            # lambda_k_j = lambda_k_j.view_as(xk)
            # getting new_gamma_subg = lambda^T x* - rho
            new_gamma_subg = torch.zeros(batch_size, device=xk.device)
            for lm in range(batch_size):
                    new_gamma_subg[lm] = lambda_k_j_trans[:,lm]@unfolded_in_flat_trans[:, lm] - max_wb_col_sum[lm]

        else:
            if xk.dim() == 3:
            # batch over multiple domains
                domain_batch_size = xk.shape[0]
                batch_size = xk.shape[1]
                unfolded_in_flat = xk.view(domain_batch_size * batch_size, *xk.shape[2:])
                fold_back = True
            else:
                unfolded_in_flat = xk
                fold_back = False

            unfolded_in_flat_trans = unfolded_in_flat.t()# output_shape * batch_size

            lambda_k_j = torch.zeros_like(unfolded_in_flat)
            lambda_k_j_trans = lambda_k_j.t()# output_shape * batch_size

            equ_layer_linear = lin_k

            batch_size = unfolded_in_flat.shape[0]
            output_len = equ_layer_linear.weights.shape[0]
            input_len = equ_layer_linear.weights.shape[1]
            lambda_wb_clamped = torch.zeros(batch_size, output_len, input_len+1, device=xk.device)

            for it in range(n_iters):
                print('simplex oracle iteration: ', it)
                # STEP-1- Finding j*
                b_concatenated_weights = torch.cat((equ_layer_linear.weights, torch.zeros_like(equ_layer_linear.bias)[:,None]),1)# for origin point
                equ_layer_linear_wb = b_concatenated_weights + equ_layer_linear.bias[:,None]
                wb_clamped = torch.clamp(equ_layer_linear_wb, 0, None)
                for lm in range(batch_size):
                    lambda_wb_clamped[lm,:,:] = wb_clamped * lambda_k_j_trans[:, lm][:,None]
                lambda_wb_col_sum = torch.sum(lambda_wb_clamped, 1)#size is batch_size*input_len
                max_wb_col_sum, indices_wb_col_sum = torch.max(lambda_wb_col_sum, dim=1)
                h_ejstar = torch.index_select(wb_clamped,1,indices_wb_col_sum)
                # STEP-2- 
                # a) getting subgradient wrt lambda: g_t = h(e^{j*})-x*
                subgradient_k_j = h_ejstar - unfolded_in_flat_trans

                # ### NORMAL SUBGRADIENT METHOD #######
                # # b) y_{t+1} = \lambda - \eta g_t
                # yk = lambda_k_j_trans - 0.1*subgradient_k_j
                # # STEP-3- Projection onto Simplex
                # lambda_k_j_trans_new = simplex_projection_sort(yk.t()).t()
                # ###############

                ##### ADAM FOR SUBGRADIENT DESCENT ####
                step_size = cut_stats.init_step_size + ((it + 1) / n_iters) * (cut_stats.final_step_size - cut_stats.init_step_size)
                lambda_k_j_trans_new = cut_stats.update_moments_take_projected_step(lay_idx, step_size, it, lambda_k_j_trans, subgradient_k_j)
                ##############

                error = torch.dist(lambda_k_j_trans, lambda_k_j_trans_new, 2)
                # print('Error is: ', error)
                lambda_k_j_trans = lambda_k_j_trans_new
                if error < 0.001:
                    break

            # lambda_k_j = lambda_k_j_trans.t().view(batch_size, *xk.shape[2:])
            # lambda_k_j = lambda_k_j.view_as(xk)
            # getting new_gamma_subg = lambda^T x* - rho
            new_gamma_subg = torch.zeros(batch_size, device=xk.device)
            for lm in range(batch_size):
                    new_gamma_subg[lm] = lambda_k_j_trans[:,lm]@unfolded_in_flat_trans[:, lm]

        return new_gamma_subg, max_wb_col_sum, lambda_k_j_trans

class DualADAMStats:
    """
    class storing (and containing operations for) the ADAM statistics for the dual variables.
    they are stored as lists of tensors, for ReLU indices from 1 to n-1.
    """
    def __init__(self, beta_0, beta1=0.9, beta2=0.999):
        """
        Given beta_0 to copy the dimensionality from, initialize all ADAM stats to 0 tensors.
        """
        # first moments
        self.m1_alpha = []
        self.m1_beta_0 = []
        self.m1_beta_1 = []
        self.m1_gammas = []
        # second moments
        self.m2_alpha = []
        self.m2_beta_0 = []
        self.m2_beta_1 = []
        self.m2_gammas = []
        for lay_idx in range(1, len(beta_0)):
            self.m1_alpha.append(torch.zeros_like(beta_0[lay_idx]))
            self.m1_beta_0.append(torch.zeros_like(beta_0[lay_idx]))
            self.m1_beta_1.append(torch.zeros_like(beta_0[lay_idx]))
            self.m2_alpha.append(torch.zeros_like(beta_0[lay_idx]))
            self.m2_beta_0.append(torch.zeros_like(beta_0[lay_idx]))
            self.m2_beta_1.append(torch.zeros_like(beta_0[lay_idx]))
            self.m1_gammas.append([])
            self.m2_gammas.append([])

        self.coeff1 = beta1
        self.coeff2 = beta2
        self.epsilon = 1e-8

    def bigm_adam_initialization(self, beta_0, bigm_adam_stats, beta1=0.9, beta2=0.999):
        # first moments
        self.m1_alpha = []
        self.m1_beta_0 = []
        self.m1_beta_1 = []
        self.m1_gammas = []
        # second moments
        self.m2_alpha = []
        self.m2_beta_0 = []
        self.m2_beta_1 = []
        self.m2_gammas = []
        for lay_idx in range(1, len(beta_0)):
            # self.m1_alpha.append(torch.zeros_like(sum_beta[lay_idx]))
            # self.m2_alpha.append(torch.zeros_like(sum_beta[lay_idx]))
            self.m1_alpha.append(bigm_adam_stats.m1_alpha[lay_idx-1])
            self.m1_beta_0.append(bigm_adam_stats.m1_beta_0[lay_idx-1])
            self.m1_beta_1.append(bigm_adam_stats.m1_beta_1[lay_idx-1])
            self.m2_alpha.append(bigm_adam_stats.m2_alpha[lay_idx-1])
            self.m2_beta_0.append(bigm_adam_stats.m2_beta_0[lay_idx-1])
            self.m2_beta_1.append(bigm_adam_stats.m2_beta_1[lay_idx-1])

            self.m1_gammas.append([])
            self.m2_gammas.append([])

        self.coeff1 = beta1
        self.coeff2 = beta2
        self.epsilon = 1e-8

    def update_moments_take_projected_step(self, weights, step_size, outer_it, dual_vars, dual_vars_subg, opt_args):
        """
        Update the ADAM moments given the subgradients, and normal gd step size, then take the projected step from
        dual_vars.
        Update performed in place on dual_vars.
        """
        for lay_idx in range(1, len(dual_vars.beta_0)):
            # Update the ADAM moments.
            self.m1_alpha[lay_idx-1].mul_(self.coeff1).add_(dual_vars_subg.alpha[lay_idx], alpha=1-self.coeff1)
            self.m1_beta_0[lay_idx-1].mul_(self.coeff1).add_(dual_vars_subg.beta_0[lay_idx], alpha=1-self.coeff1)
            self.m1_beta_1[lay_idx-1].mul_(self.coeff1).add_(dual_vars_subg.beta_1[lay_idx], alpha=1-self.coeff1)
            self.m2_alpha[lay_idx-1].mul_(self.coeff2).addcmul_(dual_vars_subg.alpha[lay_idx], dual_vars_subg.alpha[lay_idx], value=1 - self.coeff2)
            self.m2_beta_0[lay_idx-1].mul_(self.coeff2).addcmul_(dual_vars_subg.beta_0[lay_idx], dual_vars_subg.beta_0[lay_idx], value=1 - self.coeff2)
            self.m2_beta_1[lay_idx-1].mul_(self.coeff2).addcmul_(dual_vars_subg.beta_1[lay_idx], dual_vars_subg.beta_1[lay_idx], value=1 - self.coeff2)

            bias_correc1 = 1 - self.coeff1 ** (outer_it + 1)
            bias_correc2 = 1 - self.coeff2 ** (outer_it + 1)
            corrected_step_size = step_size * math.sqrt(bias_correc2) / bias_correc1

            # Take the projected (non-negativity constraints) step.
            alpha_step_size = self.m1_alpha[lay_idx-1] / (self.m2_alpha[lay_idx-1].sqrt() + self.epsilon)
            dual_vars.alpha[lay_idx] = torch.clamp(dual_vars.alpha[lay_idx] + corrected_step_size * alpha_step_size, 0, None)

            beta_0_step_size = self.m1_beta_0[lay_idx-1] / (self.m2_beta_0[lay_idx-1].sqrt() + self.epsilon)
            dual_vars.beta_0[lay_idx] = torch.clamp(dual_vars.beta_0[lay_idx] + corrected_step_size * beta_0_step_size, 0, None)

            beta_1_step_size = self.m1_beta_1[lay_idx-1] / (self.m2_beta_1[lay_idx-1].sqrt() + self.epsilon)
            dual_vars.beta_1[lay_idx] = torch.clamp(dual_vars.beta_1[lay_idx] + corrected_step_size * beta_1_step_size, 0, None)

            # update pre-computed backward passes.
            dual_vars.alpha_back[lay_idx - 1] = weights[lay_idx - 1].backward(dual_vars.alpha[lay_idx])
            dual_vars.beta_1_back[lay_idx - 1] = weights[lay_idx - 1].backward(dual_vars.beta_1[lay_idx])

            ###########################################################
            ##################### UPDATING GAMMAS #####################
            ## 1. updating already existing gammas
            gam_idx = -1
            for gam_idx in range(len(dual_vars.gamma_list[lay_idx-1])):
                self.m1_gammas[lay_idx-1][gam_idx].mul_(self.coeff1).add_(dual_vars_subg.fs[lay_idx-1][gam_idx], alpha=1-self.coeff1)
                self.m2_gammas[lay_idx-1][gam_idx].mul_(self.coeff2).addcmul_(dual_vars_subg.fs[lay_idx-1][gam_idx], dual_vars_subg.fs[lay_idx-1][gam_idx], value=1 - self.coeff2)

                # Take the projected (non-negativity constraints) step.
                gamma_step_size = self.m1_gammas[lay_idx-1][gam_idx] / (self.m2_gammas[lay_idx-1][gam_idx].sqrt() + self.epsilon)
                dual_vars.gamma_list[lay_idx-1][gam_idx] = torch.clamp(dual_vars.gamma_list[lay_idx-1][gam_idx] + corrected_step_size * gamma_step_size, 0, None)

            ################## FOR THE SIMPLEX ORACLE #################
            if (len(dual_vars.gamma_list[lay_idx-1]) <= opt_args['max_cuts'] and outer_it%opt_args['cut_frequency'] < opt_args['cut_add']):
                self.m1_gammas[lay_idx-1].append(torch.zeros_like(dual_vars_subg.gamma_list[lay_idx-1]))
                self.m2_gammas[lay_idx-1].append(torch.zeros_like(dual_vars_subg.gamma_list[lay_idx-1]))

                #from here
                self.m1_gammas[lay_idx-1][-1].mul_(self.coeff1).add_(dual_vars_subg.gamma_list[lay_idx-1], alpha=1-self.coeff1)
                self.m2_gammas[lay_idx-1][-1].mul_(self.coeff2).addcmul_(dual_vars_subg.gamma_list[lay_idx-1], dual_vars_subg.gamma_list[lay_idx-1], value=1 - self.coeff2)

                # Take the projected (non-negativity constraints) step.
                gamma_step_size = self.m1_gammas[lay_idx-1][-1] / (self.m2_gammas[lay_idx-1][-1].sqrt() + self.epsilon)
                dual_vars.gamma_list[lay_idx-1].append(torch.clamp(corrected_step_size * gamma_step_size, 0, None))

                dual_vars.rho_list[lay_idx-1].append(dual_vars_subg.rho_list[lay_idx-1])
                dual_vars.lambda_list[lay_idx-1].append(dual_vars_subg.lambda_list[lay_idx-1])

class CutADAMStats:
    """
    class storing (and containing operations for) the ADAM statistics for the dual variables.
    they are stored as lists of tensors, for ReLU indices from 1 to n-1.
    """
    def __init__(self, beta_0, init_step_size=1e-2, final_step_size=1e-4, beta1=0.9, beta2=0.999):
        """
        Given beta_0 to copy the dimensionality from, initialize all ADAM stats to 0 tensors.
        """
        # first moments
        self.m1_lambda = []
        # second moments
        self.m2_lambda = []
        for lay_idx in range(1, len(beta_0)):
            unfolded_in = beta_0[lay_idx].view(beta_0[lay_idx].shape[0] * beta_0[lay_idx].shape[1], *beta_0[lay_idx].shape[2:])
            unfolded_in_flat = unfolded_in.view(unfolded_in.shape[0],-1)# batch_size * output_shape
            unfolded_in_flat_trans = unfolded_in_flat.t()# output_shape * batch_size
            self.m1_lambda.append(torch.zeros_like(unfolded_in_flat_trans))
            self.m2_lambda.append(torch.zeros_like(unfolded_in_flat_trans))

        self.coeff1 = beta1
        self.coeff2 = beta2
        self.epsilon = 1e-8

        self.init_step_size = init_step_size
        self.final_step_size = final_step_size

    def update_moments_take_projected_step(self, lay_idx, step_size, outer_it, lambda_k_j_trans, subgradient_k_j):
        """
        Update the ADAM moments given the subgradients, and normal gd step size, then take the projected step from
        dual_vars.
        Update performed in place on dual_vars.
        """
        # Update the ADAM moments.
        self.m1_lambda[lay_idx-1].mul_(self.coeff1).add_(subgradient_k_j, alpha=1-self.coeff1)
        self.m2_lambda[lay_idx-1].mul_(self.coeff2).addcmul_(subgradient_k_j, subgradient_k_j, value=1 - self.coeff2)

        bias_correc1 = 1 - self.coeff1 ** (outer_it + 1)
        bias_correc2 = 1 - self.coeff2 ** (outer_it + 1)
        corrected_step_size = step_size * math.sqrt(bias_correc2) / bias_correc1

        # Take the projected (non-negativity constraints) step.
        lambda_step_size = self.m1_lambda[lay_idx-1] / (self.m2_lambda[lay_idx-1].sqrt() + self.epsilon)
        yk = lambda_k_j_trans - corrected_step_size * lambda_step_size

        lambda_k_j_trans_new = simplex_projection_sort(yk.t()).t()# output_shape * batch_size

        return lambda_k_j_trans_new
