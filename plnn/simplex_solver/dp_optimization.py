import itertools
import torch
from plnn.simplex_solver import utils
from plnn.branch_and_bound.utils import ParentInit
from plnn.simplex_solver import bigm_optimization
import torch
import math
import copy
import torch.nn.functional as F

def layer_primal_linear_minimization(lay_idx, f_k, g_k, cl_k, cu_k):
    """
    Given the post-activation bounds and the (functions of) dual variables of the current layer tensors
    (shape 2 * n_neurons_to_opt x c_layer_size), compute the values of the primal variables (x and z) minimizing the
    inner objective.
    :return: optimal x, optimal z (tensors, shape: 2 * n_neurons_to_opt x c_layer_size)
    """
    # opt_x_k = (torch.where(f_k >= 0, cu_k.unsqueeze(1), cl_k.unsqueeze(1)))
    f_k_matrix = f_k.view(f_k.shape[0],f_k.shape[1],-1)
    (b,c) = torch.max(f_k_matrix, 2)
    a = F.one_hot(c, f_k_matrix.shape[2])
    for i in range(f_k_matrix.shape[0]):
        for j in range(f_k_matrix.shape[1]):
            if b[i][j]<0:
                a[i][j]=0
    opt_x_k = a.view(f_k.shape).to(dtype=torch.float32)
    if lay_idx > 0:
        opt_z_k = (torch.where(g_k >= 0, torch.ones_like(g_k), torch.zeros_like(g_k)))
    else:
        # g_k is defined from 1 to n - 1.
        opt_z_k = None

    return opt_x_k, opt_z_k

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

    for beta_k_2, lin_k in zip(c_dual_vars.beta_2[1:], weights):
        bounds -= utils.bdot(beta_k_2, torch.clamp(lin_k.get_bias(), 0, None))

    return bounds

def compute_dual_subgradient(weights, dual_vars, lbs, ubs, l_preacts, u_preacts):
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
    beta_2_subg = [torch.zeros_like(dual_vars.beta_2[0])]
    xkm1, _ = layer_primal_linear_minimization(0, dual_vars.fs[0], None, lbs[0], ubs[0])
    for lay_idx in range(1, nb_relu_layers):
        # For each layer, we will do one step of subgradient descent on all dual variables at once.
        lin_k = weights[lay_idx - 1]
        # solve the inner problems.
        xk, zk = layer_primal_linear_minimization(lay_idx, dual_vars.fs[lay_idx], dual_vars.gs[lay_idx - 1],
                                                  lbs[lay_idx], ubs[lay_idx])

        # compute and store the subgradients.
        xk_hat = lin_k.forward(xkm1)
        alpha_subg.append(xk_hat - xk)
        beta_0_subg.append(xk - zk * u_preacts[lay_idx].unsqueeze(1))
        beta_1_subg.append(xk + (1 - zk) * l_preacts[lay_idx].unsqueeze(1) - xk_hat)
        ####################
        ## for dp
        beta_2_subg.append(xk - lin_k.dp_forward(xkm1))

        xkm1 = xk

    return DPDualVars(alpha_subg, beta_0_subg, beta_1_subg, beta_2_subg, None, None, None, None, None)

class DPDualVars(bigm_optimization.DualVars):
    """
    Class representing the dual variables alpha, beta_0, and beta_1, beta_2, and their functions f and g.
    They are stored as lists of tensors, for ReLU indices from 0 to n-1 for beta_0, for indices 0 to n for
    the others.
    RELAXATION: Big-m + DP
    """
    def __init__(self, alpha, beta_0, beta_1, beta_2, fs, gs, alpha_back, beta_1_back, beta_2_back):
        """
        Given the dual vars as lists of tensors (of correct length) along with their computed functions, initialize the
        class with these.
        alpha_back and beta_1_back are lists of the backward passes of alpha and beta_1. Useful to avoid
        re-computing them.
        """
        self.alpha = alpha
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.fs = fs
        self.gs = gs
        self.alpha_back = alpha_back
        self.beta_1_back = beta_1_back
        self.beta_2_back = beta_2_back

    @staticmethod
    def from_super_class(super_instance):
        """
        Return an instance of this class from an instance of the super class.
        """
        return DPDualVars(super_instance.alpha, super_instance.beta_0, super_instance.beta_1, super_instance.beta_2, super_instance.fs, super_instance.gs, super_instance.alpha_back, super_instance.beta_1_back, super_instance.beta_2_back)

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
        beta_2 = []  # Indexed from 0 to n, the first and last are always zero
        alpha_back = []  # Indexed from 1 to n,
        beta_1_back = []  # Indexed from 1 to n, last always 0
        beta_2_back = []  # Indexed from 1 to n, last always 0

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
        beta_2.append(fixed_0_inpsize)
        alpha.append(fixed_0_inpsize)
        for lay_idx, layer in enumerate(weights[:-1]):
            nb_outputs = layer.get_output_shape(beta_0[-1].shape)[2:]

            # Initialize the dual variables
            alpha.append(zero_tensor(nb_outputs))
            beta_0.append(zero_tensor(nb_outputs))
            beta_1.append(zero_tensor(nb_outputs))
            beta_2.append(zero_tensor(nb_outputs))

            # Initialize the shortcut terms
            fs.append(zero_tensor(nb_outputs))
            gs.append(zero_tensor(nb_outputs))

        # Add the fixed values that can't be changed that comes from above
        alpha.append(additional_coeffs[len(weights)])
        beta_1.append(torch.zeros_like(alpha[-1]))
        beta_2.append(torch.zeros_like(alpha[-1]))

        for lay_idx in range(1, len(alpha)):
            alpha_back.append(weights[lay_idx-1].backward(alpha[lay_idx]))
            beta_1_back.append(weights[lay_idx-1].backward(beta_1[lay_idx]))
            beta_2_back.append(weights[lay_idx-1].backward(beta_2[lay_idx]))

        # Adjust the fact that the last term for the f shorcut is not zero,
        # because it depends on alpha.
        fs[-1] = -weights[-1].backward(additional_coeffs[len(weights)])

        return DPDualVars(alpha, beta_0, beta_1, beta_2, fs, gs, alpha_back, beta_1_back, beta_2_back)

    @staticmethod
    def bigm_initialization(bigm_duals, weights, additional_coeffs, device, clbs, cubs, lower_bounds, upper_bounds, opt_args):
        """
        Given bigm dual variables, network weights, post/pre-activation lower and upper bounds,
        initialize the dp dual variables and their functions to the corresponding values of the bigm duals.
        Additionally, it returns the primal variables corresponding to the inner bigm minimization with those dual
        variables.
        """
        alpha, beta_0, beta_1, beta_2, fs, gs, alpha_back, beta_1_back, beta_2_back= \
            bigm_duals.as_dp_initialization(weights, clbs, cubs, lower_bounds, upper_bounds)

        base_duals = DPDualVars(alpha, beta_0, beta_1, beta_2, fs, gs, alpha_back, beta_1_back, beta_2_back)

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
                    (self.beta_0[lay_idx] + self.beta_1[lay_idx] + self.beta_2[lay_idx]) + self.beta_1_back[lay_idx] + self.beta_2_back[lay_idx])
            if lay_idx > 0:
                self.gs[lay_idx - 1] = (self.beta_0[lay_idx] * u_preacts[lay_idx].unsqueeze(1) +
                                        self.beta_1[lay_idx] * l_preacts[lay_idx].unsqueeze(1))

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
        self.m1_beta_2 = []
        # second moments
        self.m2_alpha = []
        self.m2_beta_0 = []
        self.m2_beta_1 = []
        self.m2_beta_2 = []
        for lay_idx in range(1, len(beta_0)):
            self.m1_alpha.append(torch.zeros_like(beta_0[lay_idx]))
            self.m1_beta_0.append(torch.zeros_like(beta_0[lay_idx]))
            self.m1_beta_1.append(torch.zeros_like(beta_0[lay_idx]))
            self.m1_beta_2.append(torch.zeros_like(beta_0[lay_idx]))

            self.m2_alpha.append(torch.zeros_like(beta_0[lay_idx]))
            self.m2_beta_0.append(torch.zeros_like(beta_0[lay_idx]))
            self.m2_beta_1.append(torch.zeros_like(beta_0[lay_idx]))
            self.m2_beta_2.append(torch.zeros_like(beta_0[lay_idx]))

        self.coeff1 = beta1
        self.coeff2 = beta2
        self.epsilon = 1e-8

    def bigm_adam_initialization(self, beta_0, bigm_adam_stats, beta1=0.9, beta2=0.999):
        # first moments
        self.m1_alpha = []
        self.m1_beta_0 = []
        self.m1_beta_1 = []
        self.m1_beta_2 = []
        # second moments
        self.m2_alpha = []
        self.m2_beta_0 = []
        self.m2_beta_1 = []
        self.m2_beta_2 = []
        for lay_idx in range(1, len(beta_0)):
            # self.m1_alpha.append(torch.zeros_like(sum_beta[lay_idx]))
            # self.m2_alpha.append(torch.zeros_like(sum_beta[lay_idx]))
            self.m1_alpha.append(bigm_adam_stats.m1_alpha[lay_idx-1])
            self.m1_beta_0.append(bigm_adam_stats.m1_beta_0[lay_idx-1])
            self.m1_beta_1.append(bigm_adam_stats.m1_beta_1[lay_idx-1])
            self.m1_beta_2.append(torch.zeros_like(beta_0[lay_idx]))
            self.m2_alpha.append(bigm_adam_stats.m2_alpha[lay_idx-1])
            self.m2_beta_0.append(bigm_adam_stats.m2_beta_0[lay_idx-1])
            self.m2_beta_1.append(bigm_adam_stats.m2_beta_1[lay_idx-1])
            self.m2_beta_2.append(torch.zeros_like(beta_0[lay_idx]))

        self.coeff1 = beta1
        self.coeff2 = beta2
        self.epsilon = 1e-8

    def update_moments_take_projected_step(self, weights, step_size, outer_it, dual_vars, dual_vars_subg):
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
            self.m1_beta_2[lay_idx-1].mul_(self.coeff1).add_(dual_vars_subg.beta_2[lay_idx], alpha=1-self.coeff1)
            
            self.m2_alpha[lay_idx-1].mul_(self.coeff2).addcmul_(dual_vars_subg.alpha[lay_idx], dual_vars_subg.alpha[lay_idx], value=1 - self.coeff2)
            self.m2_beta_0[lay_idx-1].mul_(self.coeff2).addcmul_(dual_vars_subg.beta_0[lay_idx], dual_vars_subg.beta_0[lay_idx], value=1 - self.coeff2)
            self.m2_beta_1[lay_idx-1].mul_(self.coeff2).addcmul_(dual_vars_subg.beta_1[lay_idx], dual_vars_subg.beta_1[lay_idx], value=1 - self.coeff2)
            self.m2_beta_2[lay_idx-1].mul_(self.coeff2).addcmul_(dual_vars_subg.beta_2[lay_idx], dual_vars_subg.beta_2[lay_idx], value=1 - self.coeff2)

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

            beta_2_step_size = self.m1_beta_2[lay_idx-1] / (self.m2_beta_2[lay_idx-1].sqrt() + self.epsilon)
            dual_vars.beta_2[lay_idx] = torch.clamp(dual_vars.beta_2[lay_idx] + corrected_step_size * beta_2_step_size, 0, None)

            # update pre-computed backward passes.
            dual_vars.alpha_back[lay_idx - 1] = weights[lay_idx - 1].backward(dual_vars.alpha[lay_idx])
            dual_vars.beta_1_back[lay_idx - 1] = weights[lay_idx - 1].backward(dual_vars.beta_1[lay_idx])
            ####################
            ## for dp
            dual_vars.beta_2_back[lay_idx - 1] = weights[lay_idx - 1].dp_backward(dual_vars.beta_2[lay_idx])