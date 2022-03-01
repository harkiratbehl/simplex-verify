"""
Files specific to the "dual of proximal" derivation for the Anderson relaxation.
"""

import itertools
import torch
from plnn.proxlp_solver import utils
from plnn.explp_solver import anderson_optimization


def compute_bounds(dual_vars, weights, clbs, cubs):
    """
    Compute the problem bounds, given the dual variables (instance of DualVars), their sufficient statistics,
    intermediate bounds (clbs, cubs) (as lists of tensors) and network layers (weights, LinearOp, ConvOp classes from
    proxlp_solver.utils).
    Dual variables are tensors of size opt_layer_width x layer shape, the intermediate bounds lack opt_layer_width.
    :return: a tensor of bounds, of size 2 x n_neurons of the layer to optimize. The first half is the negative of the
    upper bound of each neuron, the second the lower bound.
    """

    # Update f and g to reflect the fact that the dual used for the bounds computation does not have gamma and delta
    new_fs = []
    for f_k, gamma_k_l, gamma_k_u in zip(dual_vars.fs, dual_vars.gamma_l, dual_vars.gamma_u):
        fmgamma = f_k + gamma_k_u - gamma_k_l
        new_fs.append(fmgamma)
    new_gs = []
    for g_k, delta_k_0, delta_k_1 in zip(dual_vars.gs, dual_vars.delta_0, dual_vars.delta_1):
        gmdelta = g_k + delta_k_1 - delta_k_0
        new_gs.append(gmdelta)

    return anderson_optimization.compute_bounds(dual_vars, weights, clbs, cubs, new_fs=new_fs, new_gs=new_gs)


def betak_nnmp_step(lay_idx, weights, clbs, cubs, nubs, l_preacts, u_preacts, dual_vars, primal_anchors, eta,
                    precision=torch.float, use_preactivation=False, preact_planet=False, iter=-1):
    # Perform the optimisation on beta through Non-Negative Matching Pursuit.

    # reference current layer variables via shorter names
    sum_betak = dual_vars.sum_beta[lay_idx]
    sum_WkIbetak = dual_vars.sum_Wp1Ibetap1[lay_idx - 1]
    sum_Wk1mIubetak = dual_vars.sum_W1mIubeta[lay_idx]
    sum_WkIlbetak = dual_vars.sum_WIlbeta[lay_idx]
    betak_norm = dual_vars.beta_norm[lay_idx]

    beta_sum_lmo, beta_WI_lmo, beta_WIl_lmo, beta_W1mIu_lmo, atom_grad, _, eq_xkm1, eq_xk, eq_zk = dual_vars.betak_grad_lmo(
        lay_idx, weights, clbs, cubs, nubs, l_preacts, u_preacts, primal_anchors, eta, precision, use_preactivation,
        preact_planet)

    # compute the inner product of the current iterate (if at 0, the clamping will make it 0)
    nonnorm_inner_grad_iter = dual_vars.betak_inner_grad(lay_idx, weights, eq_xkm1, eq_xk, eq_zk)
    clamped_norm = torch.clamp(betak_norm, 1e-6, None)
    inner_grad_iterate = -1/clamped_norm * nonnorm_inner_grad_iter

    # take the max with the I_star gradient by storing the difference and conditioning further operations on it.
    # useful to avoid synchronization.
    inner_grad_atom = torch.clamp(atom_grad, 0, None).sum(dim=tuple(range(2, atom_grad.dim()))).unsqueeze(-1)
    inner_grad_diff = (inner_grad_atom - inner_grad_iterate).view((*inner_grad_atom.shape[:2], *((1,) * (atom_grad.dim() - 2))))
    # TODO: it is not true that the normalisation of the current iterate does not matter, as it influences what is the argmin here (seems to be OK anyways)

    # define the selected atom (or normalized negative iterate) as a function of sufficient stats
    clamped_norm = clamped_norm.view_as(inner_grad_diff)
    atom_beta_sum = torch.where(inner_grad_diff >= 0, beta_sum_lmo, -sum_betak/clamped_norm)
    atom_beta_WI = torch.where(
        inner_grad_diff.view((*sum_WkIbetak.shape[:2], *((1,) * (sum_WkIbetak.dim() - 2)))) >= 0,
        beta_WI_lmo, -sum_WkIbetak/clamped_norm.view((*sum_WkIbetak.shape[:2], *((1,) * (sum_WkIbetak.dim() - 2)))))
    atom_beta_WIl = torch.where(inner_grad_diff >= 0, beta_WIl_lmo, -sum_WkIlbetak/clamped_norm)
    atom_beta_W1mIu = torch.where(inner_grad_diff >= 0, beta_W1mIu_lmo, -sum_Wk1mIubetak/clamped_norm)

    # Let's compute the optimal step size:
    optimal_step_size = dual_vars.betak_optimal_step_size(
        lay_idx, weights, eq_xkm1, eq_xk, eq_zk, primal_anchors, eta, atom_beta_sum, atom_beta_WI, atom_beta_WIl,
        atom_beta_W1mIu)

    if iter != -1:
        optimal_step_size = torch.ones_like(optimal_step_size) * (2 / (2 + iter))

    # Selectively clamp step size to avoid that backwards iters (with -x_t) don't cross the origin
    optimal_step_size = torch.where(
        inner_grad_diff >= 0, optimal_step_size,
        torch.max(torch.min(optimal_step_size, clamped_norm * 0.999999), torch.zeros_like(optimal_step_size)))

    new_sum_betak = sum_betak.addcmul(optimal_step_size, atom_beta_sum)
    new_sum_WkIbetak = sum_WkIbetak.addcmul(
        optimal_step_size.view((*sum_WkIbetak.shape[:2], *((1,) * (sum_WkIbetak.dim() - 2)))), atom_beta_WI)
    new_sum_WkIubetak = sum_Wk1mIubetak.addcmul(optimal_step_size, atom_beta_W1mIu)
    new_sum_WkIlbetak = sum_WkIlbetak.addcmul(optimal_step_size, atom_beta_WIl)

    return new_sum_betak, new_sum_WkIbetak, new_sum_WkIubetak, new_sum_WkIlbetak


def alphak_nnmp_step(lay_idx, weights, dual_vars, primal_anchors, eta, iter=-1, precision=torch.float):
    # Perform the optimisation on beta through Non-Negative Matching Pursuit.

    # reference current layer variables via shorter names
    alpha_k = dual_vars.alpha[lay_idx]
    alphak_norm = dual_vars.alpha_norm[lay_idx]

    # compute alpha_k's gradient and its LMO over it.
    lmo_alphak, grad_alphak, xkm1_eq, xk_eq = dual_vars.alphak_grad_lmo(lay_idx, weights, primal_anchors, eta, precision)

    # compute the inner product of the current iterate (if at 0, the clamping will make it 0)
    inner_alpha = dual_vars.alphak_inner_grad(lay_idx, grad_alphak)
    clamped_norm = torch.clamp(alphak_norm, 1e-6, None)
    inner_grad_iterate = -1 / clamped_norm * inner_alpha

    # take the max with the best atom gradient by storing the difference and conditioning further operations on it.
    # useful to avoid synchronization.
    inner_grad_atom = utils.bdot(lmo_alphak, grad_alphak).unsqueeze(-1)
    inner_grad_diff = (inner_grad_atom - inner_grad_iterate).view((*inner_grad_atom.shape[:2], *((1,) * (lmo_alphak.dim() - 2))))
    # TODO: it is not true that the normalisation of the current iterate does not matter, as it influences what is the argmin here (seems to be OK anyways)

    # select the atom with which to take the linear combination
    clamped_norm = clamped_norm.view_as(inner_grad_diff)
    atom_alphak = torch.where(inner_grad_diff >= 0, lmo_alphak, -alpha_k / clamped_norm)

    # Let's compute the optimal step size
    optimal_step_size = dual_vars.alphak_optimal_step_size(lay_idx, weights, xkm1_eq, xk_eq, primal_anchors, eta,
                                                           atom_alphak)

    if iter != -1:
        optimal_step_size = torch.ones_like(optimal_step_size) * (2 / (2 + iter))

    # if (inner_grad_diff < 0).any():
    #    print("choosing -x_t")

    # Selectively clamp step size to avoid that backwards iters (with -x_t) don't cross the origin
    optimal_step_size = torch.where(
        inner_grad_diff >= 0, optimal_step_size,
        torch.max(torch.min(optimal_step_size, clamped_norm * 0.999999), torch.zeros_like(optimal_step_size)))

    new_alpha_k = alpha_k.addcmul(optimal_step_size, atom_alphak)

    # assert new_alpha_k_0.min() >= 0
    # assert new_alpha_k_1.min() >= 0
    return new_alpha_k


# IMPORTANT: this performs worse than updating on alpha and beta separately.
def alphak_betak_nnmp_step(lay_idx, weights, clbs, cubs, nubs, l_preacts, u_preacts, dual_vars, primal_anchors, eta,
                    precision=torch.float, use_preactivation=False, preact_planet=False, iter=-1):
    # Perform the optimisation on beta through Non-Negative Matching Pursuit.
    # TODO: the cnn adaptation of this is as bad as using a linear equivalent memorywise and slower.
    # TODO: need a specialized CUDA kernel

    # reference current layer variables via shorter names
    lin_k = weights[lay_idx - 1]
    sum_betak = dual_vars.sum_beta[lay_idx]
    sum_WkIbetak = dual_vars.sum_Wp1Ibetap1[lay_idx - 1]
    sum_Wk1mIubetak = dual_vars.sum_W1mIubeta[lay_idx]
    sum_WkIlbetak = dual_vars.sum_WIlbeta[lay_idx]
    betak_norm = dual_vars.beta_norm[lay_idx]
    alpha_k = dual_vars.alpha[lay_idx]
    alphak_norm = dual_vars.alpha_norm[lay_idx]

    # TODO: isn't there any cleaner way to do this flattening?
    # Flattening of tensors that should be treated as fully connected inputs/outputs.
    if type(lin_k) is utils.LinearOp and lin_k.flatten_from_shape is not None:
        batch_size = sum_betak.shape[:2]
        sum_betak = sum_betak.view(*batch_size, -1)
        sum_WkIbetak = sum_WkIbetak.view(*batch_size, -1)
        sum_Wk1mIubetak = sum_Wk1mIubetak.view(*batch_size, -1)
        sum_WkIlbetak = sum_WkIlbetak.view(*batch_size, -1)

    # compute alpha_k's gradient and its LMO over it.
    lmo_alphak, grad_alphak, _, _ = dual_vars.alphak_grad_lmo(lay_idx, weights, primal_anchors, eta, precision)

    # compute beta_k's gradient and its LMO over it, done jointly with alpha to compute the correct resulting
    # sufficient stats.
    beta_sum_lmo, beta_WI_lmo, beta_WIl_lmo, beta_W1mIu_lmo, beta_atom_grad, _, eq_xkm1, eq_xk, eq_zk = dual_vars.betak_grad_lmo(
        lay_idx, weights, clbs, cubs, nubs, l_preacts, u_preacts, primal_anchors, eta, precision, use_preactivation,
        preact_planet, grad_alphak=grad_alphak)

    lmo_is_alpha = (grad_alphak > beta_atom_grad) & (grad_alphak > 0)
    lmo_alphak = lmo_alphak * lmo_is_alpha.type(precision)

    # compute the inner product of the current iterate (if at 0, the clamping will make it 0)
    alpha_inner_grad = dual_vars.alphak_inner_grad(lay_idx, grad_alphak)
    beta_inner_grad = dual_vars.betak_inner_grad(lay_idx, weights, eq_xkm1, eq_xk, eq_zk)
    clamped_norm = torch.clamp(betak_norm + alphak_norm, 1e-6, None)
    inner_grad_iterate = -1/clamped_norm * (alpha_inner_grad + beta_inner_grad)

    lmo_is_beta = (beta_atom_grad > grad_alphak) & (beta_atom_grad >= 0)

    # take the max with the I_star gradient by storing the difference and conditioning further operations on it.
    # useful to avoid synchronization.

    beta_inner_grad_atom = utils.bdot(beta_atom_grad, lmo_is_beta.type(precision)).unsqueeze(-1)
    alpha_inner_grad_atom = utils.bdot(lmo_alphak, grad_alphak).unsqueeze(-1)
    inner_grad_atom = alpha_inner_grad_atom + beta_inner_grad_atom
    inner_grad_diff = (inner_grad_atom - inner_grad_iterate).\
        view((*inner_grad_atom.shape[:2], *((1,) * (beta_atom_grad.dim() - 2))))

    # define the selected atom (or normalized negative iterate) as a function of sufficient stats
    clamped_norm = clamped_norm.view_as(inner_grad_diff)
    atom_alphak = torch.where(inner_grad_diff >= 0, lmo_alphak, -alpha_k / clamped_norm)
    atom_beta_sum = torch.where(inner_grad_diff >= 0, beta_sum_lmo, -sum_betak/clamped_norm)
    atom_beta_WI = torch.where(inner_grad_diff >= 0, beta_WI_lmo, -sum_WkIbetak/clamped_norm)
    atom_beta_WIl = torch.where(inner_grad_diff >= 0, beta_WIl_lmo, -sum_WkIlbetak/clamped_norm)
    atom_beta_W1mIu = torch.where(inner_grad_diff >= 0, beta_W1mIu_lmo, -sum_Wk1mIubetak/clamped_norm)

    # Let's compute the optimal step size:
    optimal_step_size = dual_vars.alphak_betak_optimal_step_size(lay_idx, weights, eq_xkm1, eq_xk, eq_zk, eta,
                                                                 atom_alphak, atom_beta_sum, atom_beta_WI,
                                                                 atom_beta_WIl, atom_beta_W1mIu)

    if iter != -1:
        optimal_step_size = torch.ones_like(optimal_step_size) * (2 / (2 + iter))

    # Selectively clamp step size to avoid that backwards iters (with -x_t) don't cross the origin
    optimal_step_size = torch.where(
        inner_grad_diff >= 0, optimal_step_size,
        torch.max(torch.min(optimal_step_size, clamped_norm * 0.999999), torch.zeros_like(optimal_step_size)))

    new_alpha_k = alpha_k.addcmul(optimal_step_size, atom_alphak)
    new_sum_betak = sum_betak.addcmul(optimal_step_size, atom_beta_sum)
    new_sum_WkIbetak = sum_WkIbetak.addcmul(optimal_step_size, atom_beta_WI)
    new_sum_WkIubetak = sum_Wk1mIubetak.addcmul(optimal_step_size, atom_beta_W1mIu)
    new_sum_WkIlbetak = sum_WkIlbetak.addcmul(optimal_step_size, atom_beta_WIl)

    return new_alpha_k, new_sum_betak, new_sum_WkIbetak, new_sum_WkIubetak, new_sum_WkIlbetak


def compute_optimal_deltak(lay_idx, eta, dual_vars, primal_anchors):

    # Get AdaGrad terms (which are 1 in case we use the Eclidean norm rather than AdaGrad's)
    Hk_sqrt = primal_anchors.H_sqrt[lay_idx - 1] if primal_anchors.H[lay_idx - 1] is not None else 1

    etak = eta.z_eta[lay_idx-1].view((*dual_vars.gs[lay_idx-1].shape[:1], *((1,) * (dual_vars.gs[lay_idx-1].dim() - 1))))
    a = dual_vars.gs[lay_idx - 1] + (dual_vars.delta_1[lay_idx - 1] - dual_vars.delta_0[lay_idx - 1])
    b = 2 * etak * primal_anchors.zt[lay_idx - 1]
    l = torch.zeros_like(primal_anchors.zt[lay_idx - 1])
    u = 2 * etak * torch.ones_like(primal_anchors.zt[lay_idx - 1])

    return optimize_diffpos_problems(a, b, l, u, 1/Hk_sqrt)


def compute_optimal_gammak(lay_idx, clbs, cubs, eta, dual_vars, primal_anchors):

    # Get AdaGrad terms (which are 1 in case we use the Eclidean norm rather than AdaGrad's)
    Gk_sqrt = primal_anchors.G_sqrt[lay_idx] if primal_anchors.G[lay_idx] is not None else 1

    etak = eta.x_eta[lay_idx].view((*dual_vars.fs[lay_idx].shape[:1], *((1,) * (dual_vars.fs[lay_idx].dim() - 1))))
    a = dual_vars.fs[lay_idx] + (dual_vars.gamma_u[lay_idx] - dual_vars.gamma_l[lay_idx])
    b = 2 * etak * primal_anchors.xt[lay_idx]
    l = 2 * etak * clbs[lay_idx].unsqueeze(1)
    u = 2 * etak * cubs[lay_idx].unsqueeze(1)

    return optimize_diffpos_problems(a, b, l, u, 1/Gk_sqrt)


def optimize_diffpos_problems(a, b, l, u, c=1):
    '''
    The problem that we're solving is:

    max_{\delta_0, \delta_1}
       -1/2 (a + (\delta_0 - \delta_1))^2 - b * (a + (\delta_0 - \delta_1)) + l \delta_0 - u \delta_1
    st. \delta_0 >= 0, \delta_1 >=0

    Input argument:
    a, b, l, u ->  All the terms are of the shape: batch_size x n_k
                   or similar with a singleton dimension if independent
    Output:
    \delta_0_star, \delta_0_star -> batch_size x n_k
    Optimal solution
    '''
    zero_sol = torch.zeros_like(a)

    delta_0_sol0 = torch.clamp((l - b) / c - a, 0, None)
    delta_1_sol0 = zero_sol

    delta_0_sol1 = zero_sol
    delta_1_sol1 = torch.clamp(a + (b - u) / c, 0, None)

    term_sol0 = (a + delta_0_sol0 - delta_1_sol0)
    sol0_val = -(c / 2) * term_sol0 * term_sol0 - b * term_sol0 + l * delta_0_sol0 - u * delta_1_sol0
    term_sol1 = (a + delta_0_sol1 - delta_1_sol1)
    sol1_val = -(c / 2) * term_sol1 * term_sol1 - b * term_sol1 + l * delta_0_sol1 - u * delta_1_sol1

    sol1_better = sol1_val > sol0_val

    delta_0 = torch.where(sol1_better, delta_0_sol1, delta_0_sol0)
    delta_1 = torch.where(sol1_better, delta_1_sol1, delta_1_sol0)

    return delta_0, delta_1


def compute_objective(primal_anchors, dual_vars, eta, weights, clbs, cubs):
    objs = 0
    for lin_k, alpha_k in zip(weights[:-1], dual_vars.alpha[1:-1]):
        objs += utils.bdot(alpha_k, lin_k.get_bias())
    # The last layer bias is positive nevertheless
    objs += utils.bdot(torch.abs(dual_vars.alpha[-1]), weights[-1].get_bias())

    for etak, f_k, xt_k, Gk_sqrt in zip(eta.x_eta, dual_vars.fs, primal_anchors.xt, primal_anchors.G_sqrt):
        Gk_sqrt = Gk_sqrt if Gk_sqrt is not None else 1
        objs -= (1 / (4*etak)) * utils.bdot(f_k, f_k/Gk_sqrt)
        objs -= utils.bdot(xt_k, f_k)

    for etak, g_k, zt_k, Hk_sqrt in zip(eta.z_eta, dual_vars.gs, primal_anchors.zt, primal_anchors.H_sqrt):
        Hk_sqrt = Hk_sqrt if Hk_sqrt is not None else 1
        objs -= (1 / (4*etak)) * utils.bdot(g_k, g_k/Hk_sqrt)
        objs -= (utils.bdot(zt_k, g_k))

    for sum_WIlbeta_k in dual_vars.sum_WIlbeta:
        objs += sum_WIlbeta_k.sum(dim=tuple(range(2, sum_WIlbeta_k.dim())))

    for (cl_k, cu_k, gamma_k_l, gamma_k_u) in zip(clbs, cubs, dual_vars.gamma_l, dual_vars.gamma_u):
        objs += utils.bdot(gamma_k_l, cl_k.unsqueeze(1))
        objs -= utils.bdot(gamma_k_u, cu_k.unsqueeze(1))

    for delta_k_1 in dual_vars.delta_1:
        objs -= delta_k_1.sum(dim=tuple(range(2, delta_k_1.dim())))

    return objs

class ProximalEta:
    """
    Class containing the values of the primal proximal weights for the Anderson relaxation.
    """
    def __init__(self, initial_eta, final_eta, eta_len, clbs, cubs, dual_vars, normalize_from_duals=False):
        """
        Initialize the proximal weights. Given an initial and a final base value between which to move linearly, the
        base values are normalized by the norm of the upper (or lower) bounds for x and z.
        :param initial_eta: initial base value for all etas
        :param final_eta: final base value for all etas
        :param eta_len: how many etas to store
        :param clbs: (clipped) lower bounds for x, list of tensors
        :param cubs: (clipped) upper bounds for x, list of tensors
        :param dual_vars: dual variables instance of class DualVars
        :param normalize_from_duals: whether to normalize on f/g rather than ubs/lbs
        """

        # TODO: allow for normal constant eta more systematically
        # TODO: remember that w/ normalization eta it was 5e-5 on cifar (and 1e0 on the tiny random nets)

        # TODO: avoid normalising over the second batch dimension as well?

        self.initial_eta = initial_eta
        self.final_eta = final_eta
        self.c_eta = initial_eta

        self.x_eta = []
        self.z_eta = []
        for x_idx in range(eta_len):
            # self.x_eta.append(initial_eta)
            f_norm = torch.norm(dual_vars.fs[x_idx].view(dual_vars.fs[x_idx].shape[0], -1), dim=-1)
            if normalize_from_duals and (f_norm > 0).all():
                self.x_eta.append(initial_eta * f_norm.unsqueeze(-1))
            else:
                ub_norm = torch.norm(cubs[x_idx].view(cubs[x_idx].shape[0], -1), dim=-1)
                lb_norm = torch.norm(clbs[x_idx].view(clbs[x_idx].shape[0], -1), dim=-1)
                self.x_eta.append(torch.where(ub_norm > 0, initial_eta / ub_norm, initial_eta / lb_norm).unsqueeze(-1))
            if x_idx > 0:
                # self.z_eta.append(initial_eta)
                g_norm = torch.norm(dual_vars.gs[x_idx-1].view(dual_vars.gs[x_idx-1].shape[0], -1), dim=-1)
                if normalize_from_duals and (g_norm > 0).all():
                    self.z_eta.append(initial_eta * g_norm.unsqueeze(-1))
                else:
                    zub_norm = torch.norm(torch.ones_like(cubs[x_idx]).view(cubs[x_idx].shape[0], -1), dim=-1)
                    self.z_eta.append(initial_eta / zub_norm.unsqueeze(-1))

    def update(self, c_steps, nb_total_steps):
        """
        Update the proximal weights according to the weight progression.
        """
        if self.final_eta is not None:
            new_eta = self.initial_eta + (c_steps / nb_total_steps) * (self.final_eta - self.initial_eta)
            self.x_eta = [cx_eta/self.c_eta * new_eta for cx_eta in self.x_eta]
            self.z_eta = [cz_eta/self.c_eta * new_eta for cz_eta in self.z_eta]
            self.c_eta = new_eta


class ExpProxOptimizationTrace(utils.ProxOptimizationTrace):
    """
    Logger for neural network bounds optimization of the anderson relaxation, associated to a single bounds computation
    done via proximal methods.
    Contains a number of dictionaries (indexed by the network layer the optimization refers to) containing quantities
    that describe the optimization.

    bounds_progress_per_layer: dictionary of lists for the evolution of the computed batch of bounds over the a subset
        of the iterations. IMPORTANT: these are the best bounds obtained so far, which are kept track of.
        These bounds might be associated to upper (stored as their negative, in the first half of the
        vector) and lower bounds.
    current_bounds_progress_per_layer: dictionary of lists for the evolution of the computed batch of bounds over the a
        subset of the iterations. IMPORTANT: these are the bounds obtained by plugging the current dual variables into
        the dual of the primal without the proximal
        These bounds might be associated to upper (stored as their negative, in the first half of the
        vector) and lower bounds.
    objs_progress_per_layer: dictionary of lists for the evolution of the computed batch of objectives over the a subset
        of the iterations. These objectives might be associated to upper (stored in the first half of the
        vector) and lower bound computations.
    time_progress_per_layer: dictionary of lists which store the elapsed time associated to each of the iterations
        logged in the lists above.
    """

    def __init__(self):
        super().__init__()
        self.current_bounds_progress_per_layer = {}

    def add_exp_proximal_point(self, layer_idx, bounds, c_bounds, objs, logging_time=None):
        # add the bounds and objective at the current optimization state, measuring time as well
        self.add_proximal_point(layer_idx, bounds, objs, logging_time=logging_time)
        if layer_idx in self.current_bounds_progress_per_layer:
            self.current_bounds_progress_per_layer[layer_idx].append(c_bounds)
        else:
            self.current_bounds_progress_per_layer[layer_idx] = [c_bounds]

    def get_last_layer_current_bounds_means_trace(self, first_half_only_as_ub=False):
        """
        Get the evolution over time of the average of the last layer current bounds.
        :param first_half_only_as_ub: assuming that the first half of the batches contains upper bounds, flip them and
            count only those in the average
        :return: list of singleton tensors
        """
        last_layer = sorted(self.bounds_progress_per_layer.keys())[-1]
        if first_half_only_as_ub:
            if self.current_bounds_progress_per_layer[last_layer][0].dim() > 1:
                bounds_trace = [-bounds[:, :int(bounds.shape[1] / 2)].mean() for bounds in
                                self.current_bounds_progress_per_layer[last_layer]]
            else:
                bounds_trace = [-bounds[:int(len(bounds) / 2)].mean() for bounds in
                                self.current_bounds_progress_per_layer[last_layer]]
        else:
            bounds_trace = [bounds.mean() for bounds in self.current_bounds_progress_per_layer[last_layer]]
        return bounds_trace


class ProxDualVars(anderson_optimization.DualVars):
    """
    Class representing the dual variables for the "dual of prox" deivation. These are
    alpha_0, alpha_1, beta (through its sufficient statistics), delta_0, delta_1, gamma_l, gamma_u and their functions
     f and g.
    The norms of alpha and beta are kept for the purposes of NNMP.
    They are stored as lists of tensors, for ReLU indices from 0 to n-1 for all variables except alpha_1.
    """

    def __init__(self, alpha, sum_beta, sum_Wp1Ibetap1, sum_W1mIubeta, sum_WIlbeta, delta_0, delta_1,
                 gamma_l, gamma_u, fs, gs):
        """
        Given the dual vars as lists of tensors (of correct length) along with their computed functions, initialize the
        class with these.
        """
        super().__init__(alpha, sum_beta, sum_Wp1Ibetap1, sum_W1mIubeta, sum_WIlbeta, fs, gs)
        self.delta_0 = delta_0
        self.delta_1 = delta_1
        self.gamma_l = gamma_l
        self.gamma_u = gamma_u

    @staticmethod
    def from_super_class(super_instance, delta_0, delta_1, gamma_l, gamma_u):
        """
        Return an instance of this class from an instance of the super class.
        """
        return ProxDualVars(super_instance.alpha, super_instance.sum_beta, super_instance.sum_Wp1Ibetap1,
                            super_instance.sum_W1mIubeta, super_instance.sum_WIlbeta, delta_0, delta_1, gamma_l,
                            gamma_u, super_instance.fs, super_instance.gs)


    @staticmethod
    def naive_initialization(weights, additional_coeffs, device, input_size):
        """
        Given parameters from the optimize function, initialize the dual vairables and their functions as all 0s except
        some special corner cases. This is equivalent to initialising with naive interval propagation bounds.
        """
        add_coeff = next(iter(additional_coeffs.values()))
        batch_size = add_coeff.shape[:2]

        base_duals = anderson_optimization.DualVars.naive_initialization(weights, additional_coeffs, device, input_size)

        delta_0 = []  # Indexed from 1 to n-1
        delta_1 = []  # Indexed from 1 to n-1
        gamma_l = []  # Indexed from 0 to n-1
        gamma_u = []  # Indexed from 0 to n-1

        # Fill in the variable holders with variables, all initiated to zero
        zero_tensor = lambda size: torch.zeros((*batch_size, *size), device=device)
        # Insert the dual variables for the box bound
        gamma_l.append(zero_tensor(input_size))
        gamma_u.append(zero_tensor(input_size))
        for lay_idx, layer in enumerate(weights[:-1]):
            nb_outputs = layer.get_output_shape(gamma_l[-1].shape)[2:]
            # Initialize the dual variables
            delta_0.append(zero_tensor(nb_outputs))
            delta_1.append(zero_tensor(nb_outputs))
            gamma_l.append(zero_tensor(nb_outputs))
            gamma_u.append(zero_tensor(nb_outputs))

        return ProxDualVars.from_super_class(base_duals, delta_0, delta_1, gamma_l, gamma_u)

    @staticmethod
    def bigm_initialization(bigm_duals, weights, clbs, cubs, lower_bounds, upper_bounds):
        """
        Given bigm dual variables, network weights, post/pre-activation lower and upper bounds,
        initialize the Anderson dual vairables and their functions to the corresponding values of the bigm duals.
        Additionally, it returns the primal variables corresponding to the inner bigm minimization with those dual
        variables.
        """
        base_duals, primals = anderson_optimization.DualVars.bigm_initialization(
            bigm_duals, weights, clbs, cubs, lower_bounds, upper_bounds)

        # deltas and gammas are initialized to 0
        delta_0 = [];
        delta_1 = [];
        gamma_l = [];
        gamma_u = []
        for lay_idx, sum_betak in enumerate(base_duals.sum_beta):
            if lay_idx > 0:
                delta_0.append(torch.zeros_like(sum_betak))
                delta_1.append(torch.zeros_like(sum_betak))
            gamma_l.append(torch.zeros_like(sum_betak))
            gamma_u.append(torch.zeros_like(sum_betak))

        return ProxDualVars.from_super_class(base_duals, delta_0, delta_1, gamma_l, gamma_u), \
               ProxPrimalVars.from_super_class(primals)

    def zero_dual_vars(self, weights, additional_coeffs):
        """
        Set all the dual variables to 0 (and treat their functions accordingly).
        """
        super().zero_dual_vars(weights, additional_coeffs)
        for tensor in itertools.chain(self.delta_0, self.delta_1, self.gamma_l, self.gamma_u):
            tensor.zero_()

    def update_f_g_from_deltak(self, lay_idx, new_delta_k_0, new_delta_k_1):
        """
        Given new values for delta at layer lay_idx, update the dual variables and their functions.
        """
        self.gs[lay_idx - 1] += ((new_delta_k_0 - self.delta_0[lay_idx - 1]) - (new_delta_k_1 - self.delta_1[lay_idx - 1]))
        self.delta_0[lay_idx - 1] = new_delta_k_0
        self.delta_1[lay_idx - 1] = new_delta_k_1

    def update_f_g_from_gammak(self, lay_idx, new_gamma_k_l, new_gamma_k_u):
        """
        Given new values for gamma at layer lay_idx, update the dual variables and their functions.
        """
        self.fs[lay_idx] -= ((new_gamma_k_u - self.gamma_u[lay_idx]) - (new_gamma_k_l - self.gamma_l[lay_idx]))
        self.gamma_l[lay_idx] = new_gamma_k_l
        self.gamma_u[lay_idx] = new_gamma_k_u

    def alphak_grad_lmo(self, lay_idx, weights, primal_anchors, eta, precision):
        """
        Given eta values (instance of ProximalEta), list of layers and primal anchor points (instance of PrimalVars),
        compute and return the linear minimization oracle of alpha_k with its gradient, the gradient itself (and some
        intermediate quantities needer for the optimal step size).
        """
        # Get AdaGrad terms (which are 1 in case we use the Eclidean norm rather than AdaGrad's)
        Gk_sqrt = primal_anchors.G_sqrt[lay_idx] if primal_anchors.G[lay_idx] is not None else 1
        Gkm1_sqrt = primal_anchors.G_sqrt[lay_idx-1] if primal_anchors.G[lay_idx-1] is not None else 1

        xk_eq = primal_anchors.xt[lay_idx] + (1 / (2 * eta.x_eta[lay_idx].view((*self.fs[lay_idx].shape[:1], *((1,) * (self.fs[lay_idx].dim() - 1)))) * Gk_sqrt)) * self.fs[lay_idx]
        xkm1_eq = primal_anchors.xt[lay_idx - 1] + (1 / (2 * eta.x_eta[lay_idx - 1].view((*self.fs[lay_idx-1].shape[:1], *((1,) * (self.fs[lay_idx-1].dim() - 1)))) * Gkm1_sqrt)) * self.fs[lay_idx - 1]

        # Compute the gradient over alphak
        grad_alphak = -xk_eq + weights[lay_idx - 1].forward(xkm1_eq)

        # Let's compute the best atom in the dictionary according to the LMO.
        # If grad_alphak > 0, lmo_alphak = 1
        lmo_alphak = (grad_alphak > 0).type(precision)
        return lmo_alphak, grad_alphak, xkm1_eq, xk_eq

    def alphak_optimal_step_size(self, lay_idx, weights, xkm1_eq, xk_eq, primal_anchors, eta, alphak_direction):
        """
        Given eta values (instance of ProximalEta), list of layers and primal anchor points (instance of PrimalVars),
        compute and return the optimal step size to take in the direction of alphak_direction (tensor of shape alpha_k)
        """
        # Get AdaGrad terms (which are 1 in case we use the Eclidean norm rather than AdaGrad's)
        Gk_sqrt = primal_anchors.G_sqrt[lay_idx] if primal_anchors.G[lay_idx] is not None else 1
        Gkm1_sqrt = primal_anchors.G_sqrt[lay_idx - 1] if primal_anchors.G[lay_idx - 1] is not None else 1

        # Let's compute the optimal step size
        # We end up with a polynomial looking like a*s^2 + b*s + cte
        Wk_atomalphak = weights[lay_idx - 1].backward(alphak_direction)

        a = (-(1 / (4 * eta.x_eta[lay_idx])) * utils.bdot(alphak_direction, alphak_direction/Gk_sqrt)
             - (1 / (4 * eta.x_eta[lay_idx - 1])) * utils.bdot(Wk_atomalphak, Wk_atomalphak/Gkm1_sqrt))
        b = (utils.bdot(alphak_direction, weights[lay_idx - 1].get_bias()) - utils.bdot(alphak_direction, xk_eq)
             + utils.bdot(Wk_atomalphak, xkm1_eq))
        # By definition, b should be positive but there might be some
        # floating point trouble
        torch.clamp(b, 0, None, out=b)

        optimal_step_size = (- b / (2 * a)).view((*alphak_direction.shape[:2], *((1,) * (alphak_direction.dim() - 2))))
        # If a==0, that means that the conditional gradient is equal to the
        # current position, so there is no need to move.
        optimal_step_size[a == 0] = 0

        return optimal_step_size

    def alphak_inner_grad(self, lay_idx, grad_alphak):
        """
        Compute the inner product of the current alphak iterate with its gradient (provided).
        """
        return utils.bdot(self.alpha[lay_idx], grad_alphak).unsqueeze(-1)


    # TODO: must introduced a big bug here: try running it without the exp varabiables first
    def betak_grad_lmo(self, lay_idx, weights, clbs, cubs, nubs, l_preacts, u_preacts, primal_anchors, eta, precision,
                       use_preactivation, preact_planet, M=1, grad_alphak=None):
        """
        Given eta values (instance of ProximalEta), list of layers, primal anchor points (instance of PrimalVars),
        pre and post activation bounds, compute and return the linear minimization oracle of beta_k with its gradient,
        and the gradient itself.
        The LMO is expressed in terms of the four sufficient statistics for beta.
        Some useful intermediate computations are returned as well.

        :param M: M is a constant with which to multiply the LMO atom.
        :param grad_alphak: the gradient of alphak for this layer. If provided, the LMO is to be performed against
            alpha as well.
        """

        # reference current layer variables via shorter names
        lin_k = weights[lay_idx - 1]
        cl_km1 = clbs[lay_idx - 1]
        cu_km1 = cubs[lay_idx - 1]
        nub_k = nubs[lay_idx - 1]
        sum_betak = self.sum_beta[lay_idx]
        xt_km1 = primal_anchors.xt[lay_idx - 1]
        xt_k = primal_anchors.xt[lay_idx]
        f_km1 = self.fs[lay_idx - 1]
        f_k = self.fs[lay_idx]
        zt_k = primal_anchors.zt[lay_idx - 1]
        g_k = self.gs[lay_idx - 1]
        l_preact = l_preacts[lay_idx].unsqueeze(1)
        u_preact = u_preacts[lay_idx].unsqueeze(1)

        # Get AdaGrad terms (which are 1 in case we use the Eclidean norm rather than AdaGrad's)
        Gk_sqrt = primal_anchors.G_sqrt[lay_idx] if primal_anchors.G[lay_idx] is not None else 1
        Gkm1_sqrt = primal_anchors.G_sqrt[lay_idx - 1] if primal_anchors.G[lay_idx - 1] is not None else 1
        Hk_sqrt = primal_anchors.H_sqrt[lay_idx - 1] if primal_anchors.H[lay_idx - 1] is not None else 1

        xeta_k = eta.x_eta[lay_idx]
        xeta_km1 = eta.x_eta[lay_idx - 1]
        zeta_k = eta.z_eta[lay_idx - 1]
        # Compute the decision matrix
        # Build l_check and u_check
        W_k = lin_k.weights
        # Build the masks of the shape batch_size x out_shape x in_shape
        eq_xkm1 = xt_km1 + (1 / (2 * xeta_km1.view((*self.fs[lay_idx-1].shape[:1], *((1,) * (self.fs[lay_idx-1].dim() - 1)))) * Gkm1_sqrt)) * f_km1
        eq_xk = xt_k + (1 / (2 * xeta_k.view((*self.fs[lay_idx].shape[:1], *((1,) * (self.fs[lay_idx].dim() - 1)))) * Gk_sqrt)) * f_k
        eq_zk = zt_k + (1 / (2 * zeta_k.view((*self.fs[lay_idx].shape[:1], *((1,) * (self.fs[lay_idx].dim() - 1)))) * Hk_sqrt)) * g_k

        if not preact_planet:
            # Anderson relaxation.

            if type(lin_k) in [utils.ConvOp, utils.BatchConvOp]:
                # Unfold the convolutional inputs into matrices containing the parts (slices) of the input forming the
                # convolution output.
                unfolded_cu_km1 = lin_k.unfold_input(cu_km1.unsqueeze(1))
                unfolded_cl_km1 = lin_k.unfold_input(cl_km1.unsqueeze(1))

                # The matrix whose matrix product with the unfolded input makes the convolutional output (after
                # reshaping to out_shape)
                unfolded_W_k = lin_k.unfold_weights()
                # u_check and l_check are now of size out_channels x slice_len x n_slices
                u_check = torch.where((unfolded_W_k > 0).unsqueeze(-1), unfolded_cu_km1, unfolded_cl_km1)
                l_check = torch.where((unfolded_W_k > 0).unsqueeze(-1), unfolded_cl_km1, unfolded_cu_km1)
                unfolded_eq_xkm1 = lin_k.unfold_input(eq_xkm1)  # input space unfolding
                unfolded_eq_zk = lin_k.unfold_output(eq_zk)  # output space unfolding
                inp_part = l_check.unsqueeze(1) - unfolded_eq_xkm1.unsqueeze(2)
                out_part = unfolded_eq_zk.unsqueeze(-2) * (u_check - l_check).unsqueeze(1)

                masked_op = anderson_optimization.MaskedConvOp(lin_k, eq_xkm1, sum_betak)
            else:
                # Fully connected layer.
                if lin_k.flatten_from_shape is not None:
                    cu_km1 = cu_km1.view(cu_km1.shape[0], -1)
                    cl_km1 = cl_km1.view(cl_km1.shape[0], -1)
                    batch_size = eq_xkm1.shape[:2]
                    eq_xkm1 = eq_xkm1.view(*batch_size, -1)
                u_check = torch.where(W_k > 0, cu_km1.unsqueeze(1), cl_km1.unsqueeze(1))
                l_check = torch.where(W_k > 0, cl_km1.unsqueeze(1), cu_km1.unsqueeze(1))
                inp_part = l_check.unsqueeze(1) - eq_xkm1.unsqueeze(2)
                out_part = eq_zk.unsqueeze(3) * (u_check - l_check).unsqueeze(1)

                if lin_k.flatten_from_shape is not None:
                    eq_xkm1 = eq_xkm1.view_as(self.sum_Wp1Ibetap1[lay_idx - 1])

                masked_op = anderson_optimization.MaskedLinearOp(lin_k)

            d = masked_op.unmasked_multiply(inp_part + out_part)
            nonnegative_d = (d >= 0)
            Istar_km1 = nonnegative_d.type(precision)
            masked_op.set_mask(Istar_km1)

            WI_eq_xkm1 = masked_op.forward(unfolded_eq_xkm1 if type(lin_k) in [utils.ConvOp, utils.BatchConvOp]
                                           else eq_xkm1, add_bias=False)
            nub_WIu = nub_k.unsqueeze(1) - masked_op.forward(u_check, bounds_matrix_in=True, add_bias=False)
            W1mIu = nub_WIu - lin_k.get_bias()
            WIl = masked_op.forward(l_check, bounds_matrix_in=True, add_bias=False)
            zeq_NUB_m_WIlu = eq_zk * (nub_WIu + WIl)
            grad = (eq_xk - WI_eq_xkm1 - zeq_NUB_m_WIlu + WIl)
            # Grad is of shape batch_size x out_shape

            if use_preactivation:
                # When using preactivation bounds we need to check whether the LMO above yields something better or worse
                # than the gradients of the dual variables of the tightened constraints (beta_0, beta_1)
                beta0_grad = eq_xk - eq_zk * u_preact
                beta1_grad = eq_xk - lin_k.forward(eq_xkm1) + (1 - eq_zk) * l_preact

                # Compute the weight mask yielding the second best atom in the LMO (scatter is an assign-based version
                # of gather, smart slicing for pytorch)
                # The second best mask has the addition of the largest d entry for the all-0 mask case, the subtraction
                # of the smallest d entry for the all-1 mask case.
                d_min_val, d_min_ind = d.min(dim=3)
                d_max_val, d_max_ind = d.max(dim=3)
                I_doubleprime = Istar_km1.scatter(3, d_min_ind.unsqueeze(3), 0)
                I_tripleprime = Istar_km1.scatter(3, d_max_ind.unsqueeze(3), 0)

                # flat_grad = lin_k.unfold_output(grad) if type(lin_k) is utils.ConvOp else grad
                I_doubleprime_grad = grad - d_min_val.view_as(grad)
                I_tripleprime_grad = grad + d_max_val.view_as(grad)

                Istar_is_all_ones = nonnegative_d.all(
                    dim=3)  # check for which batch entry and row of I_star is all ones
                Istar_is_all_zeros = ~(nonnegative_d.any(dim=3))

                # mid I represents the LMO on the beta variables that do not belong to the big-M relaxation
                mid_I = torch.where(Istar_is_all_ones.unsqueeze(3), I_doubleprime,
                                    torch.where(Istar_is_all_zeros.unsqueeze(3), I_tripleprime, Istar_km1))
                mid_I_grad = torch.where(Istar_is_all_ones.view_as(grad), I_doubleprime_grad,
                                    torch.where(Istar_is_all_zeros.view_as(grad), I_tripleprime_grad, grad))
                lmo_is_beta0 = ((beta0_grad > beta1_grad) & (beta0_grad > mid_I_grad))
                lmo_is_beta1 = (beta1_grad > beta0_grad) & (beta1_grad > mid_I_grad)
                lmo_is_mid_I = ~(lmo_is_beta0 | lmo_is_beta1)

                atom_grad = torch.where(lmo_is_beta0, beta0_grad, torch.where(lmo_is_beta1, beta1_grad, mid_I_grad))
                if grad_alphak is not None:
                    # the LMO is to be performed against alpha (do NNMP jointly on (alpha, beta)
                    lmo_is_beta = (atom_grad > grad_alphak) & (atom_grad >= 0)
                    atom_mask = M * lmo_is_beta.type(precision)
                else:
                    atom_mask = M * (atom_grad >= 0).type(precision)

                atom_I = torch.where(
                    lmo_is_beta0.view_as(d_min_val).unsqueeze(3), torch.zeros_like(Istar_km1),
                    torch.where(lmo_is_beta1.view_as(d_min_val).unsqueeze(3), torch.ones_like(Istar_km1), mid_I))
                masked_op.set_mask(atom_I)

                W1mIu = nub_k.unsqueeze(1) - masked_op.forward(u_check, bounds_matrix_in=True)
                WIl = masked_op.forward(l_check, bounds_matrix_in=True, add_bias=False)
                beta_sum_lmo = atom_mask
                beta_WI_lmo = masked_op.backward(atom_mask)
                beta_WIl_lmo = (lmo_is_mid_I.type(precision) * WIl +
                                lmo_is_beta1.type(precision) * (l_preact - lin_k.get_bias())) * atom_mask
                beta_W1mIu_lmo = (lmo_is_mid_I.type(precision) * W1mIu +
                                  lmo_is_beta0.type(precision) * (u_preact - lin_k.get_bias())) * atom_mask

            else:
                # Do not include pre-activation bounds (much simpler structure).
                atom_grad = grad
                if grad_alphak is not None:
                    # the LMO is to be performed against alpha (do NNMP jointly on (alpha, beta)
                    lmo_is_beta = (atom_grad > grad_alphak) & (atom_grad >= 0)
                    atom_mask = M * lmo_is_beta.type(precision)
                else:
                    atom_mask = M * (atom_grad >= 0).type(precision)
                beta_sum_lmo = atom_mask
                beta_WI_lmo = masked_op.backward(atom_mask)
                beta_WIl_lmo = (WIl * atom_mask)
                beta_W1mIu_lmo = (W1mIu * atom_mask)

            WI = masked_op.WI

        else:
            # preact_planet means using the PLANET relaxation (w/ pre-act bounds), instead of the Anderson one.

            # IMPORTANT: using this for non-ambiguous ReLUs has a negative impact on performance (I've tried), and for a GPU
            # implementation, it won't reduce runtime

            beta0_grad = eq_xk - eq_zk * u_preact
            beta1_grad = eq_xk - lin_k.forward(eq_xkm1) - eq_zk * l_preact + l_preact

            lmo_is_beta0 = (beta0_grad > beta1_grad)
            lmo_is_beta1 = ~lmo_is_beta0

            atom_grad = torch.where(lmo_is_beta0, beta0_grad, beta1_grad)
            if grad_alphak is not None:
                # the LMO is to be performed against alpha (do NNMP jointly on (alpha, beta)
                lmo_is_beta = (atom_grad > grad_alphak) & (atom_grad >= 0)
                atom_mask = M * lmo_is_beta.type(precision)
            else:
                atom_mask = M * (atom_grad >= 0).type(precision)
            beta_sum_lmo = atom_mask
            beta_WI_lmo = lin_k.backward(atom_mask * lmo_is_beta1.type(precision))
            beta_WIl_lmo = lmo_is_beta1.type(precision) * (l_preact - lin_k.get_bias()) * atom_mask
            beta_W1mIu_lmo = lmo_is_beta0.type(precision) * (u_preact - lin_k.get_bias()) * atom_mask

            # TODO: this is used for FW only, which could be adapted to take lmo_is_beta1 to do the backward for
            #  beta_WI_lmo more efficiently (but this part of the code is not used, anyways)
            WI = None

        return beta_sum_lmo, beta_WI_lmo, beta_WIl_lmo, beta_W1mIu_lmo, atom_grad, WI, eq_xkm1, eq_xk, eq_zk

    @staticmethod
    def betak_optimal_step_size(lay_idx, weights, eq_xkm1, eq_xk, eq_zk, primal_anchors, eta, beta_sum_direction,
                                beta_WI_direction, beta_WIl_direction, beta_W1mIu_direction):
        """
        Given eta values (instance of ProximalEta), list of layers and primal anchor points (instance of PrimalVars),
        compute and return the optimal step size to take in the direction indicated by beta's sufficient statistics.
        """

        # Get AdaGrad terms (which are 1 in case we use the Eclidean norm rather than AdaGrad's)
        Gk_sqrt = primal_anchors.G_sqrt[lay_idx] if primal_anchors.G[lay_idx] is not None else 1
        Gkm1_sqrt = primal_anchors.G_sqrt[lay_idx - 1] if primal_anchors.G[lay_idx - 1] is not None else 1
        Hk_sqrt = primal_anchors.H_sqrt[lay_idx - 1] if primal_anchors.H[lay_idx - 1] is not None else 1

        # Let's compute the optimal step size:
        # We end up with a polynomial looking like a*s^2 + b*s + cte
        atom_gbeta = weights[lay_idx - 1].get_bias() * beta_sum_direction + beta_W1mIu_direction + beta_WIl_direction

        a = -(1 / (4 * eta.x_eta[lay_idx])) * utils.bdot(beta_sum_direction, beta_sum_direction/Gk_sqrt) - \
            (1 / (4 * eta.z_eta[lay_idx - 1])) * utils.bdot(atom_gbeta, atom_gbeta/Hk_sqrt) - \
            (1 / (4 * eta.x_eta[lay_idx - 1])) * utils.bdot(beta_WI_direction, beta_WI_direction/Gkm1_sqrt)
        b = (utils.bdot(beta_sum_direction, eq_xk)
             - utils.bdot(beta_WI_direction, eq_xkm1)
             - utils.bdot(atom_gbeta, eq_zk)
             + beta_WIl_direction.sum(dim=tuple(range(2, beta_WIl_direction.dim()))))
        # By definition, b should be positive but there might be some
        # floating point trouble
        torch.clamp(b, 0, None, out=b)

        optimal_step_size = (-b/(2 * a)).view((*beta_sum_direction.shape[:2], *((1,) * (beta_sum_direction.dim() - 2))))
        # If a==0, that means that the conditional gradient is equal to the
        # current position, so there is no need to move.
        optimal_step_size[a == 0] = 0

        return optimal_step_size

    def betak_inner_grad(self, lay_idx, weights, eq_xkm1, eq_xk, eq_zk):
        """
        Given some useful intermediate computations, compute the inner product of the current betak iterate with its
        gradient.
        """
        inner_beta_sum = (self.sum_beta[lay_idx] * (eq_xk - weights[lay_idx - 1].get_bias() * eq_zk)).\
            sum(dim=tuple(range(2, self.sum_beta[lay_idx].dim())))
        inner_WI = - (self.sum_Wp1Ibetap1[lay_idx - 1] * eq_xkm1).\
            sum(dim=tuple(range(2, self.sum_Wp1Ibetap1[lay_idx - 1].dim())))
        inner_WIl = ((1 - eq_zk) * self.sum_WIlbeta[lay_idx]).\
            sum(dim=tuple(range(2, self.sum_WIlbeta[lay_idx].dim())))
        inner_W1mIu = (-eq_zk * self.sum_W1mIubeta[lay_idx]).\
            sum(dim=tuple(range(2, self.sum_W1mIubeta[lay_idx].dim())))
        return (inner_beta_sum + inner_WI + inner_WIl + inner_W1mIu).unsqueeze(-1)

    @staticmethod
    def alphak_betak_optimal_step_size(lay_idx, weights, eq_xkm1, eq_xk, eq_zk, primal_anchors, eta, alphak_direction,
                                       beta_sum_direction, beta_WI_direction, beta_WIl_direction, beta_W1mIu_direction):
        """
        Given eta values (instance of ProximalEta), list of layers and primal anchor points (instance of PrimalVars),
        compute and return the optimal step size to take in the direction indicated by betak's sufficient statistics and
        directly provided for alphak.
        """

        # Get AdaGrad terms (which are 1 in case we use the Eclidean norm rather than AdaGrad's)
        Gk_sqrt = primal_anchors.G_sqrt[lay_idx] if primal_anchors.G[lay_idx] is not None else 1
        Gkm1_sqrt = primal_anchors.G_sqrt[lay_idx - 1] if primal_anchors.G[lay_idx - 1] is not None else 1
        Hk_sqrt = primal_anchors.H_sqrt[lay_idx - 1] if primal_anchors.H[lay_idx - 1] is not None else 1

        # Let's compute the optimal step size:
        # We end up with a polynomial looking like a*s^2 + b*s + cte
        Wk_atomalphak = weights[lay_idx - 1].backward(alphak_direction)
        atom_gbeta = weights[lay_idx - 1].get_bias() * beta_sum_direction + beta_W1mIu_direction + beta_WIl_direction

        xk_dual_diff = alphak_direction - beta_sum_direction
        xkm1_dual_diff = beta_WI_direction - Wk_atomalphak
        a = - (1 / (4 * eta.x_eta[lay_idx])) * utils.bdot(xk_dual_diff, xk_dual_diff/Gk_sqrt) \
            - (1 / (4 * eta.z_eta[lay_idx - 1])) * utils.bdot(atom_gbeta, atom_gbeta/Hk_sqrt) \
            - (1 / (4 * eta.x_eta[lay_idx - 1])) * utils.bdot(xkm1_dual_diff, xkm1_dual_diff/Gkm1_sqrt)
        b = (utils.bdot(alphak_direction, weights[lay_idx - 1].get_bias())
             + utils.bdot(beta_sum_direction - alphak_direction, eq_xk)
             + utils.bdot(Wk_atomalphak - beta_WI_direction, eq_xkm1)
             - utils.bdot(atom_gbeta, eq_zk)
             + beta_WIl_direction.sum(dim=tuple(range(2, beta_WIl_direction.dim()))))
        # By definition, b should be positive but there might be some
        # floating point trouble
        torch.clamp(b, 0, None, out=b)

        optimal_step_size = (-b / (2 * a)).view(
            (*beta_sum_direction.shape[:2], *((1,) * (beta_sum_direction.dim() - 2))))
        # If a==0, that means that the conditional gradient is equal to the
        # current position, so there is no need to move.
        optimal_step_size[a == 0] = 0

        return optimal_step_size


class ProxPrimalVars(anderson_optimization.PrimalVars):
    """
    Class representing the primal variables xt, zt for the "dual of prox" derivation.
    They are stored as lists of tensors, for ReLU indices from 0 to n-1 for xt and 1 to n-1 for zt.
    """

    def __init__(self, xt, zt):
        """
        Given the primal vars as lists of tensors (of correct length), initialize the class with these.
        """
        super().__init__(xt, zt)

        # store G and H matrices for AdaGrad
        self.G = [None] * len(xt)
        self.G_sqrt = [None] * len(xt)
        self.H = [None] * len(zt)
        self.H_sqrt = [None] * len(zt)

    @staticmethod
    def from_super_class(super_instance):
        """
        Return an instance of this class from an instance of the super class.
        """
        return ProxPrimalVars(super_instance.xt, super_instance.zt)

    @staticmethod
    def mid_box_initialization(dual_vars, clbs, cubs):
        """
        Initialize the primal variables (anchor points) to the mid-point of the box constraints (halfway through each
        variable's lower and upper bounds).
        """
        primals = anderson_optimization.PrimalVars.mid_box_initialization(dual_vars, clbs, cubs)
        return ProxPrimalVars.from_super_class(primals)

    def update_anchors(self, eta, dual_vars, clbs, cubs, project_anchors=False, adagrad=False):
        """
        Given the dual variables (list of tensors of shape
        opt_layer_width x *layer_shape), intermediate bounds (clbs, cubs) (as lists of tensors of *layer_shape) and the
        weight of the proximal terms, update the proximal primal anchor points.
        :param project_anchors: whether to project the anchors within the primal bounds (don't)
        :param adagrad: whether to use AdaGrad's norm on the proximal weights NOTE: performs worse on CIFAR-10
        :return: the updated proximal primal anchor points (lists of tensors of shape opt_layer_width x layer shape)
        """
        if adagrad:
            epsilon = 1e-8  # avoid division by zero
            # use AdaGrad's Mahalanobis' norm for proximal terms
            for lay_idx, layer in enumerate(dual_vars.sum_beta):
                # Update the (diagonal) AdaGrad matrices.
                if self.G[lay_idx] is not None:
                    self.G[lay_idx] += torch.pow(dual_vars.fs[lay_idx], 2)
                else:
                    self.G[lay_idx] = torch.pow(dual_vars.fs[lay_idx], 2)
                self.G_sqrt[lay_idx] = torch.sqrt(self.G[lay_idx] + epsilon)
                # Update the dual variables.
                self.xt[lay_idx] += (1 / (2 * eta.x_eta[lay_idx] * self.G_sqrt[lay_idx])) * dual_vars.fs[lay_idx]

                if lay_idx > 0:
                    # Update the (diagonal) AdaGrad matrices.
                    if self.H[lay_idx-1] is not None:
                        self.H[lay_idx-1] += torch.pow(dual_vars.gs[lay_idx-1], 2)
                    else:
                        self.H[lay_idx-1] = torch.pow(dual_vars.gs[lay_idx-1], 2)
                    self.H_sqrt[lay_idx-1] = torch.sqrt(self.H[lay_idx-1] + epsilon)
                    # Update the dual variables.
                    self.zt[lay_idx-1] += (1 / (2 * eta.z_eta[lay_idx-1] * self.H_sqrt[lay_idx-1])) * \
                        dual_vars.gs[lay_idx-1]
        else:
            # use Euclidean norm for proximal terms
            self.xt = [xt_k + (1 / (2 * etak.view((*f_k.shape[:1], *((1,) * (f_k.dim() - 1)))))) * f_k for etak, xt_k, f_k in zip(eta.x_eta, self.xt, dual_vars.fs)]
            self.zt = [zt_k + (1 / (2 * etak.view((*g_k.shape[:1], *((1,) * (g_k.dim() - 1)))))) * g_k for etak, zt_k, g_k in zip(eta.z_eta, self.zt, dual_vars.gs)]

        # theoretically, this is unnecessary
        if project_anchors:
            # Ensure that the anchors doesn't go beyond the feasible values for xt and zt
            # without this clipping it might indeed go out of the bounds
            for nxt_k, cu_k, cl_k in zip(self.xt, cubs, clbs):
                torch.max(nxt_k, cl_k.unsqueeze(1), out=nxt_k)
                torch.min(nxt_k, cu_k.unsqueeze(1), out=nxt_k)
            for nzt_k in self.zt:
                torch.clamp(nzt_k, 0, 1, out=nzt_k)
