import torch
import copy

from plnn.simplex_solver.utils import bdot, simplex_projection_sort
import math
import torch.nn.functional as F
import time


class AutoLirpa():
    """
    This class implements the autolirpa method using backward fashion in primal space.
    """

    def __init__(self, weights, additional_coeffs, lower_bounds, upper_bounds):
        """
        The object stores the lirpa coefficients lower_a and upper_a corresponding to the upper and lower bounds.
        """

        self.learning_rate_l = 100
        self.learning_rate_u = 1e8

        self.lower_a = []
        self.upper_a = []
        for i in range(len(weights)):
            self.lower_a.append(torch.ones_like(lower_bounds[i], requires_grad=True))
            self.lower_a[i].grad = None

            self.upper_a.append(torch.ones_like(upper_bounds[i], requires_grad=True))
            self.upper_a[i].grad = None

        #######################
        assert len(additional_coeffs) > 0

        final_lay_idx = len(weights)
        if final_lay_idx in additional_coeffs:
            # There is a coefficient on the output of the network
            rho = additional_coeffs[final_lay_idx]
            lay_idx = final_lay_idx
        else:
            # There is none. Just identify the shape from the additional coeffs
            add_coeff = next(iter(additional_coeffs.values()))
            batch_size = add_coeff.shape[:2]
            device = lower_bounds[-1].device

            lay_idx = final_lay_idx -1
            while lay_idx not in additional_coeffs:
                lay_shape = lower_bounds[lay_idx].shape[1:]
                lay_idx -= 1
            # We now reached the time where lay_idx has an additional coefficient
            rho = additional_coeffs[lay_idx]
        lay_idx -= 1

        self.initial_lay_idx = copy.deepcopy(lay_idx)
        self.initial_rho = copy.deepcopy(rho)

        #######################

    def crown_initialization(self, weights, additional_coeffs, lower_bounds, upper_bounds):
        """
        initialized the lower coefficients as per crown
        """
        for i in range(len(weights)):
            self.lower_a[i] = (upper_bounds[i] >= torch.abs(lower_bounds[i])).type(lower_bounds[i].dtype)
            self.lower_a[i].requires_grad = True
            self.lower_a[i].grad = None

    def get_bound_lirpa_backward(self, weights, additional_coeffs, lower_bounds, upper_bounds):
        """
        This function is used to do a lirpa backward pass and get the bounds with current coefficients lower_a and upper_a.
        """
        b_term = None
        rho = copy.deepcopy(self.initial_rho)
        lay_idx = copy.deepcopy(self.initial_lay_idx)
        rho_split=False

        with torch.enable_grad():
            while lay_idx > 0:
                lay = weights[lay_idx]

                if b_term is None:
                    b_term = lay.bias_backward(rho)#rho is of size (batch_size*output_size)
                else:
                    b_term += lay.bias_backward(rho)#rho is of size (batch_size*output_size)

                lbda = lay.backward(rho)
                
                lbs = lower_bounds[lay_idx]#this is of input_size as they are lower_bounds on that layer input
                ubs = upper_bounds[lay_idx]

                las = self.lower_a[lay_idx]
                uas = self.upper_a[lay_idx]

                #####
                ## beta
                beta_u = - (lbs*ubs) / (ubs - lbs)
                beta_u.masked_fill_(lbs > 0, 0)
                beta_u.masked_fill_(ubs <= 0, 0)

                beta_l = torch.zeros_like(lbs)

                ### POSSIBLE SPEEDUP
                #### this can be implemented as a convolution
                b_term += bdot(torch.where(lbda >= 0, beta_l.unsqueeze(1), beta_u.unsqueeze(1)), lbda)
                #####


                #####
                ## alpha
                alpha_u = ubs / (ubs - lbs)
                alpha_u.masked_fill_(lbs > 0, 1)
                alpha_u.masked_fill_(ubs <= 0, 0)

                # alpha_l = (ubs >= torch.abs(lbs)).type(lbs.dtype)
                alpha_l = las
                with torch.no_grad():
                    alpha_l.masked_fill_(lbs > 0, 1)
                    alpha_l.masked_fill_(ubs <= 0, 0)

                ones_ten = torch.ones_like(ubs)
                zeros_ten = torch.zeros_like(ubs)

                ## lirpa
                rho = torch.where(lbda >= 0, alpha_l.unsqueeze(1), alpha_u.unsqueeze(1)) * lbda#(output(batch_size modulo)*input shape)

                ## kw
                # rho = alpha_u.unsqueeze(1) * lbda#(output(batch_size modulo)*input shape)
                #####

                lay_idx -= 1

            ##########################
            #### compute objective
            ##########################
            bound = opt_lirpa_input_simplex(weights[0], b_term, rho)

            return bound

    def get_bound_dp_lirpa_backward(self, weights, additional_coeffs, lower_bounds, upper_bounds):
        """
        This function is used to do a dp lirpa backward pass and get the bounds with current coefficients lower_a and upper_a.
        """

        ##################
        ### compute L(x, a)
        ##################
        b_term = None
        rho = copy.deepcopy(self.initial_rho)
        lay_idx = copy.deepcopy(self.initial_lay_idx)
        rho_split=False

        with torch.enable_grad():
            while lay_idx > 0:  
                lay = weights[lay_idx]

                if b_term is None:
                    b_term = lay.bias_backward(rho)#rho is of size (batch_size*output_size)
                else:
                    b_term += lay.dp_bias_backward(rho_u_dp) + lay.bias_backward(rho_planet)

                if not rho_split:
                    lbda = lay.backward(rho)#rho is of size (batch_size*output_size)
                else:
                    lbda = lay.dp_backward(rho_u_dp) + lay.backward(rho_planet)
                
                lbs = lower_bounds[lay_idx]#this is of input_size as they are lower_bounds on that layer input
                ubs = upper_bounds[lay_idx]

                las = self.lower_a[lay_idx]
                uas = self.upper_a[lay_idx]

                #####
                ## beta
                beta_u = -  (uas * (lbs*ubs)) / (ubs - lbs)
                beta_u.masked_fill_(lbs > 0, 0)
                beta_u.masked_fill_(ubs <= 0, 0)

                beta_l = torch.zeros_like(lbs)

                ### POSSIBLE SPEEDUP
                #### this can be implemented as a convolution
                b_term += bdot(torch.where(lbda >= 0, beta_l.unsqueeze(1), beta_u.unsqueeze(1)), lbda)
                #####


                #####
                ## alpha
                alpha_u_planet = uas * ubs / (ubs - lbs)
                alpha_u_planet.masked_fill_(lbs > 0, 1)
                alpha_u_planet.masked_fill_(ubs <= 0, 0)

                alpha_u_dp = 1-uas
                alpha_u_dp.masked_fill_(lbs > 0, 0)
                alpha_u_dp.masked_fill_(ubs <= 0, 0)

                alpha_l = las
                with torch.no_grad():
                    alpha_l.masked_fill_(lbs > 0, 1)
                    alpha_l.masked_fill_(ubs <= 0, 0)

                ones_ten = torch.ones_like(ubs)
                zeros_ten = torch.zeros_like(ubs)

                rho_split=True
                start_time=time.time()
                rho_planet = torch.where(lbda >= 0, alpha_l.unsqueeze(1), alpha_u_planet.unsqueeze(1)) * lbda#(output(batch_size modulo)*input shape)
                rho_u_dp = torch.where(lbda >= 0, zeros_ten.unsqueeze(1), alpha_u_dp.unsqueeze(1)) * lbda
                # rho = torch.where(lbda >= 0, alpha_l.unsqueeze(1), alpha_u.unsqueeze(1)) * lbda#(output(batch_size modulo)*input shape)
                # rho = alpha_u.unsqueeze(1) * lbda#(output(batch_size modulo)*input shape)
                #####

                lay_idx -= 1

            bound = opt_lirpa_input_dp(weights[0], b_term, rho_planet, rho_u_dp, lower_bounds[0])

        return bound

    def get_gradients(self, bound):
        """
        This functions computes the gradients of L wrt a's (Compute dL/da)
        """
        with torch.enable_grad():
            bound_mean = bound.mean()
        bound_mean.backward()

    def pgd_update(self):
        """
        projected gradient descent optimizer with constant stepsize
        """
        with torch.no_grad():
            #### lower_a
            for i in range(1, len(self.lower_a)):
                if self.lower_a[i].grad is not None:
                    self.lower_a[i] += self.learning_rate_l * self.lower_a[i].grad
                    self.lower_a[i] = torch.clamp(self.lower_a[i], 0, 1)
                    self.lower_a[i].requires_grad = True
                    # Manually zero the gradients after updating weights
                    self.lower_a[i].grad = None

            #### upper_a
            for i in range(1, len(self.upper_a)):
                if self.upper_a[i].grad is not None:
                    self.upper_a[i] += self.learning_rate_u * self.upper_a[i].grad
                    self.upper_a[i] = torch.clamp(self.upper_a[i], 0, 1)
                    self.upper_a[i].requires_grad = True
                    # Manually zero the gradients after updating weights
                    self.upper_a[i].grad = None

    def auto_lirpa_optimizer(self, weights, additional_coeffs, lower_bounds, upper_bounds, dp=False, logger=None, opt_args=None):
        """
        # 1. Compute L(x, a)
        # 2. Compute dL/da 
        # 3. optimize a's using pgd.
        """
        ## for sgd net
        default_opt_args = {
                'nb_iter': 20,
                'lower_initial_step_size': 0.0001,
                'lower_final_step_size': 1,
                'upper_initial_step_size': 1e2,
                'upper_final_step_size': 1e3,
                'betas': (0.9, 0.999)
            }


        if opt_args is None:
            opt_args=default_opt_args

        adam_stats = LirpaADAMStats(self.lower_a, beta1=opt_args['betas'][0], beta2=opt_args['betas'][1])
        lower_init_step_size = opt_args['lower_initial_step_size']
        lower_final_step_size = opt_args['lower_final_step_size']
        upper_init_step_size = opt_args['upper_initial_step_size']
        upper_final_step_size = opt_args['upper_final_step_size']
        n_iters = opt_args["nb_iter"]

        for it_no in range(n_iters):

            # 1. Compute L(x, a)
            if dp:
                bound = self.get_bound_dp_lirpa_backward(weights, additional_coeffs, lower_bounds, upper_bounds)
            else:
                bound = self.get_bound_lirpa_backward(weights, additional_coeffs, lower_bounds, upper_bounds)

            # 2. Compute dL/da 
            self.get_gradients(bound)

            # 3. optimize a's using pgd.

            # normal subgradient ascent
            # self.pgd_update()

            # do adam for subgradient ascent
            lower_step_size = lower_init_step_size + ((it_no + 1) / n_iters) * (lower_final_step_size - lower_init_step_size)
            upper_step_size = upper_init_step_size + ((it_no + 1) / n_iters) * (upper_final_step_size - upper_init_step_size)

            adam_stats.update_moments_take_projected_step(lower_step_size, upper_step_size, it_no, self.lower_a, self.upper_a)

        return bound

class LirpaADAMStats:
    """
    class storing (and containing operations for) the ADAM statistics for the lirpa coefficients.
    they are stored as lists of tensors, for ReLU indices from 0 to n-1.
    """
    def __init__(self, lower_a, beta1=0.9, beta2=0.999):
        """
        Given lower_a to copy the dimensionality from, initialize all ADAM stats to 0 tensors.
        """
        # first moments
        self.m1_lower_a = []
        self.m1_upper_a = []
        # second moments
        self.m2_lower_a = []
        self.m2_upper_a = []

        for lay_idx in range(len(lower_a)):
            self.m1_lower_a.append(torch.zeros_like(lower_a[lay_idx]))
            self.m1_upper_a.append(torch.zeros_like(lower_a[lay_idx]))

            self.m2_lower_a.append(torch.zeros_like(lower_a[lay_idx]))
            self.m2_upper_a.append(torch.zeros_like(lower_a[lay_idx]))

        self.coeff1 = beta1
        self.coeff2 = beta2
        self.epsilon = 1e-8

    def update_moments_take_projected_step(self, lower_step_size, upper_step_size, it_no, lower_a, upper_a):
        """
        Update the ADAM moments given the subgradients, and normal gd step size, then take the projected step from
        lirpa coefficients.
        Update performed in place on lirpa coefficients.
        """

        bias_correc1 = 1 - self.coeff1 ** (it_no + 1)
        bias_correc2 = 1 - self.coeff2 ** (it_no + 1)
        lower_corrected_step_size = lower_step_size * math.sqrt(bias_correc2) / bias_correc1
        upper_corrected_step_size = upper_step_size * math.sqrt(bias_correc2) / bias_correc1

        for lay_idx in range(1, len(lower_a)):
            if lower_a[lay_idx].grad is not None:
                # Update the ADAM moments.
                self.m1_lower_a[lay_idx].mul_(self.coeff1).add_(lower_a[lay_idx].grad, alpha=1-self.coeff1)
                self.m2_lower_a[lay_idx].mul_(self.coeff2).addcmul_(lower_a[lay_idx].grad, lower_a[lay_idx].grad, value=1-self.coeff2)
                
                # Take the projected (between 0 and 1) step.
                lower_a_step_size = self.m1_lower_a[lay_idx] / (self.m2_lower_a[lay_idx].sqrt() + self.epsilon)
                lower_a[lay_idx] = torch.clamp(lower_a[lay_idx] + lower_corrected_step_size * lower_a_step_size, 0, 1)
                lower_a[lay_idx].requires_grad = True
                # Manually zero the gradients after updating weights
                lower_a[lay_idx].grad = None

        for lay_idx in range(1, len(upper_a)):
            if upper_a[lay_idx].grad is not None:
                # Update the ADAM moments.
                self.m1_upper_a[lay_idx].mul_(self.coeff1).add_(upper_a[lay_idx].grad, alpha=1-self.coeff1)
                self.m2_upper_a[lay_idx].mul_(self.coeff2).addcmul_(upper_a[lay_idx].grad, upper_a[lay_idx].grad, value=1-self.coeff2)

                # Take the projected (between 0 and 1) step.
                upper_a_step_size = self.m1_upper_a[lay_idx] / (self.m2_upper_a[lay_idx].sqrt() + self.epsilon)
                upper_a[lay_idx] = torch.clamp(upper_a[lay_idx] + upper_corrected_step_size * upper_a_step_size, 0, 1)
                upper_a[lay_idx].requires_grad = True
                # Manually zero the gradients after updating weights
                upper_a[lay_idx].grad = None


def autolirpa_opt_dp(weights, additional_coeffs, lower_bounds, upper_bounds):
    '''
    1. The upper bound has both planet and dp constraint.
    2. The lower bound has both planet constraints(function and 0).
    3. There are 2 weighting factors involved (la and ua).
    4. Input lies within a simplex.
    5. This function also optimizes the multi-neuron cut coefficients lambda.

    Optimization: this function will optimize as and ib both to get optimized lirpa bounds for this case.

    Return: this will return the upper and lower bounds
    '''
    # 1. Compute L(x, a, \lambda)
    # 2. Compute dL/da and dL/d\lambda 
    # 3. optimize over a and lambda using pgd.

    ### make the a weight tensor for all layers at once.
    ### and set their requires_grad to true.
    learning_rate_l = 1e3
    learning_rate_u = 1e3
    learning_rate_lmbd = 1e2
    lower_a = []
    upper_a = []
    lambdas = []

    for i in range(len(weights)):

        lower_a.append((upper_bounds[i] >= torch.abs(lower_bounds[i])).type(lower_bounds[i].dtype))
        lower_a[i].requires_grad = True
        upper_a.append(torch.ones_like(upper_bounds[i], requires_grad=True))## this initializes without the dp cut

        lambdas.append(torch.ones_like(upper_bounds[i].squeeze(0), requires_grad=True))

        ## empty grads at beginning
        lower_a[i].grad = None
        upper_a[i].grad = None
        lambdas[i].grad = None

    #######################
    assert len(additional_coeffs) > 0

    final_lay_idx = len(weights)
    if final_lay_idx in additional_coeffs:
        # There is a coefficient on the output of the network
        rho = additional_coeffs[final_lay_idx]
        lay_idx = final_lay_idx
    else:
        # There is none. Just identify the shape from the additional coeffs
        add_coeff = next(iter(additional_coeffs.values()))
        batch_size = add_coeff.shape[:2]
        device = lower_bounds[-1].device

        lay_idx = final_lay_idx -1
        while lay_idx not in additional_coeffs:
            lay_shape = lower_bounds[lay_idx].shape[1:]
            lay_idx -= 1
        # We now reached the time where lay_idx has an additional coefficient
        rho = additional_coeffs[lay_idx]
    lay_idx -= 1

    initial_lay_idx = copy.deepcopy(lay_idx)
    initial_rho = copy.deepcopy(rho)

    #######################

    for it in range(50):
        # print('Iteration number: ', it)
        ##################
        ### compute L(x, a)
        ##################
        b_term = None
        rho = copy.deepcopy(initial_rho)
        lay_idx = copy.deepcopy(initial_lay_idx)
        rho_split=False

        with torch.enable_grad():
            while lay_idx > 0:  
                lay = weights[lay_idx]

                #########
                ###### for lambdas
                # stopping gradient accumulation
                lambdas[lay_idx].grad = None

                lay.update_dp_weights(lambdas[lay_idx])
                #########


                if b_term is None:
                    b_term = lay.bias_backward(rho)#rho is of size (batch_size*output_size)
                else:
                    b_term += lay.dp_bias_backward(rho_u_dp) + lay.bias_backward(rho_planet)

                if not rho_split:
                    lbda = lay.backward(rho)#rho is of size (batch_size*output_size)
                else:
                    lbda = lay.dp_backward(rho_u_dp) + lay.backward(rho_planet)
                
                lbs = lower_bounds[lay_idx]#this is of input_size as they are lower_bounds on that layer input
                ubs = upper_bounds[lay_idx]

                las = lower_a[lay_idx]
                uas = upper_a[lay_idx]

                #####
                ## beta
                beta_u = -  (uas * (lbs*ubs)) / (ubs - lbs)
                beta_u.masked_fill_(lbs > 0, 0)
                beta_u.masked_fill_(ubs <= 0, 0)

                beta_l = torch.zeros_like(lbs)

                ### POSSIBLE SPEEDUP
                #### this can be implemented as a convolution
                b_term += bdot(torch.where(lbda >= 0, beta_l.unsqueeze(1), beta_u.unsqueeze(1)), lbda)
                # if lbda.dim() == 5:
                #     b_term += torch.sum(torch.where(lbda >= 0, beta_l.unsqueeze(1), beta_u.unsqueeze(1))*lbda, dim=(-3,-2,-1))
                # else:
                #     b_term += torch.sum(torch.where(lbda >= 0, beta_l.unsqueeze(1), beta_u.unsqueeze(1))*lbda, dim=(-1))
                #####


                #####
                ## alpha
                alpha_u_planet = uas * ubs / (ubs - lbs)
                alpha_u_planet.masked_fill_(lbs > 0, 1)
                alpha_u_planet.masked_fill_(ubs <= 0, 0)

                alpha_u_dp = 1-uas
                alpha_u_dp.masked_fill_(lbs > 0, 0)
                alpha_u_dp.masked_fill_(ubs <= 0, 0)

                alpha_l = las
                with torch.no_grad():
                    alpha_l.masked_fill_(lbs > 0, 1)
                    alpha_l.masked_fill_(ubs <= 0, 0)

                ones_ten = torch.ones_like(ubs)
                zeros_ten = torch.zeros_like(ubs)

                rho_split=True
                rho_planet = torch.where(lbda >= 0, alpha_l.unsqueeze(1), alpha_u_planet.unsqueeze(1)) * lbda#(output(batch_size modulo)*input shape)
                rho_u_dp = torch.where(lbda >= 0, zeros_ten.unsqueeze(1), alpha_u_dp.unsqueeze(1)) * lbda

                # rho = torch.where(lbda >= 0, alpha_l.unsqueeze(1), alpha_u.unsqueeze(1)) * lbda#(output(batch_size modulo)*input shape)
                # rho = alpha_u.unsqueeze(1) * lbda#(output(batch_size modulo)*input shape)
                #####

                lay_idx -= 1

            ##########################
            #### compute objective
            # print(weights[1].dp_weights, b_term, rho_planet, rho_u_dp)
            bound = opt_lirpa_input_dp(weights[0], b_term, rho_planet, rho_u_dp, lower_bounds[0])
            bound_mean = bound.mean()
            # print(it, bound_mean.item())
            ##########################

            #######################################
            ###### Updating a's and lambda's ######
            #######################################
            # 1. compute gradients
            bound_mean.backward()

            # 2. update step
            with torch.no_grad():
                for i in range(1, len(lower_a)):

                    #### lower_a
                    lower_a[i] += learning_rate_l * lower_a[i].grad
                    lower_a[i] = torch.clamp(lower_a[i], 0, 1)
                    lower_a[i].requires_grad = True
                    # Manually zero the gradients after updating weights
                    lower_a[i].grad = None

                    #### upper_a
                    upper_a[i] += learning_rate_u * upper_a[i].grad
                    upper_a[i] = torch.clamp(upper_a[i], 0, 1)
                    upper_a[i].requires_grad = True
                    # Manually zero the gradients after updating weights
                    upper_a[i].grad = None

                    #######################################
                    ########## UPDATING LAMBDAS ###########
                    #######################################
                    '''
                    We don't work with lambdas of input layer and the last linear layer.
                    '''
                    if it%5==0 and i>1:
                        # update lambda
                        lambda_temp = lambdas[i-1]

                        lambda_temp += learning_rate_lmbd * lambdas[i-1].grad
                        # print(lambda_temp)
                        lambda_temp = torch.clamp(lambda_temp, 0, 1)
                        # lambda_temp = simplex_projection_sort(lambda_temp.unsqueeze(0)).squeeze(0)
                        # print(lambda_temp)
                        # input('')
                        # get alpha
                        lay_in = weights[i-2]
                        init_cut_coeff = lay_in.simplex_conditioning(lambda_temp)

                        # divide lambda by alpha
                        lambda_temp = lambda_temp/init_cut_coeff
                        # print(lambda_temp)
                        # input('')
                        lambdas[i-1] = lambda_temp

                        # Manually zero the gradients after updating weights
                        lambdas[i-1].grad = None
                        lambdas[i-1].requires_grad = True

                        # update the dp-weights
                        lay_out = weights[i-1]
                        
                        with torch.enable_grad():
                            lay_out.update_dp_weights(lambda_temp)


                    #######################################
                    #######################################

            #######################################
            #######################################

    
    return bound


def opt_lirpa_input_simplex(lay, b_term, rho):
    with torch.enable_grad():
        b_term += lay.bias_backward(rho)
        lin_eq = lay.backward(rho)

        lin_eq_matrix = lin_eq.view(lin_eq.shape[0],lin_eq.shape[1],-1)

        (b,c) = torch.min(lin_eq_matrix, 2)
        bound = b_term + torch.clamp(b, None, 0)

    return bound 

def opt_lirpa_input_dp(lay, b_term, rho_planet, rho_u_dp, inp_bound=None):

    with torch.enable_grad():
        b_term += lay.dp_bias_backward(rho_u_dp) + lay.bias_backward(rho_planet)

        start_time = time.time()
        lin_eq = lay.dp_backward(rho_u_dp, inp_bound)
        # print('dp backward time: ', time.time()-start_time)
        
        start_time = time.time()
        lin_eq += lay.backward(rho_planet)
        # print('backward time: ', time.time()-start_time)


        lin_eq_matrix = lin_eq.view(lin_eq.shape[0],lin_eq.shape[1],-1)

        (b,c) = torch.min(lin_eq_matrix, 2)
        bound = b_term + torch.clamp(b, None, 0)

    return bound 
