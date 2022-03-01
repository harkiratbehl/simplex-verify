from torch import nn 
import torch
from plnn.modules import View, Flatten
from torch.nn.parameter import Parameter
from plnn.model import simplify_network
import random
import copy
import json
from tools.colt_layers import Conv2d, Normalization, ReLU, Linear, Sequential
from tools.colt_layers import Flatten as flatten
import numpy as np
from plnn.proxlp_solver.solver import SaddleLP
# from tools.eran_tools import read_eran_tf_net
import pickle

'''
Code from GNN_branching and Acitve set paper (author: Jodie, Harkirat)
This file contains all model structures we have considered
'''

## original kw small model
## 14x14x16 (3136) --> 7x7x32 (1568) --> 100 --> 10 ----(4804 ReLUs)
def mnist_model(): 
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

## 14*14*8 (1568) --> 14*14*8 (1568) --> 14*14*8 (1568) --> 392 --> 100 (5196 ReLUs)
def mnist_model_deep():
    model = nn.Sequential(
        nn.Conv2d(1, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

# first medium size model 14x14x4 (784) --> 7x7x8 (392) --> 50 --> 10 ----(1226 ReLUs)
# robust error 0.068
def mnist_model_m1():
    model = nn.Sequential(
        nn.Conv2d(1, 4, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*7*7,50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    return model


# increase the mini model by increasing the number of channels
## 8x8x8 (512) --> 4x4x16 (256) --> 50 (50) --> 10 (818)
def mini_mnist_model_m1():
    model = nn.Sequential(
        nn.Conv2d(1, 8, 2, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 2, stride=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(4*4*16,50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    return model


# without the extra 50-10 layer (originally, directly 128-10, robust error is around 0.221)
## 8x8x4 (256) --> 4x4x8 (128) --> 50 --> 10 ---- (434 ReLUs)
def mini_mnist_model(): 
    model = nn.Sequential(
        nn.Conv2d(1, 4, 2, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 8, 2, stride=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*4*4,50),
        nn.ReLU(),
        nn.Linear(50,10),
    )
    return model

#### CIFAR

# 32*32*32 (32768) --> 32*16*16 (8192) --> 64*16*16 (16384) --> 64*8*8 (4096) --> 512 --> 512 
# 54272 ReLUs
def large_cifar_model(): 
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*8*8,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    return model

# 16*16*16 (4096) --> 32*8*8 (2048) --> 100 
# 6244 ReLUs
# wide model
def cifar_model():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

# 16*16*8 (2048) -->  16*16*8 (2048) --> 16*16*8 (2048) --> 512 --> 100
# 6756 ReLUs
#deep model
def cifar_model_deep():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*8*8, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


#deeper model
def cifar_model_deeper():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*8*8, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

# 16*16*8 (2048) --> 16*8*8 (1024) --> 100 
# 3172 ReLUs (base model)
def cifar_model_m2():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(16*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

# 16*16*4 (1024) --> 8*8*8 (512) --> 100 
def cifar_model_m1(): 
    model = nn.Sequential(
        nn.Conv2d(3, 4, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*8*8, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def add_single_prop(layers, gt, cls):
    '''
    gt: ground truth lable
    cls: class we want to verify against
    '''
    additional_lin_layer = nn.Linear(10, 1, bias=True)
    lin_weights = additional_lin_layer.weight.data
    lin_weights.fill_(0)
    lin_bias = additional_lin_layer.bias.data
    lin_bias.fill_(0)
    lin_weights[0, cls] = -1
    lin_weights[0, gt] = 1

    #verif_layers2 = flatten_layers(verif_layers1,[1,14,14])
    final_layers = [layers[-1], additional_lin_layer]
    final_layer  = simplify_network(final_layers)
    verif_layers = layers[:-1] + final_layer
    for layer in verif_layers:
        for p in layer.parameters():
            p.requires_grad = False

    return verif_layers


# OVAL cifar dataset
def load_cifar_1to1_exp(model, idx, test = None, cifar_test = None):
    if model=='cifar_base_kw':
        model_name = './models/cifar_base_kw.pth'
        model = cifar_model_m2()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='cifar_wide_kw':
        model_name = './models/cifar_wide_kw.pth'
        model = cifar_model()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='cifar_deep_kw':
        model_name = './models/cifar_deep_kw.pth'
        model = cifar_model_deep()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    else:
        raise NotImplementedError

    if cifar_test is None:
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.225, 0.225, 0.225])
        cifar_test = datasets.CIFAR10('./cifardata/', train=False, download=True,
                                      transform=transforms.Compose([transforms.ToTensor(), normalize]))

        # for local usage:
        # cifar_test = datasets.CIFAR10('./cifardata.nosync/', train=False,
        #                               transform=transforms.Compose([transforms.ToTensor(), normalize]), download=True)

    x,y = cifar_test[idx]
    x = x.unsqueeze(0)
    print(x, max(x), min(x))
    # first check the model is correct at the input
    y_pred = torch.max(model(x)[0], 0)[1].item()
    print('predicted label ', y_pred, ' correct label ', y)
    if  y_pred != y: 
        print('model prediction is incorrect for the given model')
        return None, None, None
    else: 
        if test ==None:
            choices = list(range(10))
            choices.remove(y_pred)
            test = random.choice(choices)

        print('tested against ',test)
        for p in model.parameters():
            p.requires_grad =False

        layers = list(model.children())
        added_prop_layers = add_single_prop(layers, y_pred, test)
        return x, added_prop_layers, test


# OVAL mnist dataset
def load_mnist_1to1_exp(model, idx, test = None, mnist_test = None):
    if model=='mnist_base_kw':
        model_name = './models/mnist_base_kw.pth'
        model = mnist_model_m1()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='mnist_wide_kw':
        model_name = './models/mnist_wide_kw.pth'
        model = mnist_model()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='mnist_deep_kw':
        model_name = './models/mnist_deep_kw.pth'
        model = mnist_model_deep()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='mnist_2layer':
        model_name = './models/mnist_2layer.pth'
        model = mnist_2layer()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='mnist_3layer':
        model_name = './models/mnist_3layer.pth'
        model = mnist_3layer()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='mnist_kw_500':
        model_name = './models/mnist_500_159.pth'
        model = mnist_500()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='mnist_100':
        model_name = './models/mnist_100.pth'
        model = mnist_100()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='mnist_20':
        model_name = './models/mnist_20.pth'
        model = mnist_20()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='mnist_1mlp':
        model_name = './models/mnist_1mlp.pth'
        model = mnist_1mlp()
        model.load_state_dict(torch.load(model_name, map_location = "cpu"))
    elif model=='mnist_conv_1':
        model_name = './models/mnist_conv_1.pth'
        model = mnist_conv_1()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='mnist_conv_2':
        model_name = './models/mnist_conv_2.pth'
        model = mnist_conv_2()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='mnist_conv_2_conv':
        model_name = './models/mnist_conv_2_conv.pth'
        model = mnist_conv_2_conv()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='mnist_conv_2_lin':
        model_name = './models/mnist_conv_2_lin.pth'
        model = mnist_conv_2_lin()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='mnist_wide_l1':
        model_name = './data/advertorch/mnist_model_wide0.3_adv.pt'
        sudo_model = mnist_model_wide_l1()
        model = mnist_model()
        sudo_model.load_state_dict(torch.load(model_name, map_location = "cpu"))
        model_dict = dict(model.named_parameters())
        for (name1, param1), (name2, param2) in zip(sudo_model.state_dict().items(), model_dict.items()):
            model_dict[name2].data.copy_(param1.data)
    else:
        raise NotImplementedError
    if mnist_test is None:
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
        ])
        mnist_test = datasets.MNIST("./mnistdata/", train=False, download=True, transform =transform)

    x,y = mnist_test[idx]
    x = x.unsqueeze(0)
    # input(x)
    # first check the model is correct at the input
    y_pred = torch.max(model(x)[0], 0)[1].item()
    print('predicted label ', y_pred, ' correct label ', y)
    if  y_pred != y or y == test:
        print('model prediction is incorrect for the given model')
        return None
    else:
        if test ==None:
            choices = list(range(10))
            choices.remove(y_pred)
            test = random.choice(choices)

        print('tested against ',test)
        for p in model.parameters():
            p.requires_grad =False

        layers = list(model.children())
        added_prop_layers = add_single_prop(layers, y_pred, test)
        return x, added_prop_layers, test


##########################################################################################
########################## Colt Networks for CAV workshop ################################
##########################################################################################
mnist_mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1))
mnist_sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1))
cifar_mean = torch.FloatTensor([0.4914, 0.4822, 0.4465]).view((1, 3, 1, 1))
cifar_sigma = torch.FloatTensor([0.2023, 0.1994, 0.2010]).view((1, 3, 1, 1))

def get_mean_sigma(dataset):
    if dataset == 'cifar10':
        mean = torch.FloatTensor([0.4914, 0.4822, 0.4465]).view((1, 3, 1, 1))
        sigma = torch.FloatTensor([0.2023, 0.1994, 0.2010]).view((1, 3, 1, 1))
    elif dataset == 'mnist':
        mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1))
        sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1))
    return mean, sigma

def max_pool(candi_tot, lb_abs, change_sign=False):
    '''
    diff layer is provided when simplify linear layers are required
    by providing linear layers, we reduce consecutive linear layers
    to one
    '''
    layers = []
    # perform max-pooling
    # max-pooling is performed in terms of paris.
    # Each loop iteration reduces the number of candidates by two
    while candi_tot > 1:
        temp = list(range(0, candi_tot//2))
        even = [2*i for i in temp]
        odd = [i+1 for i in even]
        max_pool_layer1 = nn.Linear(candi_tot, candi_tot, bias=True)
        weight_mp_1 = torch.eye(candi_tot)
        ####### replaced this
        # weight_mp_1[even,odd] = -1
        ####### with this
        for idl in even:
            weight_mp_1[idl, idl+1] = -1
        #######
        bias_mp_1 = torch.zeros(candi_tot)
        for idl in odd:
            bias_mp_1[idl] = -lb_abs[idl]
        bias_mp_1[-1] = -lb_abs[-1]
        #import pdb; pdb.set_trace()
        max_pool_layer1.weight = Parameter(weight_mp_1, requires_grad=False)
        max_pool_layer1.bias = Parameter(bias_mp_1, requires_grad=False)
        layers.append(max_pool_layer1)
        layers.append(nn.ReLU())
        new_candi_tot = (candi_tot+1)//2
        sum_layer = nn.Linear(candi_tot, new_candi_tot, bias=True)
        sum_layer_weight = torch.zeros([new_candi_tot, candi_tot])
        ####### replaced this
        # sum_layer_weight[temp,even]=1; sum_layer_weight[temp,odd]=1
        ####### with this
        for idl in temp:
            sum_layer_weight[idl, 2*idl] = 1; sum_layer_weight[idl, 2*idl+1]=1
        #######
        sum_layer_weight[-1][-1] = 1
        sum_layer_bias = torch.zeros(new_candi_tot)
        for idl in temp:
            sum_layer_bias[idl]= lb_abs[2*idl+1]
        sum_layer_bias[-1] = lb_abs[-1]
        if change_sign is True and new_candi_tot==1:
            sum_layer.weight = Parameter(-1*sum_layer_weight, requires_grad=False)
            sum_layer.bias = Parameter(-1*sum_layer_bias, requires_grad=False)
        else:
            sum_layer.weight = Parameter(sum_layer_weight, requires_grad=False)
            sum_layer.bias = Parameter(sum_layer_bias, requires_grad=False)
        layers.append(sum_layer)

        pre_candi_tot = candi_tot
        candi_tot = new_candi_tot
        pre_lb_abs = lb_abs
        lb_abs = np.zeros(new_candi_tot)
        for idl in temp:
            lb_abs[idl]= min(pre_lb_abs[2*idl], pre_lb_abs[2*idl+1])
        lb_abs[-1] = pre_lb_abs[-1]

        #import pdb; pdb.set_trace()

    return layers

def add_properties(model, true_label, lb_abs = -1000, ret_ls = False, domain=None, max_solver_batch=10000):
    '''
    Input: pre_trained models
    Output: net layers with the mnist verification property added
    '''
    for p in model.parameters():
        p.requires_grad =False
    layers = list(model.children())

    last_layer = layers[-1]
    diff_in = last_layer.out_features
    diff_out = last_layer.out_features-1
    diff_layer = nn.Linear(diff_in,diff_out, bias=True)
    temp_weight_diff = torch.eye(10)
    temp_weight_diff[:,true_label] -= 1
    all_indices = list(range(10))
    all_indices.remove(true_label)
    weight_diff = temp_weight_diff[all_indices]
    bias_diff = torch.zeros(9)
    
    diff_layer.weight = Parameter(weight_diff, requires_grad=False)
    diff_layer.bias = Parameter(bias_diff, requires_grad=False)
    layers.append(diff_layer)
    layers = simplify_network(layers)

    cuda_verif_layers = [copy.deepcopy(lay).cuda() for lay in layers]
    # use best of naive interval propagation and KW as intermediate bounds
    intermediate_net = SaddleLP(cuda_verif_layers, max_batch=max_solver_batch)
    intermediate_net.set_solution_optimizer('best_naive_kw', None)
    intermediate_net.define_linear_approximation(domain.cuda().unsqueeze(0))
    lbs = intermediate_net.lower_bounds[-1].squeeze(0).cpu()

    candi_tot = diff_out
    # since what we are actually interested in is the minium of gt-cls,
    # we revert all the signs of the last layer
    max_pool_layers = max_pool(candi_tot, lbs, change_sign=True)

    # simplify linear layers
    simp_required_layers = layers[-1:]+max_pool_layers
    simplified_layers = simplify_network(simp_required_layers)

    final_layers = layers[:-1]+simplified_layers
    if ret_ls is False:
        return final_layers
    else:
        return [layers[:-1], simplified_layers]

def normalize(image, dataset):
    mean, sigma = get_mean_sigma(dataset)
    return (image - mean) / sigma

class SeqNet(nn.Module):

    def __init__(self):
        super(SeqNet, self).__init__()
        self.is_double = False
        self.skip_norm = False

    def forward(self, x, init_lambda=False):
        if isinstance(x, torch.Tensor) and self.is_double:
            x = x.to(dtype=torch.float64)
        x = self.blocks(x, init_lambda, skip_norm=self.skip_norm)
        return x

    def reset_bounds(self):
        for block in self.blocks:
            block.bounds = None

    def to_double(self):
        self.is_double = True
        for param_name, param_value in self.named_parameters():
            param_value.data = param_value.data.to(dtype=torch.float64)

    def forward_until(self, i, x):
        """ Forward until layer i (inclusive) """
        x = self.blocks.forward_until(i, x)
        return x

    def forward_from(self, i, x):
        """ Forward from layer i (exclusive) """
        x = self.blocks.forward_from(i, x)
        return x


class ConvMed(SeqNet):

    def __init__(self, device, dataset, n_class=10, input_size=32, input_channel=3, width1=1, width2=1, linear_size=100):
        super(ConvMed, self).__init__()

        mean, sigma = get_mean_sigma(dataset)

        layers = [
            Normalization(mean, sigma),
            Conv2d(input_channel, 16*width1, 5, stride=2, padding=2),
            ReLU((16*width1, input_size//2, input_size//2)),
            Conv2d(16*width1, 32*width2, 4, stride=2, padding=1),
            ReLU((32*width2, input_size//4, input_size//4)),
            flatten(),
            Linear(32*width2*(input_size // 4)*(input_size // 4), linear_size),
            ReLU(linear_size),
            Linear(linear_size, n_class),
        ]
        self.blocks = Sequential(*layers)

class ConvMedBig(SeqNet):

    def __init__(self, device, dataset, n_class=10, input_size=32, input_channel=3, width1=1, width2=1, width3=1, linear_size=100):
        super(ConvMedBig, self).__init__()

        mean, sigma = get_mean_sigma(dataset)
        self.normalizer = Normalization(mean, sigma)

        layers = [
            Normalization(mean, sigma),
            Conv2d(input_channel, 16*width1, 3, stride=1, padding=1, dim=input_size),
            ReLU((16*width1, input_size, input_size)),
            Conv2d(16*width1, 16*width2, 4, stride=2, padding=1, dim=input_size//2),
            ReLU((16*width2, input_size//2, input_size//2)),
            Conv2d(16*width2, 32*width3, 4, stride=2, padding=1, dim=input_size//2),
            ReLU((32*width3, input_size//4, input_size//4)),
            Flatten(),
            Linear(32*width3*(input_size // 4)*(input_size // 4), linear_size),
            ReLU(linear_size),
            Linear(linear_size, n_class),
        ]
        self.blocks = Sequential(*layers)

def convmed_colt_to_pytorch(n_class=10, input_size=32, input_channel=3, width1=1, width2=1, linear_size=100):
    model = nn.Sequential(
        nn.Conv2d(input_channel, 16*width1, 5, 2, 2, 1, 1, True),
        nn.ReLU(),
        nn.Conv2d(16*width1, 32*width2, 4, 2, 1, 1, 1, True),
        nn.ReLU(),
        Flatten(),#maybe flatten(),
        nn.Linear(32*width2*(input_size // 4)*(input_size // 4), linear_size, bias=True),
        nn.ReLU(),
        nn.Linear(linear_size, n_class, bias=True)
    )
    return model

def convmedbig_colt_to_pytorch(n_class=10, input_size=32, input_channel=3, width1=1, width2=1, width3=1, linear_size=100):
    model = nn.Sequential(
        nn.Conv2d(input_channel, 16*width1, 3, 1, 1, 1, 1, True),
        nn.ReLU(),
        nn.Conv2d(16*width1, 16*width2, 4, 2, 1, 1, 1, True),
        nn.ReLU(),
        nn.Conv2d(16*width2, 32*width3, 4, 2, 1, 1, 1, True),
        nn.ReLU(),
        Flatten(),#maybe flatten(),
        nn.Linear(32*width3*(input_size // 4)*(input_size // 4), linear_size, bias=True),
        nn.ReLU(),
        nn.Linear(linear_size, n_class, bias=True)
    )
    return model

def copyParams(module_src, module_dest):
    params_src = module_src.named_parameters()
    params_dest = module_dest.named_parameters()

    dict_dest = dict(params_dest)
    for name, param in params_src:
        split_name = name.split('.')
        if len(split_name) == 5: #eth case
            changed_name = (str(int(split_name[2])-1) + '.' + split_name[4])
            if changed_name in dict_dest:
                dict_dest[changed_name].data.copy_(param.data)


def get_network(dataset, device, net_name, net_loc, input_size, input_channel, n_class):
    if net_name.startswith('convmed_'):
        tokens = net_name.split('_')
        obj = ConvMed
        width1 = int(tokens[2])
        width2 = int(tokens[3])
        linear_size = int(tokens[4])
        net = obj(device, dataset, n_class, input_size, input_channel, width1=width1, width2=width2, linear_size=linear_size)
        net = net.to(device)
        net.load_state_dict(torch.load(net_loc))
        new_net = convmed_colt_to_pytorch(n_class, input_size, input_channel, width1=width1, width2=width2, linear_size=linear_size)
        copyParams(net, new_net)
    elif net_name.startswith('convmedbig_'):
        tokens = net_name.split('_')
        assert tokens[0] == 'convmedbig'
        width1 = int(tokens[2])
        width2 = int(tokens[3])
        width3 = int(tokens[4])
        linear_size = int(tokens[5])
        net = ConvMedBig(device, dataset, n_class, input_size, input_channel, width1, width2, width3, linear_size=linear_size)
        net = net.to(device)
        net.load_state_dict(torch.load(net_loc))

        new_net = convmedbig_colt_to_pytorch(n_class, input_size, input_channel, width1=width1, width2=width2, width3=width3, linear_size=linear_size)
        copyParams(net, new_net)
    else:
        assert False, 'Unknown network!'

    
    return new_net


def load_1to1_eth(dataset, model, idx = None, test = None, mnist_test = None, eps_temp=None, lb_abs=-1000, max_solver_batch=10000):
    device = 'cpu'
    if model=='mnist_0.1':
        net_name = 'convmed_flat_2_2_100'
        net_loc = './models/mnist_0.1_convmed_flat_2_2_100.pt'
        model = get_network(dataset, device, net_name, net_loc, 28, 1, 10)
    elif model=='mnist_0.3':
        net_name = 'convmed_flat_2_4_250'
        net_loc = './models/mnist_0.3_convmed_flat_2_4_250.pt'
        model = get_network(dataset, device, net_name, net_loc, 28, 1, 10)
    elif model=='cifar10_8_255':
        net_name = 'convmed_flat_2_4_250'
        net_loc = './models/cifar10_8_255_convmed_flat_2_4_250.pt'
        model = get_network(dataset, device, net_name, net_loc, 32, 3, 10)
    elif model=='cifar10_2_255':
        net_name = 'convmedbig_flat_2_2_4_250'
        net_loc = './models/cifar10_2_255_convmedbig_flat_2_2_4_250.pt'
        model = get_network(dataset, device, net_name, net_loc, 32, 3, 10)
    elif 'mnist_convSmallRELU' in model:
        net_name = model
        model = torch.load('./models/'+ model + '.pkl')
        eps_temp = 0.12
    elif 'cifar10_convSmallRELU' in model:
        net_name = model
        # net_loc = './models/cifar10_convSmallRELU__Point.pyt'
        # net_loc = './models/cifar10_convSmallRELU__DiffAI.pyt'
        # model = get_network_eran(dataset, device, net_name, net_loc)
        model = torch.load('./models/'+ model + '.pkl')
        eps_temp = 0.03
        eps_temp = 2/255
    elif 'mnist_convMedGRELU' in model:
        net_name = model
        # net_loc = './models/cifar10_convSmallRELU__Point.pyt'
        model = torch.load('./models/'+ model + '.pkl')
        # model = torch.load('./models/'+ model + '.pkl')
        eps_temp = 0.12
    else:
        raise NotImplementedError

    # print(model)
    # print(idx)
    current_test = test[idx]
    image= np.float64(current_test[1:len(current_test)])/np.float64(255)
    y=np.int(current_test[0])
    if dataset == 'cifar10':
        # x = normalize(torch.from_numpy(np.array(image, dtype=np.float32).reshape([1, 3, 32, 32])).float())
        x = normalize(torch.from_numpy(np.array(image, dtype=np.float32).reshape([1, 32, 32, 3]).transpose(0,3,1,2)), dataset)
    elif dataset == 'mnist':
        x = normalize(torch.from_numpy(np.array(image, dtype=np.float32).reshape([1, 1, 28, 28])).float(), dataset)
    # import torchvision.datasets as datasets
    # import torchvision.transforms as transforms
    # normalize_d = transforms.Normalize(mean=[0.1307],
    #                                      std=[0.3081])
    # mnist_test = datasets.MNIST('./data/', train=False, download=True,
    #                               transform=transforms.Compose([transforms.ToTensor(), normalize_d]))
    # x,y = mnist_test[3]
    # x = x.unsqueeze(0)

    # first check the model is correct at the input
    y_pred = torch.max(model(x)[0], 0)[1].item()
    print('predicted label ', y_pred, ' correct label ', y)
    if  y_pred != y: 
        print('model prediction is incorrect for the given model')
        return None, None, None, None
    else:
        # layers = list(model.children())
        # added_prop_layers = add_single_prop(layers, y_pred, test)
        # return x, added_prop_layers, test
        if dataset == 'cifar10':
            x_m_eps = normalize(torch.from_numpy(np.array(
                image - eps_temp, dtype=np.float32).reshape([1, 32, 32, 3]).transpose(0, 3, 1, 2)).clamp(0,1), dataset)
            x_p_eps = normalize(torch.from_numpy(np.array(
                image + eps_temp, dtype=np.float32).reshape([1, 32, 32, 3]).transpose(0, 3, 1, 2)).clamp(0,1), dataset)
        elif dataset == 'mnist':
            x_m_eps = normalize(torch.from_numpy(
                np.array(image - eps_temp, dtype=np.float32).reshape([1, 1, 28, 28])).clamp(0,1).float(), dataset)
            x_p_eps = normalize(torch.from_numpy(
                np.array(image + eps_temp, dtype=np.float32).reshape([1, 1, 28, 28])).clamp(0,1).float(), dataset)

        domain = torch.stack([x_m_eps.squeeze(0), x_p_eps.squeeze(0)], dim=-1)
        added_prop_layers = add_properties(model, y, lb_abs, domain=domain, max_solver_batch=max_solver_batch)
        for layer in added_prop_layers:
            for p in layer.parameters():
                p.requires_grad = False
        return x, added_prop_layers, None, domain


##########################################################################################
############################## Networks for Simplex paper ################################
##########################################################################################

def mnist_2layer(): 
    model = nn.Sequential(
        Flatten(),
        nn.Linear(28*28,500),
        nn.ReLU(),
        nn.Linear(500,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def mnist_3layer(): 
    model = nn.Sequential(
        Flatten(),
        nn.Linear(28*28,500),
        nn.ReLU(),
        nn.Linear(500,100),
        nn.ReLU(),
        nn.Linear(100, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    return model

# Robust error 0.138  Error 0.024
def mnist_500(): 
    model = nn.Sequential(
        Flatten(),
        nn.Linear(28*28,500),
        nn.ReLU(),
        nn.Linear(500, 10)
    )
    return model

def mnist_100(): 
    model = nn.Sequential(
        Flatten(),
        nn.Linear(28*28, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def mnist_20(): 
    model = nn.Sequential(
        Flatten(),
        nn.Linear(28*28, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    return model

def mnist_1mlp(): 
    model = nn.Sequential(
        Flatten(),
        nn.Linear(28*28,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def mnist_conv_2(): 
    model = nn.Sequential(
        nn.Conv2d(1, 2, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(2*14*14,10),
    )
    return model

def mnist_conv_1(): 
    model = nn.Sequential(
        nn.Conv2d(1, 1, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(1*14*14,10),
    )
    return model
    
def mnist_conv_2_lin(): 
    model = nn.Sequential(
        nn.Conv2d(1, 2, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(2*14*14,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def mnist_conv_2_conv(): 
    model = nn.Sequential(
        nn.Conv2d(1, 2, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(2, 2, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(2*7*7, 10)
    )
    return model


###################################### FOR GLOBAL SPEC ################################

import torch.nn as nn
from mmbt.mmbt.models.bow import GloveBowEncoder
from mmbt.mmbt.models.image import ImageEncoder

from mmbt.mmbt.data.vocab import Vocab

def get_vocab(glove_path):
    vocab = Vocab()
    word_list = get_glove_words(glove_path)
    vocab.add(word_list)

    return vocab

def load_glove(embed, glove_path):
    print("Loading glove")
    vocab = get_vocab(glove_path)
    pretrained_embeds = np.zeros(
        (400005, 300), dtype=np.float32
    )
    for line in open(glove_path):
        w, v = line.split(" ", 1)
        if w in vocab.stoi:
            pretrained_embeds[vocab.stoi[w]] = np.array(
                [float(x) for x in v.split()], dtype=np.float32
            )
    embed.weight = torch.nn.Parameter(torch.from_numpy(pretrained_embeds))

def MultimodalConcatBowOnlyClfRelu(num_layers):
    # embed = nn.Embedding(400005, 300)
    # load_glove(embed, glove_path)
    # embed.weight.requires_grad = False

    # txt = GloveBowEncoder(args)(txt)
    # img = ImageEncoder(args)(img)

    # self.layers = nn.ModuleList()

    in_dim = 300 + (2048 * 3)
    out_dim = 101

    layers = []
    hidden_dim = [in_dim, 2048*2, 2048, 300, out_dim*2]

    for l in range(num_layers):
        layers.append(nn.Linear(in_dim, hidden_dim[l]))
        layers.append(nn.ReLU())
        in_dim = hidden_dim[l]

    layers.append(nn.Linear(in_dim, out_dim))

    model = nn.Sequential(*layers)
    return model


###############################################################
## for l1 training

## original kw small model
## 14x14x16 (3136) --> 7x7x32 (1568) --> 100 --> 10 ----(4804 ReLUs)
class mnist_model_wide_l1(nn.Module):

    def __init__(self):
        super(mnist_model_wide_l1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.linear1 = nn.Linear(32*7*7,100)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(100, 10)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out

####### CIFAR

# 16*16*16 (4096) --> 32*8*8 (2048) --> 100 
# 6244 ReLUs
# wide model
class cifar_model_wide_l1(nn.Module):

    def __init__(self):
        super(cifar_model_wide_l1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.linear1 = nn.Linear(32*8*8,100)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(100, 10)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out

# 16*16*8 (2048) -->  16*16*8 (2048) --> 16*16*8 (2048) --> 512 --> 100
# 6756 ReLUs
#deep model
class cifar_model_deep_l1(nn.Module):

    def __init__(self):
        super(cifar_model_deep_l1, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(8, 8, 4, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        self.linear1 = nn.Linear(8*8*8, 100)
        self.relu5 = nn.ReLU()
        self.linear2 = nn.Linear(100, 10)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        out = self.relu4(self.conv4(out))
        out = out.view(out.size(0), -1)
        out = self.relu5(self.linear1(out))
        out = self.linear2(out)
        return out

# 16*16*8 (2048) -->  16*16*8 (2048) --> 16*16*8 (2048) --> 16*16*8 (2048) --> 16*16*8 (2048) --> 512 --> 100 --> 100
# 10952 ReLUs
#deeper model
class cifar_model_deeper_l1(nn.Module):

    def __init__(self):
        super(cifar_model_deeper_l1, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(8, 8, 4, stride=2, padding=1)
        self.relu6 = nn.ReLU()
        self.linear1 = nn.Linear(8*8*8, 100)
        self.relu7 = nn.ReLU()
        self.linear2 = nn.Linear(100, 100)
        self.relu8 = nn.ReLU()
        self.linear3 = nn.Linear(100, 10)

    def forward_l1(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        out = self.relu4(self.conv4(out))
        out = self.relu5(self.conv5(out))
        out = self.relu6(self.conv6(out))
        out = out.view(out.size(0), -1)
        out = self.relu7(self.linear1(out))
        out = self.relu8(self.linear2(out))
        out = self.linear3(out)
        return out

# 32*32*32 (32768) --> 32*16*16 (8192) --> 64*16*16 (16384) --> 64*8*8 (4096) --> 512 --> 512 
# 54272 ReLUs
class cifar_model_large_l1(nn.Module):

    def __init__(self):
        super(cifar_model_large_l1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        self.linear1 = nn.Linear(64*8*8,512)
        self.relu5 = nn.ReLU()
        self.linear2 = nn.Linear(512,512)
        self.relu6 = nn.ReLU()
        self.linear3 = nn.Linear(512,10)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        out = self.relu4(self.conv4(out))
        out = out.view(out.size(0), -1)
        out = self.relu5(self.linear1(out))
        out = self.relu6(self.linear2(out))
        out = self.linear3(out)
        return out

# 16*16*8 (2048) --> 16*8*8 (1024) --> 100 
# 3172 ReLUs (base model)
class cifar_model_base_l1(nn.Module):

    def __init__(self):
        super(cifar_model_base_l1, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, 4, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.linear1 = nn.Linear(16*8*8,100)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(100, 10)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out

def load_cifar_l1_network(model):
    if model=='cifar_model_wide_l1':
        model_name = './data/advertorch/cifar_model_wide_0_0.3_10_adv_0.8.pt'
        sudo_model = cifar_model_wide_l1()
        model = cifar_model()
        sudo_model.load_state_dict(torch.load(model_name, map_location = "cpu"))
        model_dict = dict(model.named_parameters())
        for (name1, param1), (name2, param2) in zip(sudo_model.state_dict().items(), model_dict.items()):
            model_dict[name2].data.copy_(param1.data)
    elif model=='cifar_model_deep_l1':
        model_name = './data/advertorch/cifar_model_deep_0_0.3_10_adv_0.8.pt'
        sudo_model = cifar_model_deep_l1()
        model = cifar_model_deep()
        sudo_model.load_state_dict(torch.load(model_name, map_location = "cpu"))
        model_dict = dict(model.named_parameters())
        for (name1, param1), (name2, param2) in zip(sudo_model.state_dict().items(), model_dict.items()):
            model_dict[name2].data.copy_(param1.data)
    elif model=='cifar_model_deeper_l1':
        model_name = './data/advertorch/cifar_model_deeper_0_0.3_10_adv_0.8.pt'
        sudo_model = cifar_model_deeper_l1()
        model = cifar_model_deeper()
        sudo_model.load_state_dict(torch.load(model_name, map_location = "cpu"))
        model_dict = dict(model.named_parameters())
        for (name1, param1), (name2, param2) in zip(sudo_model.state_dict().items(), model_dict.items()):
            model_dict[name2].data.copy_(param1.data)
    elif model=='cifar_model_large_l1':
        model_name = './data/advertorch/cifar_model_large_0_0.3_10_adv_0.8.pt'
        sudo_model = cifar_model_large_l1()
        model = large_cifar_model()
        sudo_model.load_state_dict(torch.load(model_name, map_location = "cpu"))
        model_dict = dict(model.named_parameters())
        for (name1, param1), (name2, param2) in zip(sudo_model.state_dict().items(), model_dict.items()):
            model_dict[name2].data.copy_(param1.data)
    elif model=='cifar_model_base_l1':
        model_name = './data/advertorch/cifar_model_base_0_0.3_10_adv_0.8.pt'
        sudo_model = cifar_model_base_l1()
        model = cifar_model_m2()
        sudo_model.load_state_dict(torch.load(model_name, map_location = "cpu"))
        model_dict = dict(model.named_parameters())
        for (name1, param1), (name2, param2) in zip(sudo_model.state_dict().items(), model_dict.items()):
            model_dict[name2].data.copy_(param1.data)
    else:
        raise NotImplementedError

    return model
