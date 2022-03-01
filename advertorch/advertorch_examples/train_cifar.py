# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import print_function

import os
import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.test_utils import cifar_model_wide, cifar_model_deep, cifar_model_large, cifar_model_deeper, cifar_model_base
from advertorch_examples.utils import get_cifar10_train_loader, cifar_loaders
from advertorch_examples.utils import get_cifar10_test_loader
from advertorch_examples.utils import TRAINED_MODEL_PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CIFAR')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--mode', default="adv", help="cln | adv")
    parser.add_argument('--train_batch_size', default=1, type=int)
    parser.add_argument('--test_batch_size', default=1, type=int)
    parser.add_argument('--log_interval', default=40, type=int)
    parser.add_argument('--model_name', default="cifar_model_deep", type=str)
    parser.add_argument('--eps', default=0.3, type=float)
    parser.add_argument('--nb_iter', default=10, type=float)
    parser.add_argument('--adv_weight', default=0.8, type=float)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.mode == "cln":
        flag_advtrain = False
        nb_epoch = 50
        model_filename = args.model_name + "_cln.pt"
        model_filename = args.model_name + "_last_cln.pt"
    elif args.mode == "adv":
        flag_advtrain = True
        nb_epoch = 200
        model_filename = args.model_name + "_0_" + str(args.eps) + "_" + str(args.nb_iter) + "_adv_" + str(args.adv_weight) + ".pt"
        model_last_filename = args.model_name + "_0_" + str(args.eps) + "_" + str(args.nb_iter) + "_last_adv_" + str(args.adv_weight) + ".pt"
    else:
        raise

    # train_loader = get_cifar10_train_loader(
    #     batch_size=args.train_batch_size, shuffle=True)
    # test_loader = get_cifar10_test_loader(
    #     batch_size=args.test_batch_size, shuffle=False)

    train_loader, test_loader = cifar_loaders(args.train_batch_size)

    if args.model_name == "cifar_model_wide": 
        model = cifar_model_wide()
    elif args.model_name == "cifar_model_deep": 
        model = cifar_model_deep()
    elif args.model_name == "cifar_model_large": 
        model = cifar_model_large()
    elif args.model_name == "cifar_model_deeper": 
        model = cifar_model_deeper()
    elif args.model_name == "cifar_model_base": 
        model = cifar_model_base()
    model.to(device)

    # optimizer = optim.Adam(model.parameters(), lr=1e-4)

    optimizer = optim.SGD(model.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    if flag_advtrain:
        # from advertorch.attacks import LinfPGDAttack
        # adversary = LinfPGDAttack(
        #     model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
        #     nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0,
        #     clip_max=1.0, targeted=False)

        from advertorch.attacks import L1PGDAttack
        adversary = L1PGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.5,
            nb_iter=args.nb_iter, eps_iter=0.25, rand_init=True, clip_min=-1.0,
            clip_max=1.0, targeted=False, l1_sparsity=0.7)

    best_correct = 0
    for epoch in range(nb_epoch):

        # model.train()
        # for batch_idx, (data, target) in enumerate(train_loader):
        #     data, target = data.to(device), target.to(device)
        #     ori = copy.deepcopy(data)
        #     if flag_advtrain:
        #         # when performing attack, the model needs to be in eval mode
        #         # also the parameters should NOT be accumulating gradients
        #         with ctx_noparamgrad_and_eval(model):
        #             data = adversary.perturb(data, target)

        #     optimizer.zero_grad()
        #     output = model(data)
        #     if flag_advtrain:
        #         loss = args.adv_weight*F.cross_entropy(output, target, reduction='elementwise_mean')
        #     else:
        #         loss = F.cross_entropy(output, target, reduction='elementwise_mean')

        #     loss.backward()

        #     ####
        #     if flag_advtrain:
        #         output = model(ori)
        #         loss = (1-args.adv_weight)*F.cross_entropy(output, target, reduction='elementwise_mean')
        #         loss.backward()

        #     ###

        #     optimizer.step()
        #     if batch_idx % args.log_interval == 0:
        #         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #             epoch, batch_idx *
        #             len(data), len(train_loader.dataset),
        #             100. * batch_idx / len(train_loader), loss.item()))

        # scheduler.step()

        model.eval()
        model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, model_filename)))
        test_clnloss = 0
        clncorrect = 0

        if flag_advtrain:
            test_advloss = 0
            advcorrect = 0

        num_test_images=0
        for clndata, target in test_loader:
            num_test_images+=1
            if num_test_images>=1001:
                break
            clndata, target = clndata.to(device), target.to(device)
            with torch.no_grad():
                output = model(clndata)
            test_clnloss += F.cross_entropy(
                output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            clncorrect += pred.eq(target.view_as(pred)).sum().item()

            if flag_advtrain:
                advdata = adversary.perturb(clndata, target)
                with torch.no_grad():
                    output = model(advdata)
                test_advloss += F.cross_entropy(
                    output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                advcorrect += pred.eq(target.view_as(pred)).sum().item()

        

        test_clnloss /= len(test_loader.dataset)
        print('\nTest set: avg cln loss: {:.4f},'
              ' cln acc: {}/{} ({:.0f}%)\n'.format(
                  test_clnloss, clncorrect, len(test_loader.dataset),
                  100. * clncorrect / len(test_loader.dataset)))
        if flag_advtrain:
            test_advloss /= len(test_loader.dataset)
            print('Test set: avg adv loss: {:.4f},'
                  ' adv acc: {}/{} ({:.0f}%)\n'.format(
                      test_advloss, advcorrect, len(test_loader.dataset),
                      100. * advcorrect / len(test_loader.dataset)))

        if clncorrect>best_correct:
            best_correct = clncorrect
            torch.save(model.state_dict(), os.path.join(TRAINED_MODEL_PATH, model_filename))

    torch.save(
        model.state_dict(),
        os.path.join(TRAINED_MODEL_PATH, model_last_filename))
