#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

from mmbt.mmbt.models.bow import GloveBowEncoder
from mmbt.mmbt.models.image import ImageEncoder
from mmbt.mmbt.utils.utils import simplex_projection_sort, projection_simplex_sort

import numpy as np
import pickle
import copy

class MultimodalConcatBowClfRelu(nn.Module):
    def __init__(self, args):
        super(MultimodalConcatBowClfRelu, self).__init__()
        self.args = args
        self.clf = nn.Linear(
            args.embed_sz + (args.img_hidden_sz * args.num_image_embeds), args.n_classes
        )
        self.txtenc = GloveBowEncoder(args)
        self.imgenc = ImageEncoder(args)

        self.layers = nn.ModuleList()

        in_dim = args.embed_sz + (args.img_hidden_sz * args.num_image_embeds)
        out_dim = args.n_classes

        hidden_dim = [in_dim, args.img_hidden_sz*2, args.img_hidden_sz, args.embed_sz, out_dim*2]
        # hidden_dim = [args.img_hidden_sz, args.embed_sz, 100, out_dim*2]
        hidden_dim = [args.embed_sz, out_dim*2]

        for l in range(args.num_layers):
            self.layers.append(nn.Linear(in_dim, hidden_dim[l]))
            self.layers.append(nn.ReLU())
            in_dim = hidden_dim[l]

        self.layers.append(nn.Linear(in_dim, out_dim))


    def forward(self, txt, img):
        txt = self.txtenc(txt)
        img = self.imgenc(img)
        img = torch.flatten(img, start_dim=1)
        x = torch.cat([txt, img], -1)

        for layer in self.layers:
            x = layer(x)

        return x


def fast_simplex_pgd(model, y_batch, criterion, max_iter=10, momentum=False, l1_sparsity=0.7, alpha=0.25):
    """
    Generates adversarial examples using  projected gradient descent (PGD).
    If adversaries have been generated, retrieve them.
    
    Input:
        - txt_batch : batch images to compute adversaries 
        - y_batch : labels of the batch
        - max_iter : # of iterations to generate adversarial example (FGSM=1)
         - is_normalized :type of input normalization (0: no normalization, 1: zero-mean per-channel normalization)
    
    Output:
        - x : batch containing adversarial examples
    """
    
    # converting from categorial to multi-hot


    top_k = 1000
    txt_new = []
    for lf in range(1):
         randnums= np.random.randint(1, top_k, 500).tolist()
         txt_new.append(randnums)
    txt_new = np.array(txt_new)
    txt_new = torch.from_numpy(txt_new)

    cat_new = torch.zeros(txt_new.shape[0], top_k, dtype=torch.float).cuda()
    for bt_idx in range(txt_new.shape[0]):
        for el in range(txt_new.shape[1]):
            if txt_new[bt_idx, el] is not None:
                cat_new[bt_idx, txt_new[bt_idx, el]] += 1.0

    cat_new = cat_new/500.0

    x = cat_new.clone().detach().requires_grad_(True).cuda()

    # Compute alpha. Alpha might vary depending on the type of normalization.
    alpha = alpha
    
    # # Set velocity for momentum
    # if momentum:
    #     g = torch.zeros(txt_batch.size(0), 1, 1, 1).cuda()
    
    for _ in range(max_iter):
        logits = model(x)
        loss = criterion(logits, y_batch)
        
        loss.backward()
        
        # Get gradient
        grad = x.grad.data
        abs_grad = torch.abs(grad)

        batch_size = grad.size(0)
        view = abs_grad.view(batch_size, -1)
        view_size = view.size(1)
        if l1_sparsity is None:
            vals, idx = view.topk(1)
        else:
            vals, idx = view.topk(
                int(np.round((1 - l1_sparsity) * view_size)))

        out = torch.zeros_like(view).scatter_(1, idx, vals)
        out = out.view_as(grad)
        grad = grad.sign() * (out > 0).float()
        
        # Momentum : You should not be using the mean here...
        if momentum:
            g = self.momentum * g.data + noise / torch.mean(torch.abs(noise), dim=(1,2,3), keepdim=True)
            noise = g.clone().detach()
        
        # print('Grad non zero: ', torch.nonzero(grad.data).shape)
        # Compute Adversary
        x.data = x.data + alpha * grad

        # Clamp data between valid ranges
        # delta.data = simplex_projection_sort(delta.data)
        x.data = projection_simplex_sort(x.data.cpu(), txt_new.shape[1])
        x.data = x.data.type(torch.cuda.FloatTensor).cuda()

        x.grad.zero_()

    return x
