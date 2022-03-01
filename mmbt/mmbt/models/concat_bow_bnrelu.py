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


class MultimodalConcatBowClfBNRelu(nn.Module):
    def __init__(self, args):
        super(MultimodalConcatBowClfBNRelu, self).__init__()
        self.args = args
        self.clf = nn.Linear(
            args.embed_sz + (args.img_hidden_sz * args.num_image_embeds), args.n_classes
        )
        self.txtenc = GloveBowEncoder(args)
        self.imgenc = ImageEncoder(args)

        self.layers = nn.ModuleList()

        in_dim = args.embed_sz + (args.img_hidden_sz * args.num_image_embeds)
        out_dim = args.n_classes

        dropout=0.5
        batch_norm=True
        num_layers=2

        hidden_dim = [in_dim, args.img_hidden_sz*2, args.img_hidden_sz, args.embed_sz, out_dim*2]

        for l in range(args.num_layers):
            self.layers.append(nn.Linear(in_dim, hidden_dim[l]))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_dim[l]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
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
