#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from mmbt.mmbt.models.bert import BertClf
from mmbt.mmbt.models.bow import GloveBowClf
from mmbt.mmbt.models.concat_bert import MultimodalConcatBertClf
from mmbt.mmbt.models.concat_bow import  MultimodalConcatBowClf
from mmbt.mmbt.models.concat_bow_relu import  MultimodalConcatBowClfRelu
from mmbt.mmbt.models.concat_bow_bnrelu import  MultimodalConcatBowClfBNRelu
from mmbt.mmbt.models.image import ImageClf
from mmbt.mmbt.models.mmbt import MultimodalBertClf


MODELS = {
    "bert": BertClf,
    "bow": GloveBowClf,
    "concatbow": MultimodalConcatBowClf,
    "concatbowrelu": MultimodalConcatBowClfRelu,
    "concatbowbnrelu": MultimodalConcatBowClfBNRelu,
    "concatbert": MultimodalConcatBertClf,
    "img": ImageClf,
    "mmbt": MultimodalBertClf,
}


def get_model(args):
    return MODELS[args.model](args)
