#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import functools
import json
import os
from collections import Counter

import torch
import torchvision.transforms as transforms
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import DataLoader

from mmbt.mmbt.data.dataset import JsonlDataset
from mmbt.mmbt.data.vocab import Vocab


def get_transforms(args):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )


def get_labels_and_frequencies(path):
    label_freqs = Counter()
    data_labels = [json.loads(line)["label"] for line in open(path)]
    if type(data_labels[0]) == list:
        for label_row in data_labels:
            label_freqs.update(label_row)
    else:
        label_freqs.update(data_labels)

    return list(label_freqs.keys()), label_freqs


def get_glove_words(path):
    word_list = []
    for line in open(path):
        w, _ = line.split(" ", 1)
        word_list.append(w)
    return word_list


def get_vocab(args):
    vocab = Vocab()
    if args.model in ["bert", "mmbt", "concatbert"]:
        bert_tokenizer = BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=True
        )
        vocab.stoi = bert_tokenizer.vocab
        vocab.itos = bert_tokenizer.ids_to_tokens
        vocab.vocab_sz = len(vocab.itos)

    else:
        word_list = get_glove_words(args.glove_path)
        vocab.add(word_list)

    return vocab


def collate_fn(batch, args):
    lens = [len(row[0]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len).long()
    text_tensor = torch.zeros(bsz, max_seq_len).long()
    segment_tensor = torch.zeros(bsz, max_seq_len).long()

    img_tensor = None
    if args.model in ["img", "concatbow", "concatbert", "mmbt", "concatbowrelu", "concatbowbnrelu"]:
        img_tensor = torch.stack([row[2] for row in batch])

    if args.task_type == "multilabel":
        # Multilabel case
        tgt_tensor = torch.stack([row[3] for row in batch])
    else:
        # Single Label case
        tgt_tensor = torch.cat([row[3] for row in batch]).long()

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        tokens, segment = input_row[:2]
        text_tensor[i_batch, :length] = tokens
        segment_tensor[i_batch, :length] = segment
        mask_tensor[i_batch, :length] = 1

    return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor


def get_data_loaders(args):
    tokenizer = (
        BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
        if args.model in ["bert", "mmbt", "concatbert"]
        else str.split
    )

    transforms = get_transforms(args)

    # args.labels, args.label_freqs = get_labels_and_frequencies(
    #     os.path.join(args.data_path, args.task, "train.jsonl")
    # )
    # args.label_freqs= Counter({'macaroni_and_cheese': 658, 'sushi': 655, 'risotto': 653, 'waffles': 649, 'pizza': 648, 'apple_pie': 647, 'hot_dog': 647, 'french_toast': 647, 'pancakes': 646, 'gnocchi': 645, 'hummus': 645, 'hamburger': 644, 'frozen_yogurt': 643, 'cheesecake': 642, 'tacos': 641, 'ice_cream': 640, 'tiramisu': 640, 'ravioli': 640, 'bruschetta': 639, 'chocolate_cake': 638, 'lasagna': 638, 'donuts': 638, 'omelette': 636, 'macarons': 636, 'gyoza': 636, 'ramen': 636, 'cannoli': 635, 'churros': 632, 'deviled_eggs': 632, 'steak': 631, 'guacamole': 630, 'falafel': 629, 'cup_cakes': 629, 'dumplings': 629, 'shrimp_and_grits': 629, 'ceviche': 626, 'miso_soup': 625, 'bread_pudding': 624, 'grilled_salmon': 623, 'panna_cotta': 620, 'bibimbap': 620, 'croque_madame': 620, 'beef_tartare': 619, 'chocolate_mousse': 619, 'beet_salad': 619, 'beignets': 617, 'huevos_rancheros': 616, 'red_velvet_cake': 614, 'nachos': 612, 'beef_carpaccio': 612, 'cheese_plate': 611, 'oysters': 609, 'caprese_salad': 608, 'club_sandwich': 607, 'clam_chowder': 607, 'takoyaki': 607, 'prime_rib': 605, 'spaghetti_carbonara': 600, 'pulled_pork_sandwich': 595, 'foie_gras': 595, 'samosa': 593, 'lobster_roll_sandwich': 591, 'french_fries': 585, 'fried_calamari': 583, 'breakfast_burrito': 583, 'grilled_cheese_sandwich': 582, 'poutine': 581, 'onion_rings': 576, 'eggs_benedict': 575, 'edamame': 575, 'carrot_cake': 571, 'spring_rolls': 571, 'pad_thai': 567, 'hot_and_sour_soup': 560, 'caesar_salad': 560, 'mussels': 557, 'escargots': 555, 'scallops': 552, 'greek_salad': 551, 'sashimi': 547, 'pork_chop': 545, 'fried_rice': 545, 'creme_brulee': 545, 'french_onion_soup': 535, 'baby_back_ribs': 535, 'chicken_quesadilla': 532, 'baklava': 529, 'pho': 528, 'crab_cakes': 525, 'garlic_bread': 524, 'chicken_curry': 523, 'seaweed_salad': 520, 'chicken_wings': 519, 'lobster_bisque': 516, 'tuna_tartare': 515, 'strawberry_shortcake': 514, 'paella': 507, 'fish_and_chips': 504, 'filet_mignon': 502, 'spaghetti_bolognese': 496, 'peking_duck': 494})
    # args.labels=['falafel', 'beef_tartare', 'apple_pie', 'pulled_pork_sandwich', 'pizza', 'oysters', 'macaroni_and_cheese', 'chocolate_cake', 'chocolate_mousse', 'panna_cotta', 'lasagna', 'fried_calamari', 'cup_cakes', 'gnocchi', 'donuts', 'chicken_wings', 'waffles', 'baklava', 'hot_and_sour_soup', 'churros', 'french_fries', 'beet_salad', 'tuna_tartare', 'lobster_roll_sandwich', 'pad_thai', 'scallops', 'omelette', 'risotto', 'dumplings', 'cheesecake', 'pork_chop', 'escargots', 'hamburger', 'hot_dog', 'samosa', 'chicken_quesadilla', 'caprese_salad', 'beignets', 'pho', 'nachos', 'french_onion_soup', 'red_velvet_cake', 'foie_gras', 'caesar_salad', 'frozen_yogurt', 'prime_rib', 'carrot_cake', 'eggs_benedict', 'bibimbap', 'garlic_bread', 'sashimi', 'huevos_rancheros', 'spring_rolls', 'tacos', 'hummus', 'mussels', 'strawberry_shortcake', 'chicken_curry', 'guacamole', 'baby_back_ribs', 'club_sandwich', 'beef_carpaccio', 'breakfast_burrito', 'grilled_salmon', 'fried_rice', 'clam_chowder', 'macarons', 'gyoza', 'ramen', 'ice_cream', 'pancakes', 'onion_rings', 'cannoli', 'bruschetta', 'deviled_eggs', 'cheese_plate', 'sushi', 'spaghetti_bolognese', 'paella', 'filet_mignon', 'bread_pudding', 'creme_brulee', 'french_toast', 'tiramisu', 'shrimp_and_grits', 'poutine', 'seaweed_salad', 'ravioli', 'greek_salad', 'steak', 'miso_soup', 'edamame', 'grilled_cheese_sandwich', 'crab_cakes', 'lobster_bisque', 'ceviche', 'fish_and_chips', 'takoyaki', 'spaghetti_carbonara', 'croque_madame', 'peking_duck']

    args.label_freqs = Counter({'apple_pie': 653, 'waffles': 651, 'pizza': 650, 'ice_cream': 642, 'donuts': 640, 'french_fries': 586, 'onion_rings': 581, 'pad_thai': 568, 'chicken_curry': 524, 'chicken_wings': 521})
    args.labels = ['donuts', 'pizza', 'french_fries', 'ice_cream', 'onion_rings', 'chicken_wings', 'pad_thai', 'apple_pie', 'chicken_curry', 'waffles']

    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz
    args.n_classes = len(args.labels)

    train = JsonlDataset(
        os.path.join(args.data_path, args.task, "train.jsonl"),
        tokenizer,
        transforms,
        vocab,
        args,
    )

    args.train_data_len = len(train)

    dev = JsonlDataset(
        os.path.join(args.data_path, args.task, "dev.jsonl"),
        tokenizer,
        transforms,
        vocab,
        args,
    )

    collate = functools.partial(collate_fn, args=args)

    train_loader = DataLoader(
        train,
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        dev,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    test_set = JsonlDataset(
        os.path.join(args.data_path, args.task, "test.jsonl"),
        tokenizer,
        transforms,
        vocab,
        args,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    if args.task == "vsnli":
        test_hard = JsonlDataset(
            os.path.join(args.data_path, args.task, "test_hard.jsonl"),
            tokenizer,
            transforms,
            vocab,
            args,
        )

        test_hard_loader = DataLoader(
            test_hard,
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=args.n_workers,
            collate_fn=collate,
        )

        test = {"test": test_loader, "test_hard": test_hard_loader}

    else:
        # test_gt = JsonlDataset(
        #     os.path.join(args.data_path, args.task, "test_hard_gt.jsonl"),
        #     tokenizer,
        #     transforms,
        #     vocab,
        #     args,
        # )

        # test_gt_loader = DataLoader(
        #     test_gt,
        #     batch_size=args.batch_sz,
        #     shuffle=False,
        #     num_workers=args.n_workers,
        #     collate_fn=collate,
        # )

        # test = {
        #     "test": test_loader,
        #     "test_gt": test_gt_loader,
        # }

        test = {
            "test": test_loader,
            # "test_gt": test_gt_loader,
        }

    return train_loader, val_loader, test
