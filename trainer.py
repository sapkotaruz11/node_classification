from dataset import RDFDatasets
from configs import get_configs
from model import RGCN
import torch as th
import torch.nn.functional as F
import torch
from hetro_features import HeteroFeature
from utils import get_nodes_dict
import time
import numpy as np
from torch import nn
import shutil

import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
dataset = "mutag"
my_dataset = RDFDatasets(dataset)
configs = get_configs(dataset)
g = my_dataset.g.to(device)
category = my_dataset.category


hidden_dim = configs["hidden_dim"]
out_dim = my_dataset.num_classes
e_types = g.etypes
num_bases = configs["n_bases"]
lr = configs["lr"]
weight_decay = configs["weight_decay"]
epochs = configs["max_epoch"]
act = None

train_idx = my_dataset.train_idx.to(device)
valid_idx = my_dataset.valid_idx.to(device)
labels = my_dataset.labels.to(device)
# input_feature = HeteroFeature({}, get_nodes_dict(g), hidden_dim,
#                                             act=act).to(device)
input_feature = HeteroFeature({}, get_nodes_dict(g), hidden_dim, act=act).to(device)
model = RGCN(hidden_dim, hidden_dim, out_dim, e_types, num_bases, category).to(device)

# Define the optimizer
optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

loss_fn = F.cross_entropy
model.add_module("input_feature", input_feature)
optimizer.add_param_group({"params": input_feature.parameters()})

PATH = f"trained_models/{dataset}_trained.pt"
if not PATH:
    print("start training...")
    dur = []
    h_dict = model.input_feature()
    model.train()
    for epoch in range(50):
        t0 = time.time()
        logits = model(g, h_dict)[category]
        loss = loss_fn(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        t1 = time.time()

        dur.append(t1 - t0)
        train_acc = th.sum(
            logits[train_idx].argmax(dim=1) == labels[train_idx]
        ).item() / len(train_idx)
        val_loss = F.cross_entropy(logits[valid_idx], labels[valid_idx])
        val_acc = th.sum(
            logits[valid_idx].argmax(dim=1) == labels[valid_idx]
        ).item() / len(valid_idx)
        print(
            "Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Valid Acc: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".format(
                epoch,
                train_acc,
                loss.item(),
                val_acc,
                val_loss.item(),
                np.average(dur),
            )
        )
    to_PATH = f"trained_models/{dataset}_trained.pt"
    if to_PATH:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            PATH,
        )
else:
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
