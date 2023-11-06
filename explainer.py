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
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
epoch = checkpoint["epoch"]
loss = checkpoint["loss"]
feat = model.input_feature()

from dglnn_local.subgraphx import HeteroSubgraphX
from dgl import khop_in_subgraph, khop_out_subgraph, node_subgraph
import dgl

explainer = HeteroSubgraphX(model, num_hops=1)
# in_graph = khop_in_subgraph(g, {"d": [0]}, 1)[0]
# out_graph = khop_out_subgraph(g, {"d": [0]}, 1)[0]
# comb_graph = dgl.merge([in_graph, out_graph])
# # exp_graph = node_subgraph(g, comb_graph.nodes)
# data_dict = {}
# for ntype in in_graph.ntypes:
#     for i in in_graph.nodes[ntype].data:
#         data_dict[ntype] = in_graph.nodes[ntype].data[i].tolist()
# feat_1 = nn.ParameterDict()
# for item in feat:
#     feat_1[item] = feat[item][data_dict[item]]

explanation = explainer.explain_graph(g, feat, target_class=1)
print(explanation)
