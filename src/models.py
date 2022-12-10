import torch
from torch import nn

from src.custom_layers import GCNLayer, SumLayer


def create_graph_class_seq(first_layer_d, out_num, droupout_rate=0.1):
    """Chains a sequence of GCNlayers with input dimension first_layer_d,
     and output dimension out_num, and dropout_rate of 0.1"""
    batch_size = 7  # empirically chosen batch size

    return torch.nn.Sequential(
        GCNLayer(batch_size, d=first_layer_d),
        # d is picked to match longest dim
        GCNLayer(batch_size),
        GCNLayer(batch_size),
        GCNLayer(batch_size),
        GCNLayer(batch_size, last=True),  # mark last gcn layer
        torch.nn.Dropout(p=droupout_rate),
        SumLayer(),
        # flatten first to use linear layer on it.
        nn.Flatten(),
        torch.nn.Linear(64, 64, dtype=torch.float64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, out_num, dtype=torch.float64)
    )


def nci_model():
    """Build model,
     where first GCN layer has max dim 38 and 2 output features."""
    return create_graph_class_seq(38, 2)


def enz_model():
    """Build model,
     where first GCN layer has max dim 22 and 6 output features."""
    return create_graph_class_seq(22, 6)
