import pickle
import warnings

import networkx as nx
import numpy as np
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torch


warnings.filterwarnings("ignore")

with open('../datasets/NCI1/data.pkl', 'rb') as f:
    nci1 = pickle.load(f)

with open('../datasets/ENZYMES/data.pkl', 'rb') as f:
    enzymes = pickle.load(f)


def get_adjacency_matrix(g):
    # get normal adjancency matrix
    am = nx.adjacency_matrix(g, dtype=float)
    # set diag to 1, since normalized adjency matrix works also when i = j
    am.setdiag(1)
    # for each node in matrix devide row and col by its 1/sqrt(degree) value
    # such that at the end all values are 1/sqrt(d_i, d_j)
    for i, node_id in enumerate(g.nodes):
        d_i_sqrt = np.sqrt(g.degree[node_id] + 1)
        am[i, :] /= d_i_sqrt
        am[:, i] /= d_i_sqrt
    return torch.tensor(am.todense())


def get_attributes(g):
    """Extract data attributes from nodes and use one hot on it."""
    node_attr = torch.tensor(
        [node[1]["node_attributes"] for node in g.nodes(data=True)]
    )
    norm = torch.norm(node_attr, dim=1, keepdim=True)
    # L2 norm on lables for stability
    return node_attr * (1 / norm).repeat(1, node_attr.shape[1])


def get_vertex_embedding(g):
    """Extract data from nodes and use one hot on it."""
    node_labels = torch.tensor(
        [node[1]["node_label"] for node in g.nodes(data=True)]
    )
    return F.one_hot(node_labels).double()


def get_vertex_embedding_with_attributes(g):
    """Use extracted node labels (one hot encoded) and attribute data from nodes
    and concatinae the arrays."""
    node_labels = get_vertex_embedding(g)
    attributes = get_attributes(g)
    return torch.cat((attributes, node_labels), dim=1)


def get_graph_labels(data):
    print("Getting labels")
    return torch.tensor([g.graph['label'] for g in data])


def get_dataset(graphs, y, batch_size=64, with_attrib=False):
    print("Processing dataset:")
    # load data with attributes or without
    print("Processing vertex embeddings")
    if not with_attrib:
        H = [get_vertex_embedding(g) for g in graphs]
    else:
        H = [get_vertex_embedding_with_attributes(g) for g in graphs]
    # get max matrix size as (my, mx)
    my = max(h.shape[0] for h in H)
    mx = max(h.shape[1] for h in H)
    print(f"Feature dims are maximum y: {my}, maximum x: {mx}")
    # pad H matrix to its max values given by max matrix size from above
    # the feature vector has therefor equal form
    H_padded = torch.tensor(np.array([
        F.pad(h, (0, mx - h.shape[1], 0, my - h.shape[0])).numpy()
        for h in H
    ]))

    print("Processing adjacency matrixes")
    # pad normalized adjacency matrixes
    A = [get_adjacency_matrix(g) for g in graphs]
    # calculate max y side of adjac. matrixes
    mlen = max(a.shape[0] for a in A)
    A_padded = torch.tensor(np.array([
        F.pad(a, (0, mlen - a.shape[0], 0, mlen - a.shape[0])).numpy()
        for a in A
    ]))

    # use tensordataset to pack data and later on to use in a dataloader
    return TensorDataset(A_padded, H_padded, y)
