import argparse as ap

from src.data import get_graph_labels, nci1, get_dataset, enzymes
from src.models import nci_model, enz_model
from src.validation import cross_val

#parser for selecting the GNN/datasets
parser = ap.parser = ap.ArgumentParser(description='Process some integers.')

parser.add_argument('dataset', choices=['NCI1', 'Enzymes', 'Citeseer', 'Cora'],
                    help='Specify which dataset '
                         '(NCI1, Enzymes, Citeseer, Cora) should be used.')

args = parser.parse_args()


def main():
    # start the training as specified in the argument
    dataset = args.dataset.lower()
    if dataset == 'nci1':
        y = get_graph_labels(nci1)
        loader_nci = get_dataset(nci1, y)
        cross_val(nci_model, loader_nci, num_epochs=45)

    if dataset == 'enzymes':
        y = get_graph_labels(enzymes) - 1
        loader_enz = get_dataset(enzymes, y, with_attrib=True)
        cross_val(enz_model, loader_enz, num_epochs=50)

    if dataset == 'citeseer':
        raise NotImplementedError()

    if dataset == 'cora':
        raise NotImplementedError()


