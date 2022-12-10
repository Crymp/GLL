import torch as torch


class GCNLayer(torch.nn.Module):
    def __init__(self, batch_size, d: int = 64, last: bool = False):
        super().__init__()
        # set last flag if it is the last layer of GCN
        self.last = last
        self.relu = torch.nn.ReLU()
        # trainable weight matrix of the l-th layer
        # first layer should be set to have first dim d depending on input
        self.W = torch.nn.Parameter(torch.randn(
            (d, 64),
            dtype=torch.float64
        ))
        self.d = d

    def forward(self, data):
        A, H = data
        # multiply A and H with batches
        AH = torch.bmm(A, H)
        # use A*H product and multiply it by weight.
        # same wheights are repeated for all batches with .repeat method
        inner_product = torch.bmm(AH, self.W.repeat(AH.shape[0], 1, 1))
        # for last layer omit returning A
        if self.last:
            return self.relu(inner_product)
        return A, self.relu(inner_product)


class SumLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, H):
        # sums H elementwise for each graph
        return torch.sum(H, 1)
