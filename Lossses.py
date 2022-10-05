import torch
import torch.nn as nn
import torch.nn.functional as F


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

def Regloss(x, y):
    reg_loss = nn.SmoothL1Loss()

    return reg_loss(x, y)

def BarlowTwins_loss(x, y, bn_size=10, lambd=0.0051):
    Batch_size = x.size(0)
    bn = nn.BatchNorm1d(bn_size, affine=False).cuda()
    c = bn(x).t().mm(bn(y))
    c.div_(Batch_size)
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    Bl = on_diag + lambd * off_diag
    return Bl