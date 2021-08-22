import numpy as np
import torch


def tilted_loss(q, y, f):
    """
    Quantile Regression Loss
    """
    e = (y-f)
    return (torch.maximum(q*e, (q-1)*e)).mean(0, keepdim=True)


def mdn_loss_fn(y, mu, sigma, pi):
    """
    Mixture Density Networks Loss
    """
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    loss = torch.exp(m.log_prob(y+1e-12))
    loss = torch.sum(loss*pi, dim=1)
    loss = -torch.log(loss+1e-12)
    return torch.mean(loss)


def mean_l1_norm(arr, bin=50):
    ans = [np.linalg.norm(x)/bin for x in arr]
    return ans
