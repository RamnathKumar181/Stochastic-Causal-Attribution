import torch
import torch.nn.functional as F


class MDN(torch.nn.Module):
    def __init__(self, n_feature, n_gaussians):
        super(MDN, self).__init__()
        self.z_h = torch.nn.Sequential(
            torch.nn.Linear(n_feature, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU()
        )
        self.z_pi = torch.nn.Linear(50, n_gaussians)
        self.z_mu = torch.nn.Linear(50, n_gaussians)
        self.z_sigma = torch.nn.Linear(50, n_gaussians)

    def forward(self, x):
        z_h = self.z_h(x)
        pi = F.softmax(self.z_pi(z_h), -1)
        mu = self.z_mu(z_h)
        sigma = torch.exp(self.z_sigma(z_h))
        return pi, mu, sigma
