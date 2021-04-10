import torch
import math
from torch import Tensor
from typing import Tuple


class PlanarTransformation(torch.nn.Module):
    """
    One layer of a planar flow. It can perform a planar transformation and
    return its log determinant (it also makes sure the function is invertible
    by adjusting u in the case where uw^t < -1)
    """
    def __init__(self, dim: int = 2):
        super().__init__()
        self.u = torch.nn.Parameter(torch.normal(0, 0.01, (1, dim)))
        self.w = torch.nn.Parameter(torch.normal(0, 0.01, (1, dim)))
        self.b = torch.nn.Parameter(torch.zeros((1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a planar transformation on x.
        :param x: the input, a tensor of size (batch_size, dim)
        :returns: the output, a tensor of size (batch_size, dim)
        """
        if torch.matmul(self.u, self.w.T) < -1:
            self.adjust_u()

        z = torch.matmul(torch.transpose(self.u, 0, 1), torch.tanh(torch.matmul(self.w, torch.transpose(x, 0, 1)) + self.b)) 
        x_ = x + z.permute(1,0)
        return x_

    def get_logdet(self, x: torch.Tensor, eps:float = 1e-8) -> torch.Tensor:
        """
        Calculate the log determinant of a planar tranformation on x
        :param x: the input, a tensor of size (batch_size, dim)
        :param eps: a small value used for numerical stability
        :returns: the output, a tensor of size (batch_size, 1)
        """
        if torch.mm(self.u, self.w.T) < -1:
            self.adjust_u()

        d1 = 1 - torch.square(
                torch.tanh(
                    torch.matmul(
                        self.w, torch.transpose(x, 0, 1)) + self.b))
        d2 = torch.matmul(torch.transpose(self.w, 0, 1), d1)
        det = (1 + torch.matmul(self.u, d2)).permute(1, 0)
        
        return torch.log(eps + det)

    def adjust_u(self):
        """
        A function that adjust u to make sure the planar transformation is invertible
        """
        wu = torch.mm(self.u, self.w.T)
        m_wu = -1 + torch.log(1 + torch.exp(wu))
        self.u.data = self.u + torch.mm((m_wu - wu), self.w) / torch.norm(self.w, p=2, dim=1) ** 2


class Flow(torch.nn.Module):
    """
    A planar flow constituted of 'num_layers' planar transformations.
    Return the transformed data and the log determinant of the transformation.
    """
    def __init__(self, dim: int, num_layers: int):
        """
        :param dim: dimensionality of each transformation
        :param num_layers: total number of planar transformation
        """
        super().__init__()
        self.layers = torch.nn.ModuleList([PlanarTransformation(dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        """
        :param x: the input, a tensor of size (batch_size, dim)
        :returns: x, the transformed input and
                  logdet, the log determinant of the complete transformation
        """
        logdet = 0

        for layer in self.layers:
            # 6. WRITE YOUR CODE HERE
            pass

        return x, logdet


def loss1(z0: torch.Tensor, logdet: torch.Tensor) -> torch.Tensor:
    """
    Calculate the negative likelihood of a normal (0,I) that has been transformed by a flow
    :param z0: a tensor of size (batch_size, dim)
    :param logdet: a tensor of size (batch_size, 1)
    :returns: the log likelihood for each z0, thus a tensor of size (batch_size, 1)
    """
    # 6. WRITE YOUR CODE HERE
    dim = z0.size(1)
    d = torch.distributions.MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    nll = torch.unsqueeze((- (d.log_prob(z0) + logdet)), 1)

    return nll


def loss2(target_density: torch.Tensor, z0: torch.Tensor, logdet: torch.Tensor) -> torch.Tensor:
    """
    Calculate the loss to train a flow where we don't use samples, but we have access to the target distribution
    :param target_density: a tensor of size (batch_size)
    :param z0: a tensor of size (batch_size, dim)
    :param logdet: a tensor of size (batch_size, 1)
    :returns: the log likelihood for each z0, thus a tensor of size (batch_size, 1)
    """
    # 8. WRITE YOUR CODE HERE
    dim = z0.size(1)
    d = torch.distributions.MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    loss = torch.unsqueeze(target_density + d.log_prob(z0) - logdet, 1)
    return loss
