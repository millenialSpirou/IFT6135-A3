# -*- coding: utf-8 -*-
r"""
:mod:`gan` -- Generative Adversarial Network procedures
=======================================================

.. module:: gan
   :platform: Unix
   :synopsis: Implements losses, regularization, train and eval steps for GANs

.. info::
   Student version

"""
from typing import (Text, Optional, Callable, Iterator)

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


AVAILABLE_LOSSES = ['JSD', 'W1']
AVAILABLE_REGULARIZERS = ['GP', 'R1', None]


def jsd(cp: Tensor, cq: Tensor) -> Tensor:
    r"""Approximate the Jensen-Shannon divergence.

    Given a set of values from a critic, this function calculates
    an approximation to the Jensen-Shannon divergence between P and Q,
    :math:`\text{JSD}(P, Q)`.

    .. note::
       **By convention**, critic targets positive values for real data
       in this function.
    .. seealso::
       The code here should be aligned with the answer you gave in
       question 4.1 of the theoretical assignment.

    Args:
        cp: A tensor of size ``(N,)`` containing evaluations of the critic at
            samples of :math:`P`, the real distribution.
        cq: A tensor of size ``(N,)`` containing evaluations of the critic at
            samples of :math:`Q`, the fake distribution.

    Returns:
        The estimated Jensen-Shannon divergence, a scalar tensor of size ``()``.

    """
    expected1 = torch.mean(torch.log(torch.sigmoid(cp)))
    expected2 = torch.mean(torch.log(torch.sigmoid(-cq)))
    expected = expected1 + expected2

    jsd = (expected + np.log(4)) / 2 

    return jsd



def w1(cp: Tensor, cq: Tensor) -> Tensor:
    r"""Approximate the 1-Wasserstein distance.

    Given a set of values from a critic, this function calculates
    an approximation to the 1-Wasserstein distance between P and Q,
    :math:`\text{W1}(P, Q)`.

    .. note::
       It is assumed by the definition of the Wasserstein-Kantorovich metric that
       the critic is a 1-Lipschitz function. Be sure the critic model is under the
       effect of appropriate capacity constraints when using this.
    .. note::
       **By convention**, critic targets positive values for real data
       in this function.

    Args:
        cp: A tensor of size ``(N,)`` containing evaluations of the critic at
            samples of :math:`P`, the real distribution.
        cq: A tensor of size ``(N,)`` containing evaluations of the critic at
            samples of :math:`Q`, the fake distribution.

    Returns:
        The estimated 1-Wasserstein distance, a scalar tensor of size ``()``.

    """
    return torch.mean(cp) - torch.mean(cq)


def r1gp(p: Tensor, cp: Tensor,
         q: Tensor, cq: Tensor, critic: Callable[[Tensor], Tensor]) -> Tensor:
    r"""Implement the R1, zero-centered, Gradient Penalty.

    This gradient penalty is a zero-centered gradient penalty applied to the
    real data.

    .. math::
       \mathcal{R}_1(\psi) = \mathbb{E}_{p \sim P} \|\nabla_x C_\psi(p)\|^2

    .. note::
       Possibly, not all inputs are needed in the implementation. Use whatever
       is necessary and convenient for your implementation.
    .. note::
       By convention, `p` are samples from real data, and `q` generated
       samples.
    .. note::
       This function need to work for arbitrary sample tensor shapes, so that
       it can deal both with the 'dirac' and 'CIFAR10' tasks!

    Args:
        p:  A tensor of size ``(N, *sample.size())`` which contain samples of
            :math:`P`, the real distribution.
        cp: A tensor of size ``(N,)`` containing evaluations of the critic at
            `p`, samples of :math:`P`, the real distribution.
        q:  A tensor of size ``(N, *sample.size())`` which contain samples of
            :math:`Q`, the fake distribution.
        cq: A tensor of size ``(N,)`` containing evaluations of the critic at
            `q`, samples of :math:`Q`, the fake distribution.
        critic: The module which makes `p` into `cp`, and `q` into `cq`. A function
            approximating the critic.

    Returns:
        :math:`\mathcal{R}_1` regularization term, a scalar tensor of size ``()``.

    """
    cp = critic(p)
    grad = torch.autograd.grad(cp, p,
            grad_outputs=torch.ones_like(cp), retain_graph=True,
            create_graph=True, only_inputs=True)[0]

    grad_norm = torch.square(torch.norm(grad, 2, dim=-1))
    
    return torch.mean(grad_norm)


def ogp(p: Tensor, cp: Tensor,
        q: Tensor, cq: Tensor, critic: Callable[[Tensor], Tensor]) -> Tensor:
    r"""Implement the Original, one-centered, Gradient Penalty.

    This gradient penalty is a one-centered gradient penalty applied to a
    manifold mix-up between the real and fake data.

    .. math::
       \mathcal{G}(\psi, \theta) = \mathbb{E}_{\epsilon \sim U} \mathbb{E}_{p \sim P}
       \mathbb{E}_{q \sim Q_\theta} (\|\nabla_x C_\psi(\epsilon p + (1 - \epsilon) q)\| - 1)^2

    .. note::
       Possibly, not all inputs are needed in the implementation. Use whatever
       is necessary and convenient for your implementation.
    .. note::
       By convention, `p` are samples from real data, and `q` generated
       samples.
    .. note::
       This function need to work for arbitrary sample tensor shapes, so that
       it can deal both with the 'dirac' and 'CIFAR10' tasks!

    Args:
        p:  A tensor of size ``(N, *sample.size())`` which contain samples of
            :math:`P`, the real distribution.
        cp: A tensor of size ``(N,)`` containing evaluations of the critic at
            `p`, samples of :math:`P`, the real distribution.
        q:  A tensor of size ``(N, *sample.size())`` which contain samples of
            :math:`Q`, the fake distribution.
        cq: A tensor of size ``(N,)`` containing evaluations of the critic at
            `q`, samples of :math:`Q`, the fake distribution.
        critic: The module which makes `p` into `cp`, and `q` into `cq`. A function
            approximating the critic.

    Returns:
        :math:`\mathcal{G}` regularization term, a scalar tensor of size ``()``.

    """
    eps = torch.rand(p.size(0)).unsqueeze(1)
    xhat = eps * p + (1 - eps) * q
    f_xhat = critic(xhat)

    grad = torch.autograd.grad(f_xhat, xhat,
            grad_outputs=torch.ones_like(f_xhat), retain_graph=True,
            create_graph=True, only_inputs=True)[0]

    
    grad_norm = torch.square(torch.norm(grad, 2, dim=-1) - torch.ones(p.size(0)))
    
    return torch.mean(grad_norm)


def make_losses(loss_type: Text,
                reg_type: Optional[Text]=None, reg_cf: float=1.):
    f"""Construct losses for adversarial training.

    For adversarial training we need two losses, one for the generator and one
    for the discriminator/critic. Use the hyperparameters given as arguments
    to build the functions we are going to use in our training.

    Args:
        loss_type: A string declaring which pair of losses should be used
            out of the {AVAILABLE_LOSSES}.
        reg_type: A string declaring which regularization function we should
            use for the critic out of the {AVAILABLE_REGULARIZERS}.
        reg_cf: A positive number which multiplies the outcome of critic's
            regularization function described by `reg_type`. The regularization
            coefficient.

    Returns:
        A pair of losses (`metric`, `critic_loss`):
         - `metric` is a callable which calculates the distance (JSD, W1, etc)
           between 2 distributions, :math:`P` and :math:`Q`.
         - `critic_loss` is a callable which calculate the loss that the
           critic must minimize.
    """
    # XXX DO NOT ALTER THIS PART
    assert(reg_cf > 0)

    if loss_type == 'JSD':
        metric_head = jsd
    elif loss_type == 'W1':
        metric_head = w1
    else:
        raise NotImplementedError(f'{loss_type} not in {AVAILABLE_LOSSES}')

    if reg_type == 'R1':
        reg_fn = r1gp
    elif reg_type == 'GP':
        reg_fn = ogp
    elif reg_type is None:
        reg_fn = lambda p, cp, q, cq, critic: 0.
    else:
        raise NotImplementedError(f'{reg_type} not in {AVAILABLE_REGULARIZERS}')
    # END XXX

    def metric(p: Tensor, q: Tensor,
               critic: Callable[[Tensor], Tensor]) -> Tensor:
        r"""Approximate a metric between two distributions, :math:`P` and :math:`Q`.

        Combine calls to `critic` and the `metric_head` above

        Args:
            p: Samples from :math:`P`. Size: ``(N, *sample.size())``
            q: Samples from :math:`Q`. Size: ``(N, *sample.size())``
            critic: :math:`\mathbb{R^d} \to \mathbb{R}`

        Returns:
            Approximated metric given by `loss_type`,
            a scalar tensor of size ``()``.
        """
        return metric_head(critic(p), critic(q))


    def critic_loss(p: Tensor, q: Tensor,
                    critic: Callable[[Tensor], Tensor]) -> Tensor:
        r"""Approximate a metric between two distributions, :math:`P` and :math:`Q`.

        Combine calls to `critic`, `metric_head` and `reg_fn`. Use `reg_cf`.

        Args:
            p: Samples from :math:`P`. Size: ``(N, *sample.size())``
            q: Samples from :math:`Q`. Size: ``(N, *sample.size())``
            critic: :math:`\mathbb{R^d} \to \mathbb{R}`

        Returns:
            Loss for the `critic`, a scalar tensor of size ``()``.
        """
        cp = critic(p)
        cq = critic(q)
        return  - metric_head(cp, cq) + reg_cf * reg_fn(p, cp, q, cq, critic)

    return metric, critic_loss


def apply_exponential_moving_average(source: nn.Module, target: nn.Module,
                                     alpha: float=0.998):
    r"""Apply the exponential moving average update to a target model.

    This function updates the parameters of `target` network using an
    exponential moving average of the parameters of a `source` network.
    In other words it implements:

    .. math::
       \tilde{\theta}_{k+1} = \alpha * \tilde{\theta}_k + (1 - \alpha) * \theta_{k+1}

    where :math:`\theta` are the parameters of `source` network and
    :math:`\tilde{\theta}` the parameters of `target` network.

    We use this to obtain a better and more stable generator model in GANs.

    .. info::
       In PyTorch, you can get access to a dictionary containing entries
       of (names, module's parameters) by invoking ``model.named_parameters()``.

    Args:
        source: A model whose exponential moving average parameters across
           training time we want to have.
        target: A cloned version of `source` which holds the exponential
           moving averaged parameters.
        alpha: The weight by which we apply exponential moving average,
           :math:`\alpha` in the formula above.

    Returns:
        None (all updates in the `target` model must happen inplace)

    """
    with torch.no_grad():
        t = target.state_dict()
        s = source.state_dict()
        for k in t:
            param_t = t[k]
            param_s = s[k]
            t[k] = alpha * param_t + (1 - alpha) * param_s

        target.load_state_dict(t)


def make_train_step(loss_type: Text,
                    critic_inner_iters: int=1,
                    reg_type: Optional[Text]=None, reg_cf: float=1.,
                    alpha_ema: Optional[float]=None):
    """Construct a train step for a generative adversarial network.

    Use the hyperparameters given as arguments to build a routine which
    performs a single iteration of alternating stochastic gradient
    ascent-descent.

    Both the `critic` and `generator` are to be optimized in this routine.

    Args:
        loss_type: A string declaring which pair of losses should be used
            out of the {AVAILABLE_LOSSES}.
        critic_inner_iters: Number of optimization steps for the critic per
            optimization step for the generator.
        reg_type: A string declaring which regularization function we should
            use for the critic out of the {AVAILABLE_REGULARIZERS}.
        reg_cf: A positive number which multiplies the outcome of critic's
            regularization function described by `reg_type`. The regularization
            coefficient.
        alpha_ema: If not `None`, then we apply exponential moving average to
            a test generator according to this weight.

    Returns:
        A `train_step` to iterate over during training.

    """
    # XXX DO NOT ALTER THIS PART
    assert(critic_inner_iters >= 1)
    assert(alpha_ema is None or (alpha_ema > 0 and alpha_ema < 1))
    metric, critic_loss = make_losses(loss_type, reg_type, reg_cf)
    # END XXX

    def train_step(train_data_iter: Iterator[Tensor],
                   critic: Callable[[Tensor], Tensor],
                   critic_optim: torch.optim.Optimizer,
                   generator: Callable[[Optional[Tensor]], Tensor],
                   test_generator: nn.Module,
                   generator_optim: torch.optim.Optimizer) -> Tensor:
        """Implement a training step.

        Use ``next(train_data_iter)`` to get a batch of train data from
        the target dataset.
        Use ``generator()`` to get a batch of generated data from
        the generator model.
        Call them multiple times to implement update steps for the critic and
        the generator.

        Args:
            train_data_iter: An infinite iterator over the training dataset.
            critic: A function mapping the data space to the reals,
                :math:`\\mathbb{R^d} \to \\mathbb{R}`.
            critic_optim: PyTorch optimizer responsible for the
                parameters of `critic`.
            generator: A function mapping noise to the data space,
                :math:`\\mathbb{R^z} \to \\mathbb{R^d}`.
            test_generator: Another function mapping noise to the data space,
                updates its parameters by the exponential moving average of
                the parameters of `generator` model.
            generator_optim: PyTorch optimizer responsible for the
                parameters of `generator`.

        Returns:
            The metric minimized by the generator calculated over a batch of data,
            a scalar tensor of size ``()``.
        """
        # TODO
        true_X = next(train_data_iter)
        fake_X = generator()
        raise NotImplementedError("Implement as part of assignment 3.")

    return train_step


def make_eval_step(stats_filename: Text, eval_data_iter: Iterator[Tensor],
                   dimz: int, persisting_Z: int=100, device=None):
    """Construct an evaluation step for a generative adversarial network.

    Args:
        stats_filename: Path to file where precomputed inception statistics
            for the task in hand are to be saved/found.
        eval_data_iter: A finite iterator over the validation dataset.
        dimz: Number of dimensions in the iid noise given to generator.
        persisting_Z: Number of noise vectors to be kept for validation.
        device (Optional[torch.device]): device which will perform validation.

    Returns:
        An `eval_step` to iterate periodically over during training.
    """
    from utils.train import (eval_ctx, average_meter)
    import score
    inception_net = score.load_inception_net().to(device=device)
    score.prepare_inception(stats_filename, eval_data_iter, net=inception_net)
    persisting_Z = torch.randn(persisting_Z, dimz)

    def eval_step(eval_data_iter: Iterator[Tensor],
                  critic: Callable[[Tensor], Tensor],
                  generator: Callable[[Optional[Tensor]], Tensor]):
        # Warmup generator, to calculate appropriate batch statistics (bc of BN)
        generator.train()
        for mod in generator.modules():
            try:
                mod.reset_running_stats()
            except AttributeError:
                pass
        for _ in range(1000):
            generator()

        average_cp = average_meter()
        average_cq = average_meter()
        with eval_ctx(critic, generator):
            # Run samples through the critic
            for p in eval_data_iter:
                q = generator()
                cp = critic(p)
                cq = critic(q)
                average_cp.update(cp)
                average_cq.update(cq)

            # Get generated samples from persiting test noise
            samples = generator(persisting_Z.to(device=device))

            # Get Inception Score and Fréchet Inception Distance
            IS, _, _, FID = score.get_scores(generator,
                                             test_filename=stats_filename,
                                             net=inception_net)

        results = dict(IS=float(IS), FID=float(FID),
                       cp=float(average_cp.avg), cq=float(average_cq.avg))

        return samples, results

    return eval_step


######  PUBLIC TESTS  ######

import pytest


def test_jsd():
    assert(jsd(torch.randn(5), torch.randn(5)).size() == torch.Size([]))


def test_w1():
    assert(w1(torch.randn(5), torch.randn(5)).size() == torch.Size([]))


def test_r1gp():
    a = torch.randn(5).requires_grad_()
    b = a + 10
    c = lambda x: x + 5
    assert(r1gp(a, b, a, b, c).size() == torch.Size([]))


def test_r1gp_2():
    a = torch.randn(5, 3, 4, 4).requires_grad_()
    b = a.mean((-1, -2, -3))
    c = lambda x: x.sum((-1, -2, -3))
    assert(r1gp(a, b, a, b, c).size() == torch.Size([]))


def test_ogp():
    a = torch.randn(5).requires_grad_()
    b = a + 10
    c = lambda x: x + 5
    assert(ogp(a, b, a, b, c).size() == torch.Size([]))


def test_ogp_2():
    a = torch.randn(5, 3, 4, 4).requires_grad_()
    b = a.mean((-1, -2, -3))
    c = lambda x: x.sum((-1, -2, -3))
    assert(ogp(a, b, a, b, c).size() == torch.Size([]))


@pytest.mark.parametrize("loss_type", ['JSD', 'W1'])
@pytest.mark.parametrize("reg_type", [None, 'GP', 'R1'])
def test_make_losses(loss_type, reg_type):
    metric, critic_loss = make_losses(loss_type, reg_type=reg_type)
    critic = lambda x: x.mean(-1)
    assert(metric(torch.randn(5, 10), torch.randn(5, 10), critic).size() == torch.Size([]))
    assert(critic_loss(torch.randn(5, 10), torch.randn(5, 10), critic).size() == torch.Size([]))


def test_edge_loss_type():
    with pytest.raises(NotImplementedError):
        make_losses('asdfa', reg_type=None)


def test_edge_reg_type():
    with pytest.raises(NotImplementedError):
        make_losses('JSD', reg_type='asdfafa')


def test_edge_reg_cf():
    with pytest.raises(AssertionError):
        make_losses('JSD', reg_type='GP', reg_cf=-0.1)


def test_edge_critic_inner_iters():
    with pytest.raises(AssertionError):
        make_train_step('JSD', reg_type='GP', reg_cf=1., critic_inner_iters=0)


def test_edge_alpha_ema():
    with pytest.raises(AssertionError):
        make_train_step('JSD', reg_type='GP', reg_cf=1., critic_inner_iters=0,
                        alpha_ema=-1.)


def test_edge_alpha_ema2():
    with pytest.raises(AssertionError):
        make_train_step('JSD', reg_type='GP', reg_cf=1., critic_inner_iters=0,
                        alpha_ema=2.)


if __name__ == "__main__":
    import sys
    pytest.main(sys.argv)
