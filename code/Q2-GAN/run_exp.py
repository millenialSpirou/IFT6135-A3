# -*- coding: utf-8 -*-
r"""
Assignment 3/Practical Problem 2: Generative Adversatial Networks
=================================================================

Author: Christos Tsirigotis <christos.tsirigotis-at-umontreal-dot-ca>

For this problem you are requested to complete code in order to train
a Generative Adversarial Network (GAN). The code you need to complete is concerned
with the implementation of GAN losses and regularization schemes, as well as
their training procedure. You can find the code to be implemented in
`gan.py`. The models for the tasks at hand are already determined at `model.py`.

.. note::
   Generally, you are strongly encouraged to attempt solving **Problems 4 and 5**
   of the theoretical assignment before the practical GAN assignment.
   The material in those problems is going to help you in proceeding with this practical.

While developing you can use the publicly available tests as a first
way to check the validity of your implementation, regarding the input-output
Tensor shape manipulation.

To test your code locally, please execute on a terminal::

    python gan.py

We suggest that the first thing you attempt after implementation is to
execute various cases of the `dirac`-matching task, preferably in an
interactive Jupyter notebook environment, like the one provided by Google Colab.
The enclosed notebook `main.ipynb` provided in the assignment's repository
has everything you need to do this.

Using the `dirac` task, you should be able to demonstrate to yourself empirical
evidence of the validity of your implementation. Your predictions from Problem
5 of the theoretical assignment should match your observations.

After making sense of theoretical and empirical results together for the dirac
matching task, you can proceed to the CIFAR10 image generation task.
We ask to train a specific set of hyperparameter configurations whose results
you need to report. Results include curves of Fréchet Inception Distance (FID)
at evaluation, along generator's training loss (metric), average duration of
training iteration (make sure experiments run on the same machine for this to
be comparable), and final samples (at 100000 generator updates). You will be
requested to interpret your findings.

Excerpt of predefined configurations in `main.ipynb`::

    from functools import partial
    Dirac_Hps = partial(Hyperparameters, max_iters=5000, dirac_target=0., task='dirac',
                        optimizer='SGD', generator_lr=0.1, critic_lr=0.1, critic_wd=0.,
                        critic_inner_iters=1)
    configs = {
        'dirac-jsd-1': Dirac_Hps(loss_type='JSD', generator_alpha_ema=None),
        'dirac-jsd-2': Dirac_Hps(loss_type='JSD', critic_reg_cf=0.01, critic_reg_type='R1', generator_alpha_ema=None),
        'dirac-jsd-3': Dirac_Hps(loss_type='JSD', critic_reg_cf=0.1, critic_reg_type='R1', generator_alpha_ema=None),
        'dirac-jsd-4': Dirac_Hps(loss_type='JSD', critic_reg_cf=0.01, critic_reg_type='GP', generator_alpha_ema=None),
        'dirac-jsd-5': Dirac_Hps(loss_type='JSD', critic_reg_cf=0.1, critic_reg_type='GP', generator_alpha_ema=None),
        'dirac-w1-1': Dirac_Hps(loss_type='W1', generator_alpha_ema=None),
        'dirac-w1-2': Dirac_Hps(loss_type='W1', critic_reg_cf=0.01, critic_reg_type='R1', generator_alpha_ema=None),
        'dirac-w1-3': Dirac_Hps(loss_type='W1', critic_reg_cf=0.1, critic_reg_type='R1', generator_alpha_ema=None),
        'dirac-w1-4': Dirac_Hps(loss_type='W1', critic_reg_cf=0.01, critic_reg_type='GP', generator_alpha_ema=None),
        'dirac-w1-5': Dirac_Hps(loss_type='W1', critic_reg_cf=0.1, critic_reg_type='GP', generator_alpha_ema=None),
        #
        'dirac-jsd-6': Dirac_Hps(loss_type='JSD'),
        'dirac-jsd-7': Dirac_Hps(loss_type='JSD', critic_reg_cf=0.01, critic_reg_type='R1'),
        'dirac-jsd-8': Dirac_Hps(loss_type='JSD', critic_reg_cf=0.1, critic_reg_type='R1'),
        'dirac-jsd-9': Dirac_Hps(loss_type='JSD', critic_reg_cf=0.01, critic_reg_type='GP'),
        'dirac-jsd-10': Dirac_Hps(loss_type='JSD', critic_reg_cf=0.1, critic_reg_type='GP'),
        'dirac-w1-6': Dirac_Hps(loss_type='W1'),
        'dirac-w1-7': Dirac_Hps(loss_type='W1', critic_reg_cf=0.01, critic_reg_type='R1'),
        'dirac-w1-8': Dirac_Hps(loss_type='W1', critic_reg_cf=0.1, critic_reg_type='R1'),
        'dirac-w1-9': Dirac_Hps(loss_type='W1', critic_reg_cf=0.01, critic_reg_type='GP'),
        'dirac-w1-10': Dirac_Hps(loss_type='W1', critic_reg_cf=0.1, critic_reg_type='GP'),
        #
        'cifar10-jsd-1': Hyperparameters(loss_type='JSD'),
        'cifar10-jsd-2': Hyperparameters(loss_type='JSD', critic_use_sn=True),
        'cifar10-jsd-3': Hyperparameters(loss_type='JSD', critic_reg_cf=1., critic_reg_type='R1'),
        'cifar10-w1-2': Hyperparameters(loss_type='W1', critic_use_sn=True),
        'cifar10-w1-3': Hyperparameters(loss_type='W1', critic_reg_cf=1., critic_reg_type='R1'),
    }

To get help regarding command line arguments in this executable Python file, use::

    python run_exp.py --help

"""
import os
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import (Optional, Text, Tuple)

import torch
import torchvision
from matplotlib import pyplot as plt

from data import Dataset
import optim
import model
import dirac
import gan
from utils import (State, config_logger)
from utils.train import (accumulator, running_average_meter)
from utils.meta import SingletonError


Vector = Tuple[float]


@dataclass
class Management:
    data_path: Text = os.path.abspath('datasets')  # Directory where datasets will be downloaded and reside
    exp_path: Text = os.path.abspath('experiments')  # Directory where experiment results, samples and logs will be
    exp_name: Text = 'default-exp'  # An identifier for the experiment to be run
    log_every: int = 500  # How often training info will be logged per number of generator updates
    eval_every: int = 2000  # How often evaluation occurs per number of generator updates
    viz: bool = False  # Present visualizations during evaluation and post-training
    save_logs: bool = True  # If True, save logs to `log_path`
    deterministic: bool = False
    cuda: int = 0  # Use CUDA device 0, if available
    preload_to_gpu: bool = True  # If True, then preload entire dataset to GPU, if used
    num_workers: int = 2  # Number of batch assembling workers

    @property
    def log_file(self):
        return os.path.join(self.log_path, 'log.txt')

    @property
    def log_path(self):
        return os.path.join(self.exp_path, self.exp_name)


@dataclass
class Hyperparameters:
    task: Text = 'CIFAR10'  # Choice of tasks {'CIFAR10', 'dirac'}
    batch_size: int = 64  # Number of samples in a training/test batch
    seed: int = 31415  # Fix this for reproducibility
    dirac_target: float = 0.

    generator_dimz: int = 100  # Number of dimensions of iid gaussian noise in generator's input
    generator_dimh: int = 128  # Width of the layers
    generator_alpha_ema: Optional[float] = 0.998  # If not None, apply exponential moving average to generator's parameters by this weight
    critic_dimh: int = 128  # Width of the layers
    critic_use_sn: bool = False  # Substitute critic's layers with spectrally normalized versions

    optimizer: Text = 'Adam'  # or SGD
    generator_lr: float = 2e-4  # Learning rate
    generator_betas: Vector = (0., 0.99)  # First is used as momentum/beta1, second is beta2 in Adam
    generator_wd: float = 0.
    critic_lr: float = 4e-4
    critic_betas: Vector = (0., 0.99)
    critic_wd: float = 1e-6

    loss_type: Text = 'JSD'  # Name of adversarial objectives used. Choice from {'JSD', 'W1'}
    critic_reg_type: Optional[Text] = None  # Name of critic regularization. Choice from {None, 'GP', 'R1'}
    critic_reg_cf: float = 0.1  # Coefficient for critic's regularization
    critic_inner_iters: int = 2  # Number of critic updates per generator update
    max_iters: int = 100000  # Maximum number of generator updates


def main(management, hps):
    # Setup Experiment
    task = hps.task
    config_logger(management.log_file, saving=management.save_logs)
    logger = logging.getLogger('Exp')
    train_logger = logging.getLogger('Exp.Train')
    eval_logger = logging.getLogger('Exp.Eval')
    try:
        state = State(hps.seed, management)
    except SingletonError:
        State.instance = None
        state = State(hps.seed, management)

    logger.info(f"Initializing experiment `{management.exp_name}` with hyperparameters:\n%s",
                repr(hps))
    stats = accumulator()

    # Setup Data
    if task == 'dirac':
        def train_data():
            dirac = State().convert(torch.Tensor(1, 1).fill_(hps.dirac_target))
            while True:
                yield dirac
        train_iter = train_data()
    else:
        dataset_cfg = dict(type=task,
                           root=management.data_path, download=True,
                           preload_to_gpu=management.preload_to_gpu,
                           num_threads=management.num_workers,
                           batch_size=hps.batch_size,
                           )
        train_data = Dataset(**dataset_cfg, mode='train')
        eval_data = Dataset(**dataset_cfg, mode='test')
        train_iter = train_data.sampler(infinite=True, project=0)

    # Setup Generator
    if task == 'dirac':
        generator = dirac.DiracGenerator()
        stats.g_params.append(generator.param.clone().detach().cpu())
    else:
        generator = model.Generator(dimz=hps.generator_dimz,
                                    dimh=hps.generator_dimh,
                                    default_batch_size=hps.batch_size)
    test_generator = generator
    if hps.generator_alpha_ema is not None:
        test_generator = deepcopy(generator)
        test_generator.to(device=State().device)
        test_generator.train()
    generator.to(device=State().device)
    generator.train()
    generator_optim = optim.init_optimizer(generator.parameters(),
                                          type=hps.optimizer,
                                          lr=hps.generator_lr,
                                          betas=hps.generator_betas,
                                          wd=hps.generator_wd)
    logger.info("Generator:\n%s", generator)

    # Setup Critic
    if task == 'dirac':
        critic = dirac.DiracCritic()
        stats.c_params.append(critic.param.clone().detach().cpu())
    else:
        critic = model.Critic(dimh=hps.critic_dimh, sn=hps.critic_use_sn)
    critic.to(device=State().device)
    critic.train()
    critic_optim = optim.init_optimizer(critic.parameters(),
                                        type=hps.optimizer,
                                        lr=hps.critic_lr,
                                        betas=hps.critic_betas,
                                        wd=hps.critic_wd)
    logger.info("Critic:\n%s", critic)

    # Train
    step = 0
    train_loss_meter = running_average_meter()
    train_step = gan.make_train_step(hps.loss_type,
                                     critic_inner_iters=hps.critic_inner_iters,
                                     reg_type=hps.critic_reg_type,
                                     reg_cf=hps.critic_reg_cf,
                                     alpha_ema=hps.generator_alpha_ema)
    if task != 'dirac':
        eval_step = gan.make_eval_step(os.path.join(management.exp_path, task + '_inception_stats.npz'),
                                       eval_data.sampler(infinite=False, project=0),
                                       hps.generator_dimz, persisting_Z=100,
                                       device=State().device)

    logger.info("Training")
    while True:
        if step >= hps.max_iters: break
        step += 1
        train_loss = train_step(train_iter, critic, critic_optim,
                                generator, test_generator, generator_optim)
        train_loss_meter.update(train_loss.clone().detach())

        if step % management.log_every == 0 and task != 'dirac':
            train_logger.info("step %d | loss(%s) %.3f (%.3f)",
                              step, hps.loss_type,
                              train_loss_meter.avg.item(), train_loss_meter.val.item())
        if task == 'dirac':
            stats.g_params.append(test_generator.param.clone().detach().cpu())
            stats.c_params.append(critic.param.clone().detach().cpu())

        if step % management.eval_every == 0 and task != 'dirac':
            eval_iter = eval_data.sampler(infinite=False, project=0)
            samples, results = eval_step(eval_iter, critic, test_generator)
            if management.viz:
                from IPython.display import clear_output, display, update_display
                grid_img = torchvision.utils.make_grid(samples,
                                                       nrow=10, normalize=True, value_range=(-1., 1.), padding=0)
                plt.imshow(grid_img.permute(1, 2, 0).cpu())
                display(plt.gcf())
            eval_logger.info("step %d | " + ' | '.join([f'{k} {v:.3f}' for k, v in results.items()]), step)
            torchvision.utils.save_image(samples.cpu(), os.path.join(management.log_path, f'samples-{step}.png'),
                                         nrow=10, normalize=True, value_range=(-1., 1.), padding=0)

    logger.info("Final Evaluation")
    if task == 'dirac':
        g_params = torch.stack(stats.g_params)
        c_params = torch.stack(stats.c_params)
        trajectory = torch.cat([c_params, g_params], dim=-1).numpy()
        logger.info(f"Final point in parameter space: {trajectory[-1]}")
        anima = dirac.animate(trajectory, hps)
        if management.viz:
            from IPython.display import HTML, display
            display(HTML(anima.to_html5_video()))
        anima.save(os.path.join(management.log_path, 'evolution.mp4'))
    else:
        eval_iter = eval_data.sampler(infinite=False, project=0)
        samples, results = eval_step(eval_iter, critic, test_generator)
        logger.info("step %d | " + ' | '.join([f'{k} {v:.3f}' for k, v in results.items()]), step)
        torchvision.utils.save_image(samples, os.path.join(management.log_path, f'samples-final.png'),
                                     nrow=10, normalize=True, value_range=(-1., 1.), padding=0)


if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser("Assignment 3, Practical Problem 2: Generative Adversarial Networks")
    parser.add_arguments(Management, dest="management")
    parser.add_arguments(Hyperparameters, dest="hyperparameters")
    args = parser.parse_args()
    main(args.management, args.hyperparameters)
