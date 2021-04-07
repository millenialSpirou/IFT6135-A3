import torch
import random
import numpy as np
from solution import Flow, loss1, loss2
from utils import density_arcs, density_sine, plot_density, plot_learned_density, plot_samples


def train_from_samples(model, optimizer, z0_distr, dataset, name: str,
                       num_layers: int, batch_size: int = 128, num_iter: int = 20000):
    """Train the normalizing flow from samples"""
    loss_sum = 0

    for i in range(num_iter + 1):
        # pick randomly 'batch_size' examples
        idx = np.random.choice(len(dataset), size=batch_size, replace=False)
        data = torch.tensor(dataset[idx])

        x, logdet = model(data)

        loss = torch.mean(loss1(x, logdet))
        loss_sum += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss average
        if i % 200 == 0 and i != 0:
            print(f"loss: {loss_sum/200}")
            loss_sum = 0

        # plot the learned density and save it
        if i == num_iter:
            model.eval()
            plot_learned_density(model, f"learned_density_{name}_l{num_layers}")


def train_from_density(model, optimizer, z0_distr, density, name: str,
                       num_layers: int, batch_size: int = 128, num_iter: int =
                       20000):
    """Train the normalizing flow from density"""
    loss_sum = 0

    for i in range(num_iter + 1):
        z0 = z0_distr.sample((batch_size,))

        x, logdet = model(z0)

        loss = torch.mean(loss2(density(x), z0, logdet))
        loss_sum += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss average
        if i % 200 == 0 and i != 0:
            print(f"loss: {loss_sum/200}")
            loss_sum = 0

        # plot samples and save it
        if i == num_iter:
            z0 = z0_distr.sample((500,))
            x, logdet = model(z0)
            x_ = x[:, 0].detach().numpy()
            y_ = x[:, 1].detach().numpy()
            plot_samples(x_, y_, f"samples_{name}_l{num_layers}")


def launch_experiments_from_density(density_dict: dict, num_layers_list: list, dim: int = 2,
                                    lr: float = 5e-4, num_iter: int = 20000):
    z0_distr = torch.distributions.MultivariateNormal(torch.zeros(dim), torch.eye(dim))

    for name, density in density_dict.items():
        plot_density(density, name)
        for num_layers in num_layers_list:
            print(f"Training on {name} density with {num_layers} layers")
            print("=" * 50)
            model = Flow(dim, num_layers)
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
            train_from_density(model, optimizer, z0_distr, density, name,
                               num_layers, num_iter=num_iter)


def launch_experiments_from_samples(density_dict: dict, num_layers_list: list, dim: int = 2,
                                    lr: float = 5e-4, num_iter: int = 20000):
    z0_distr = torch.distributions.MultivariateNormal(torch.zeros(dim), torch.eye(dim))

    for name, density in density_dict.items():
        name = f"data_{name}"
        data = np.load(f"{name}.npy")

        for num_layers in num_layers_list:
            print(f"Training on {name} with {num_layers} layers")
            print("=" * 50)
            model = Flow(dim, num_layers)
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
            train_from_samples(model, optimizer, z0_distr, data, name,
                               num_layers, num_iter=num_iter)


if __name__ == '__main__':
    torch.manual_seed(43)
    np.random.seed(43)
    random.seed(43)

    density_dict = {"arcs": density_arcs, "sine": density_sine}
    num_layers_list = [2, 8, 32]

    launch_experiments_from_density(density_dict, num_layers_list)
    # launch_experiments_from_samples(density_dict, num_layers_list)
