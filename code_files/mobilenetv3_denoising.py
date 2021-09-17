import math
import os
import matplotlib.pyplot as plt
import time
import pickle
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import logging

from model import mobilenetv3
# ==================================================================================================================== #
""" Hyperparameters """
# Note: these may be change when using a pretrained model!
batch_size = 64
input_size = 128
scaling_depth = 4  # model will scale the images down by 2^scaling_depth during encoding
std = 0.2
lr = 5e-4
epochs = 100
random_seed = 42
pretrained_run_number = None  # None for training new model, number for loading pretrained model
filesPath = '/inputs/TAU/DL/'
dataPath = filesPath + '/data/ffhq/images'  # '/data/celeba/img_align_celeba'  # '/data/ffhq/images'

np.random.seed(random_seed)
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., device='cpu'):
        self.std = std
        self.mean = mean
        self.device = device

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).to(self.device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# ==================================================================================================================== #
if __name__ == '__main__':
    ######################################
    # Choose device
    ######################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print("Device: {}".format(torch.cuda.get_device_name(0)))
    print("Device type: {}".format(device))

    ######################################
    # Create new run folder
    ######################################
    runs_dirs = os.listdir(filesPath + 'outputs')
    curr_run = np.max([int(dir_name.split('run')[1]) for dir_name in runs_dirs]) + 1
    dest_path = filesPath + 'outputs/run' + str(curr_run) + '/'
    os.mkdir(dest_path)

    logging.basicConfig(filename=dest_path + 'log.log', filemode='w',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    ######################################
    # Use pre-trained nets
    ######################################
    if pretrained_run_number is not None:
        print('Loading pretrained model number ' + str(pretrained_run_number))
        with open(filesPath + 'outputs/run' + str(pretrained_run_number) + '/output_dict.pickle', 'rb') as handle:
            output_dict = pickle.load(handle)

        input_size = output_dict['input_size']
        std = output_dict['std']
        lr = output_dict['lr']
        epochs = output_dict['epochs']
        batch_size = output_dict['batch_size']
        scaling_depth = output_dict['scaling_depth']

        model = mobilenetv3(input_size=output_dict['input_size'], in_channels=3, scaling_depth=scaling_depth,
                            weights_dict=output_dict['net_dict'], device=device).to(device)

    ######################################
    # Get data and dataloaders
    ######################################
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    # transforms.CenterCrop(178),  # TODO: this may be a problem if we change to dataset with smaller images
                                    transforms.Resize(input_size),
                                    transforms.RandomHorizontalFlip()])

    dataset = datasets.ImageFolder(root=dataPath, transform=transform)
    train_inds, val_inds = train_test_split(torch.arange(len(dataset)), test_size=0.2)

    train_sampler = torch.utils.data.SubsetRandomSampler(train_inds)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_inds)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    ######################################
    # Train a model from scratch
    ######################################
    if pretrained_run_number is None:
        model = mobilenetv3(input_size=input_size, in_channels=3, scaling_depth=scaling_depth, device=device).to(device)
        noiser = AddGaussianNoise(std=std, device=device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_loss = []
        val_loss = []
        min_val_loss = math.inf

        print('Training Started! input_size: ', input_size)
        logging.info('Training Started! input_size: ' + str(input_size))
        for epoch in range(epochs):
            start_time = time.time()
            model.train()
            train_total = 0
            train_tot_loss = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                x_noised = noiser(x)
                output = model(x_noised)
                optimizer.zero_grad()
                loss = criterion(output, x)
                loss.backward()
                optimizer.step()
                train_tot_loss += loss.item()
                train_total += len(y)
            train_loss.append(train_tot_loss / len(train_loader))

            # validation evaluation:
            model.eval()
            val_total = 0
            val_tot_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    x_noised = noiser(x)
                    output = model(x_noised)
                    loss = criterion(output, x)
                    val_tot_loss += loss.item()
                    val_total += len(y)
                val_loss.append(val_tot_loss / len(val_loader))

            if val_loss[-1] < min_val_loss:
                min_val_loss = val_loss[-1]
                best_model_dict = model.state_dict()

            end_time = time.time()
            logging.info('[Epoch {}/{}] -> Train Loss: {:.4f}, Validation Loss: {:.4f}, Time: {:.1f}'.format(
                epoch + 1, epochs, train_loss[-1], val_loss[-1], end_time - start_time))

            print('[Epoch {}/{}] -> Train Loss: {:.4f}, Validation Loss: {:.4f}, Time: {:.1f}'.format(
                epoch + 1, epochs, train_loss[-1], val_loss[-1], end_time - start_time))

        # Save net dicts
        output_dict = {
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "scaling_depth": scaling_depth,
            "std": std,
            "input_size": input_size,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "net_dict": best_model_dict
        }

        with open(dest_path + 'output_dict.pickle', 'wb') as handle:
            pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Plot training losses:
        plt.figure(figsize=(10, 8))
        plt.plot(range(1, epochs + 1), train_loss, label='Train')
        plt.plot(range(1, epochs + 1), val_loss, label='Validation')
        plt.title('Training L2 Loss', fontsize=18)
        plt.xlabel('Epoch', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.grid()
        plt.legend(fontsize=18)
        plt.tight_layout()
        plt.savefig(dest_path + 'convergence.png')

    ######################################
    # Plots
    ######################################
    x, y = next(iter(val_loader))
    noiser = AddGaussianNoise(std=std, device='cpu')
    x_noised = noiser(x)
    out = model(x_noised.to(device)).cpu()
    all_images_modes = torch.cat([torch.stack([x[idx], x_noised[idx], out[idx]]) for idx in range(len(x))], dim=0)
    grid = torchvision.utils.make_grid(
        (all_images_modes - torch.min(all_images_modes)) / (torch.max(all_images_modes) - torch.min(all_images_modes)),
        nrow=9)
    plt.figure(figsize=(30, 30))
    plt.imshow(grid.permute(1, 2, 0))
    plt.tight_layout()
    plt.savefig(dest_path + 'output_grid.png')
