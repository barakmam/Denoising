import math
import os
import matplotlib.pyplot as plt
import time
import pickle
import numpy as np
import pandas as pd
import scipy.stats

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import logging

from model import mobilenetv3
from torch.utils.tensorboard import SummaryWriter

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
pretrained_run_number = None  # 15  # None for training new model, number for loading pretrained model
train_model = True  # False True
use_cpu = False

filesPath = '/inputs/TAU/DL/'
data_type = 'ffhq'  # 'celeba'
dataPath = filesPath + '/data/' + data_type + '/images'  #


np.random.seed(random_seed)
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., device=torch.device('cpu')):
        self.std = std
        self.mean = mean
        self.device = device

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).to(self.device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def test_time(dataloader, model, noiser, device):
    # validation evaluation:
    model.eval()
    val_total = 0
    batches_time = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            x_noised = noiser(x).to(device)
            t_start = time.time()
            _ = model(x_noised)
            time_took = time.time() - t_start
            batches_time.append(time_took)
            val_total += len(y)
    avg = sum(batches_time)/val_total
    std = np.std(batches_time)/dataloader.batch_size
    return avg, std


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def write_dict_to_file(src_dict, file_name):
    with open(file_name, 'w') as f:
        for key, value in src_dict.items():
            if 'loss' in key:
                f.write('last %s:%s\n' % (key, value[-1]))
            elif key != "net_dict":
                f.write('%s:%s\n' % (key, value))


def calc_psnr(model, dataloader, device):
    model.eval()
    mse = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            x_noised = noiser(x).to(device)
            out = model(x_noised)
            mse.append(torch.mean(((out - x)/2)**2, dim=(1, 2, 3)))
    avg_mse = torch.mean(torch.hstack(mse))
    avg_psnr = 20*np.log10(1/avg_mse.cpu())
    return avg_psnr


def create_net_diagram(model, dataloader, dest_path):
    writer = SummaryWriter(dest_path + 'diagram')
    x, y = next(iter(dataloader))
    writer.add_graph(model, x.to(device))
    writer.close()


def plot_monitored_images(monitored_imgs, noised_monitored_imgs, model, dest_path, epoch):
    writer = SummaryWriter(dest_path + 'images')
    model.eval()
    with torch.no_grad():
        out = model(noised_monitored_imgs)
    all_images_modes = torch.cat([torch.stack([monitored_imgs[idx], noised_monitored_imgs[idx], out[idx]]) for idx in range(4)], dim=0)
    grid = torchvision.utils.make_grid((all_images_modes + 1) / 2, nrow=3)
    fig = plt.figure(figsize=(10, 12))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.tight_layout()
    writer.add_figure('Denoising Process vs. Epochs', fig, global_step=epoch)


# ==================================================================================================================== #
if __name__ == '__main__':
    ######################################
    # Choose device
    ######################################
    device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')
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
        epochs = epochs if train_model else output_dict['epochs']
        batch_size = output_dict['batch_size']
        scaling_depth = output_dict['scaling_depth']

        model = mobilenetv3(input_size=output_dict['input_size'], in_channels=3, scaling_depth=scaling_depth,
                            weights_dict=output_dict['net_dict'], device=device).to(device)
    else:
        model = mobilenetv3(input_size=input_size, in_channels=3, scaling_depth=scaling_depth, device=device).to(device)

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

    noiser = AddGaussianNoise(std=std, device=device)
    ######################################
    # Train a model from scratch
    ######################################
    if train_model:
        writer = SummaryWriter(dest_path + 'images')
        monitored_imgs, _ = next(iter(val_loader))
        monitored_imgs = monitored_imgs.to(device)
        noised_monitored_imgs = noiser(monitored_imgs)
        train_loss = []
        val_loss = []
        if pretrained_run_number is not None:
            train_loss = output_dict['train_loss']
            val_loss = output_dict['val_loss']

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

            plot_monitored_images(monitored_imgs, noised_monitored_imgs, model, dest_path, epoch)
            if val_loss[-1] < min_val_loss:
                min_val_loss = val_loss[-1]
                best_model_dict = model.state_dict()

            end_time = time.time()
            logging.info('[Epoch {}/{}] -> Train Loss: {:.4f}, Validation Loss: {:.4f}, Time: {:.1f}'.format(
                epoch + 1, epochs, train_loss[-1], val_loss[-1], end_time - start_time))

            print('[Epoch {}/{}] -> Train Loss: {:.4f}, Validation Loss: {:.4f}, Time: {:.1f}'.format(
                epoch + 1, epochs, train_loss[-1], val_loss[-1], end_time - start_time))

        print('Training Finished! saved in run: ', curr_run)
        logging.info('Training Finished! saved in run: ' + str(curr_run))

        # num of params:
        n_params = count_parameters(model)
        psnr = calc_psnr(model, val_loader, device)

        # Save net dicts
        output_dict = {
            "data_type": data_type,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "scaling_depth": scaling_depth,
            "std": std,
            "input_size": input_size,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "n_params": n_params,
            "per_image_psnr": psnr,
            "net_dict": best_model_dict
        }

        with open(dest_path + 'output_dict.pickle', 'wb') as handle:
            pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        write_dict_to_file(output_dict, dest_path + 'run_info.txt')

        # Plot training losses:
        plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train')
        plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation')
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
    x = x[:16]
    noiser = AddGaussianNoise(std=std, device=torch.device('cpu'))
    x_noised = noiser(x)
    out = model(x_noised.to(device)).cpu()
    all_images_modes = torch.cat([torch.stack([x[idx], x_noised[idx], out[idx]]) for idx in range(len(x))], dim=0)
    grid = torchvision.utils.make_grid((all_images_modes + 1)/2, nrow=9)
    plt.figure(figsize=(30, 30))
    plt.imshow(grid.permute(1, 2, 0))
    plt.tight_layout()
    # plt.show()
    plt.savefig(dest_path + 'output_grid.png')


    ######################################
    # Plot Time Stats
    ######################################
    noiser = AddGaussianNoise(std=std, device=device)
    inference_avg_time = []
    inference_ci_time = []
    test_batches = [1, 16, 32, 64, 128, 256]
    confidence_level = scipy.stats.norm.ppf(0.05/2)  # for ci of 0.95
    val_sampler = torch.utils.data.SubsetRandomSampler(val_inds)
    for batch in test_batches:
        tmp_val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=val_sampler, drop_last=True)
        image_avg_time, batch_std = test_time(tmp_val_loader, model, noiser, device)
        inference_avg_time.append(image_avg_time)
        inference_ci_time.append(confidence_level*batch_std/np.sqrt(len(tmp_val_loader)))

    inference_avg_time = np.array(inference_avg_time)
    inference_ci_time = np.array(inference_ci_time)
    plt.figure()
    plt.bar(test_batches, inference_avg_time*1e3, yerr=inference_ci_time*1e3, align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.title('Inference Time vs. Batch Size (' + str(device) + ')')
    plt.xlabel('batch size')
    plt.ylabel('time per image [ms]')
    plt.grid()
    plt.savefig(dest_path + 'time_stats_' + str(device))

    # create model diagram
    create_net_diagram(model, val_loader, dest_path)


