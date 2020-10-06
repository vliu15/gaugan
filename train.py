# The MIT License
#
# Copyright (c) 2020 Vincent Liu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import argparse
from datetime import datetime

import os
import yaml
import torch
from tqdm import tqdm
from hydra.utils import instantiate
from omegaconf import OmegaConf

from modules.dataset import CityscapesDataset
from modules.loss import GauGANLoss
from utils import get_lr_lambda, weights_init


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.yml')
    return parser.parse_args()


def train(dataloaders, gaugan, optimizers, schedulers, train_config, start_epoch, device):
    ''' Training loop for Pix2PixHD '''
    # unpack all modules
    train_dataloader, val_dataloader = dataloaders
    g_optimizer, d_optimizer = optimizers
    g_scheduler, d_scheduler = schedulers

    # initialize logging
    loss = GauGANLoss(device=device)
    log_dir = os.path.join(train_config.log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(log_dir, mode=0o775, exist_ok=False)

    for epoch in range(start_epoch, train_config.epochs):

        # training epoch
        mean_g_loss = 0.0
        mean_d_loss = 0.0
        epoch_steps = 0
        gaugan.train()
        pbar = tqdm(train_dataloader, position=0, desc='train [G loss: -.----][D loss: -.----]')
        for (x_real, labels) in pbar:
            x_real = x_real.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                g_loss, d_loss, x_fake = loss(
                    x_real, labels, gaugan,
                )

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            mean_g_loss += g_loss.item()
            mean_d_loss += d_loss.item()
            epoch_steps += 1

            pbar.set_description(desc=f'train [G loss: {mean_g_loss/epoch_steps:.4f}][D loss: {mean_d_loss/epoch_steps:.4f}]')

        if epoch+1 % train_config.save_every == 0:
            torch.save({
                'gaugan_model_dict': gaugan.state_dict(),
                'g_optim_dict': g_optimizer.state_dict(),
                'd_optim_dict': d_optimizer.state_dict(),
                'epoch': epoch,
            }, os.path.join(log_dir, f'epoch={epoch}.pt'))

        g_scheduler.step()
        d_scheduler.step()

        # validation epoch
        mean_g_loss = 0.0
        mean_d_loss = 0.0
        epoch_steps = 0
        gaugan.eval()
        pbar = tqdm(val_dataloader, position=0, desc='val [G loss: -.----][D loss: -.----]')
        for (x_real, labels) in pbar:
            x_real = x_real.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                    g_loss, d_loss, x_fake = loss(
                        x_real, labels, insts, bounds, encoder, generator, discriminator,
                    )

            mean_g_loss += g_loss.item()
            mean_d_loss += d_loss.item()
            epoch_steps += 1

            pbar.set_description(desc=f'val [G loss: {mean_g_loss/epoch_steps:.4f}][D loss: {mean_d_loss/epoch_steps:.4f}]')


def main():
    args = parse_arguments()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        config = OmegaConf.create(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    gaugan = instantiate(config.gaugan).apply(weights_init)

    g_optimizer = torch.optim.Adam(
        list(gaugan.generator.parameters()) + list(gaugan.encoder.parameters()), **config.g_optim,
    )
    d_optimizer = torch.optim.Adam(list(gaugan.discriminator.parameters()), **config.d_optim)
    g_scheduler = torch.optim.lr_scheduler.LambdaLR(
        g_optimizer,
        get_lr_lambda(config.train.epochs, config.train.decay_after),
    )
    d_scheduler = torch.optim.lr_scheduler.LambdaLR(
        d_optimizer,
        get_lr_lambda(config.train.epochs, config.train.decay_after),
    )

    torch.save({
        'gaugan_model_dict': gaugan.state_dict(),
        'g_optim_dict': g_optimizer.state_dict(),
        'd_optim_dict': d_optimizer.state_dict(),
        'epoch': 0,
    }, os.path.join(config.train.log_dir, f'epoch={0}.pt'))
    return

    start_epoch = 0
    if config.resume_checkpoint is not None:
        state_dict = torch.load(config.resume_checkpoint)

        gaugan.load_state_dict(state_dict['gaugan_model_dict'])
        g_optimizer.load_state_dict(state_dict['g_optim_dict'])
        d_optimizer.load_state_dict(state_dict['d_optim_dict'])
        start_epoch = state_dict['epoch']
        print(f'Starting GauGAN training from checkpoints')

    else:
        print('Starting GauGAN training from random initialization')

    train_dataloader = torch.utils.data.DataLoader(
        instantiate(config.train_dataset),
        collate_fn=CityscapesDataset.collate_fn,
        **config.train_dataloader,
    )
    val_dataloader = torch.utils.data.DataLoader(
        instantiate(config.val_dataset),
        collate_fn=CityscapesDataset.collate_fn,
        **config.val_dataloader,
    )

    train(
        [train_dataloader, val_dataloader],
        gaugan,
        [g_optimizer, d_optimizer],
        [g_scheduler, d_scheduler],
        config.train, start_epoch, device,
    )


if __name__ == '__main__':
    main()