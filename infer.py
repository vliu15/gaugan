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
import yaml

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from modules.dataset import CityscapesDataset
from utils import show_tensor_images


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.yml')
    parser.add_argument('-e', '--encode', action='store_true', default=False)
    parser.add_argument('-n', '--n_show', type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_arguments()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        config = OmegaConf.create(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    gaugan = instantiate(config.gaugan).to(device)
    gaugan.load_state_dict(torch.load(config.resume_checkpoint)['gaugan_model_dict'])

    test_dataloader = torch.utils.data.DataLoader(
        instantiate(config.test_dataset),
        collate_fn=CityscapesDataset.collate_fn,
        **config.test_dataloader,
    )

    n = 0
    for (x, l) in test_dataloader:
        if n == args.n_show:
            break

        x = x.to(device)
        l = l.to(device)

        if not args.encode:
            z = gaugan.sample_z(mu=None, logvar=None, n_samples=x.size(0))
        else:
            mu, logvar = gaugan.encode(x)
            z = gaugan.sample_z(mu=mu, logvar=logvar)
        
        x_fake = gaugan.generate(z, l)
        show_tensor_images(x_fake.to(x.dtype))
        show_tensor_images(x)

        n+= 1


if __name__ == '__main__':
    main()
