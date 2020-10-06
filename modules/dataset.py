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

import os
from collections.abc import Iterable

import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


class CityscapesDataset(torch.utils.data.Dataset):
    ''' Implements dataset with preprocessing for Cityscapes dataset '''

    def __init__(self, paths, img_size=(256, 512), n_classes=35):
        super().__init__()

        self.n_classes = n_classes

        # collect list of examples
        self.examples = {}
        if type(paths) == str:
            self.load_examples_from_dir(paths)
        elif type(paths) == list:
            for path in paths:
                self.load_examples_from_dir(path)
        else:
            raise ValueError('`paths` should be a single path or list of paths')

        self.examples = list(self.examples.values())
        assert all(len(example) == 2 for example in self.examples)

        # initialize transforms for the real color image
        self.img_transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.Lambda(lambda img: np.array(img)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # initialize transforms for semantic label maps
        self.map_transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.Lambda(lambda img: np.array(img)),
            transforms.ToTensor(),
        ])

    def load_examples_from_dir(self, abs_path):
        '''
        Given a folder of examples, this function returns a list of paired examples.
        '''
        assert os.path.isdir(abs_path)

        img_suffix = '_leftImg8bit.png'
        label_suffix = '_gtFine_labelIds.png'

        for root, _, files in os.walk(abs_path):
            for f in files:
                if f.endswith(img_suffix):
                    prefix = f[:-len(img_suffix)]
                    attr = 'orig_img'
                elif f.endswith(label_suffix):
                    prefix = f[:-len(label_suffix)]
                    attr = 'label_map'
                else:
                    continue

                if prefix not in self.examples.keys():
                    self.examples[prefix] = {}
                self.examples[prefix][attr] = root + '/' + f

    def __getitem__(self, idx):
        example = self.examples[idx]

        # load image and maps
        img = Image.open(example['orig_img']).convert('RGB') # color image: (3, h, w)
        label = Image.open(example['label_map'])             # semantic label map: (1, h, w)

        # apply corresponding transforms
        img = self.img_transforms(img)
        label = self.map_transforms(label).long() * 255

        # convert labels to one-hot vectors
        label = F.one_hot(label, num_classes=self.n_classes)
        label = label.squeeze(0).permute(2, 0, 1).to(img.dtype)

        return (img, label)

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def collate_fn(batch):
        imgs, labels = [], []
        for (x, l) in batch:
            imgs.append(x)
            labels.append(l)
        return torch.stack(imgs, dim=0), torch.stack(labels, dim=0)
