import warnings
import torch

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torchvision.datasets.voc as voc

from typing import Any, Callable, Optional, Tuple
from PIL import Image
import numpy as np
import os
import random
import string

import xml.etree.ElementTree as ET


def _gen_rnd_str():
    N = random.randint(5, 12)
    return ''.join(random.choices(string.ascii_letters + ' '*6, k=N))


def _add_lbl(img: Image, position: Tuple[int, int]) -> Image:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)

    draw.text(position, _gen_rnd_str())
    return img


def _add_box(image: Image, position: Tuple[int, int], size: Tuple[int, int]) -> Image:
    rect = Image.new("RGB", size, (0, 0, 0))
    image.paste(rect, position)
    return image


def _transform(image: Image, bias_method: str) -> Image:
    W, H = image.size
    if bias_method.lower() == 'pad':
        trans = transforms.Compose([
            transforms.CenterCrop(W-10),
            transforms.Pad(5)
        ])
        return trans(image)
    elif bias_method.lower() == 'id':
        return image
    elif bias_method.lower() == 'randombox':
        rect_size = (H // 10, W // 4)
        rect_max_pos = (H - rect_size[0], W - rect_size[1])
        rect_pos = (random.randint(0, rect_max_pos[0]), random.randint(0, rect_max_pos[1]))
        return _add_box(image, rect_pos, rect_size)
    elif bias_method.lower() == 'box':
        rect_size = (H // 10, W // 4)
        rect_pos = (0, 0)
        return _add_box(image, rect_pos, rect_size)
    elif bias_method.lower() == 'label':
        return _add_lbl(image, position=(W // 10, H // 10))
    elif bias_method.lower() == 'randomlabel':
        position = (random.randint(0, W), random.randint(0, H))
        return _add_lbl(image, position=position)
    return image


def _add_bias(image: Image, bias_method: Optional[str]) -> Tuple[Any, bool]:
    if bias_method is None:
        return image, False
    else:
        return _transform(image, bias_method), True


def _init_prob(bias_probability: Optional[float] = None) -> float:
    if bias_probability is None:
        return 1
    elif not 0 <= bias_probability <= 1:
        warnings.warn("Bias_probability value not in the [0, 1] range. Setting it to 1 by default.")
        return 1
    return bias_probability


class BiasedDataset(torchvision.datasets.VisionDataset):
    def __init__(self,
                 root: str,
                 biased_class: int = None,
                 bias_method: Optional[str] = None,
                 bias_probability: Optional[float] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(root=root,
                         transform=transform,
                         target_transform=target_transform)
        self.biased_class = biased_class
        self.bias_method = bias_method
        self.bias_probability = _init_prob(bias_probability)

        # Loading biased item list. Required for complete repeatibility
        int_prob = int(self.bias_probability * 100)
        filename = 'bias_list_prob_%d.txt' % int_prob
        if filename not in os.listdir(root):
            # Create new file for that probability.
            with open(os.path.join(root, filename), 'w') as f:
                for i in range(self.__len__()):
                    if np.random.rand() < self.bias_probability:
                        f.write('%d\n' % i)
        with open(os.path.join(root, filename), 'r') as f:
            bias_list = f.readlines()

        self.bias_list = list(map(int, bias_list))


class BiasedCIFAR10(BiasedDataset, torchvision.datasets.CIFAR10):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 biased_class: int = None,
                 bias_method: Optional[str] = None,
                 bias_probability: Optional[float] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False):
        BiasedDataset.__init__(self,
                               root=root,
                               transform=transform,
                               biased_class=biased_class,
                               bias_method=bias_method,
                               bias_probability=bias_probability,
                               target_transform=target_transform)
        torchvision.datasets.CIFAR10.__init__(self,
                                              root=root,
                                              train=train,
                                              transform=transform,
                                              target_transform=target_transform,
                                              download=download)

    def __getitem__(self, index: int) -> Tuple[Any, Any, bool]:
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        biased = False
        if self.biased_class is not None and self.biased_class == target and index in self.bias_list:
            img, biased = _add_bias(img, self.bias_method)

        CIFAR10_MEAN, CIFAR10_STDDEV = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        normalize = transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STDDEV)
        trans = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        return trans(img), target, biased


class BiasedPascalVOC_Dataset(BiasedDataset, voc.VOCDetection):
    def __init__(self,
                 root: str,
                 image_set: str,
                 biased_class: int = None,
                 bias_method: Optional[str] = None,
                 bias_probability: Optional[float] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False):
        BiasedDataset.__init__(self,
                               root=root,
                               transform=transform,
                               target_transform=target_transform,
                               biased_class=biased_class,
                               bias_method=bias_method,
                               bias_probability=bias_probability)
        voc.VOCDetection.__init__(self,
                                  root=root,
                                  image_set=image_set,
                                  download=download,
                                  transform=transform,
                                  target_transform=target_transform)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            img = self.transform(img)

        biased = False
        if self.biased_class is not None and self.biased_class in np.where(target == 1)[0] and index in self.bias_list:
            img, biased = _add_bias(img, self.bias_method)

        VOC_MEAN, VOC_STDDEV = [102.9801/255, 115.9465/255, 122.7717/255], [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=VOC_MEAN, std=VOC_STDDEV)
        trans = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        return trans(img), target, biased


class BiasedImageNet(BiasedDataset, torchvision.datasets.ImageFolder):
    # TODO: Review the entire class using BiasedDataset
    def __init__(self,
                 root: str,
                 biased_class: int = None,
                 bias_method: Optional[str] = None,
                 bias_probability: Optional[float] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        BiasedDataset.__init__(self,
                               root=root,
                               transform=transform,
                               target_transform=target_transform,
                               biased_class=biased_class,
                               bias_method=bias_method,
                               bias_probability=bias_probability)
        torchvision.datasets.ImageFolder.__init__(self,
                                                  root=root,
                                                  transform=transform,
                                                  target_transform=target_transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        biased = False
        if self.biased_class is not None and self.biased_class == target and index in self.bias_list:
            img, biased = _add_bias(img, self.bias_method)

        IMAGENET_MEAN, IMAGENET_STDDEV = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV)
        trans = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        return trans(img), target, biased
