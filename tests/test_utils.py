import torch

from smbd.utils import from_grad_to_heatmap, rescale_imgs


def test_rescale_imgs():
    input = torch.ones(3, 4, 4)
    output_size = torch.Size([3, 2, 2])
    assert rescale_imgs(input, 2).size() == output_size


def test_from_grad_to_heatmap():
    input = torch.ones(4, 4, 3)
    output = torch.ones(4, 4)
    assert torch.all(((from_grad_to_heatmap(input) == output).bool()))
