from captum.attr._utils.attribution import Attribution
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import sys
import json

from typing import Tuple, Callable, Optional, cast, List
from captum.attr import Saliency, GuidedGradCam, Occlusion, Lime, GradientShap, LayerGradCam, IntegratedGradients, Deconvolution, DeepLift, GuidedBackprop


def rescale_imgs(imgs: torch.Tensor, rescale: int) -> List[torch.Tensor]:
    return torch.nn.MaxPool2d(kernel_size=rescale, stride=rescale)(imgs.unsqueeze(1)).squeeze(1)


def init_dataset(dataset,
                 model,
                 biased_class: Optional[int] = None,
                 bias_method: Optional[str] = None,
                 # TODO: Fix the following line, give consistency to the entire dataset pipeline
                 #  bias_probability: Optional[float] = None) -> Optional[Tuple[data.Dataset, data.Dataset, int, nn.modules.loss._Loss]]:
                 bias_probability: Optional[float] = None):
    input_size = _get_input_size(model)
    if isinstance(dataset, str):
        return _saved_datasets(dataset, input_size, biased_class, bias_method, bias_probability)
    else:
        return dataset


def init_dataloader(dataloader) -> Tuple[Callable, Callable]:

    if isinstance(dataloader, str):
        return _saved_dataloaders(dataloader)
    else:
        return dataloader


def init_model(model, output_size):

    if isinstance(model, str):
        out = _saved_models(model, output_size)
    else:
        out = model
    return cast(nn.Module, out)


def from_grad_to_heatmap(grad):
    return torch.mean(grad, dim=2)

def _normalize(grad):
    max_grad = torch.max(grad, dim=2)


def _saved_dataloaders(dataloader: str) -> Tuple[Callable, Callable]:
    """
    Takes a string indicating a predetermined dataloader

    Args:
        dataloader (str): Dataloader name

    Returns:
        data.DataLoader: Dataloader object
    """
    if dataloader == "standard":
        batch_size = 32

        def trainloader(trainset): return data.DataLoader(trainset, batch_size=batch_size,
                                                          shuffle=True, num_workers=4, pin_memory=True)

        def testloader(testset): return data.DataLoader(testset, batch_size=1,
                                                        shuffle=False, num_workers=4, pin_memory=True)

    return (trainloader, testloader)


def _get_input_size(model: str) -> Optional[int]:
    """
    Returns the model's input size

    Args:
        model (str): The name of the models between Inception model, VGG19, AlexNet, etc.

    Returns:
        int: the requested module
    """
    if model.lower() == 'alexnet':
        return 224
    if model.lower() == 'alexnet-cifar':
        return 32
    if model.lower() == 'voc-multilabel':
        return 227
    if model.lower() in ('squeezenet', 'vgg'):
        return 224
    if model.lower() == 'inception':
        return 229

    return None


def _saved_models(model: str, output_size: int) -> Optional[nn.Module]:
    """
    Takes a string indicating a pretrained model between Inception model, VGG19, AlexNet, etc.

    Args:
        model (str): The name of the models between Inception model, VGG19, AlexNet, etc.

    Returns:
        nn.Module: the requested module
    """
    if model.lower() == 'alexnet':
        return _alexnet(output_size)
    if model.lower() == 'alexnet-cifar':
        return _alexnet_cifar()
    if model.lower() == 'alexnet-cifar':
        return _alexnet_cifar()
    elif model.lower() == 'alexnet':
        return _alexnet(output_size)
    elif model.lower() == 'squeezenet':
        return _squeezenet(output_size)
    elif model.lower() == 'inception':
        return _inception(output_size)
    elif model.lower() == 'vgg':
        return _vgg(output_size)

    return None


def _saved_datasets(dataset: str,
                    input_size: Optional[int],
                    biased_class: Optional[int] = None,
                    bias_method: Optional[str] = None,
                    bias_probability: Optional[float] = None) -> Optional[Tuple[data.Dataset, data.Dataset, int, nn.modules.loss._Loss]]:
    """
    Takes a string indicating a dataset PASCALVOC, MNIST, CIFAR10, ImageNet. Adds the bias expressed on the specified class

    Args:
        dataset (str): [description]
        biased_class: class on which the bias is added
        bias_method: string for the bias type used.

    Returns:
        data.Dataset: [description]
    """
    if input_size is None:
        raise Exception("Network input size is None")
    if dataset.lower() == "cifar10":
        return _cifar10(input_size, biased_class=biased_class, bias_method=bias_method,
                        bias_probability=bias_probability) + (10, nn.CrossEntropyLoss())
    elif dataset.lower() == "voc":
        return _voc_pascal(input_size, biased_class=biased_class, bias_method=bias_method,
                           bias_probability=bias_probability) + (20, nn.BCEWithLogitsLoss(reduction='sum'))
    elif dataset.lower() == "imagenet":
        return _imagenet(input_size, biased_class=biased_class, bias_method=bias_method,
                         bias_probability=bias_probability) + (1000, nn.CrossEntropyLoss())
    return None


def _from_data_to_win(data: torch.Tensor):
    _, C, H, W = data.size()
    return C, H // 3, W // 3


def _from_data_to_strides(data: torch.Tensor):
    _, _, H, W = data.size()
    return (H + W) // 12


def init_explainer(explainer: str, model: nn.Module) -> Attribution:
    GRAD_CAM_LAYER = 4

    if explainer.lower() == "saliency":
        return Saliency(model)
    elif explainer.lower() == "lrp":
        sys.exit("There is no stable release of LRP for PyTorch. It will be integrated as soon as possible!")
    elif explainer.lower() == "occlusion":
        return Occlusion(model)
    elif explainer.lower() == "lime":
        return Lime(model)
    elif explainer.lower() == "gradshap":
        return GradientShap(model)
    elif explainer.lower() == "layergradcam":
        if isinstance(model, torch.Tensor):
            return LayerGradCam(model, list(model.features)[GRAD_CAM_LAYER])
        else:
            return LayerGradCam(model, model)
    elif explainer.lower() == "guidedgradcam":
        if isinstance(model, torch.Tensor):
            return GuidedGradCam(model, list(model.features)[GRAD_CAM_LAYER])
        else:
            return GuidedGradCam(model, model)
    elif explainer.lower() == "intgrad":
        return IntegratedGradients(model)
    elif explainer.lower() == "deeplift":
        return DeepLift(model)
    elif explainer.lower() == "deconv":
        return Deconvolution(model)
    elif explainer.lower() == "guidedbackprop":
        return GuidedBackprop(model)
    return None


def get_attribution(explainer: Attribution) -> Optional[Callable]:
    GRADIENT_SHAP_N_SAMPLES = 20
    if (isinstance(explainer, Saliency) or
            isinstance(explainer, IntegratedGradients) or
            isinstance(explainer, DeepLift) or
            isinstance(explainer, Deconvolution) or
            isinstance(explainer, GuidedBackprop) or
            isinstance(explainer, LayerGradCam) or
            isinstance(explainer, GuidedGradCam)
            ):
        return lambda input_data, target: explainer.attribute(input_data, target=target)
    elif isinstance(explainer, Occlusion):
        return lambda input_data, target: explainer.attribute(
            input_data, target=target, sliding_window_shapes=_from_data_to_win(input_data),
            strides=_from_data_to_strides(input_data))
    elif isinstance(explainer, Lime):
        return lambda input_data, target: explainer.attribute(
            input_data, target=target, feature_mask=_create_mask(input_data, (3, 3)))
    elif isinstance(explainer, GradientShap):
        return lambda input_data, target: explainer.attribute(
            input_data, torch.randn(
                [GRADIENT_SHAP_N_SAMPLES, *input_data.size()[1:]],
                device=input_data.get_device()),
            target=target)
    return None


def _create_mask(x: torch.Tensor, kernel: Tuple[int, int]) -> torch.Tensor:
    from itertools import product
    y = torch.zeros_like(x, dtype=torch.long)
    for index, (i, j) in enumerate(product(range(0, x.size()[-2], kernel[0]), range(0, x.size()[-1], kernel[1]))):
        y[:, :, i:i+kernel[0], j:j+kernel[1]] = index * torch.ones_like(y[:, :, i:i+kernel[0], j:j+kernel[1]])
    return y


def _voc_pascal(input_size: int,
                biased_class: Optional[int] = None,
                bias_method: Optional[str] = None,
                bias_probability: Optional[float] = None):
    from smbd.datasets.biased import BiasedPascalVOC_Dataset

    VOC_PATH = 7*'../'+'nas/data/nesti_pacini/datasets/VOC'

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size))
    ])

    idx2label = _human_readable_voc_labels()

    def transform_targets(x):
        list_obj = x['annotation']['object']
        names_obj = [ob['name'] for ob in list_obj]
        target = np.zeros(len(idx2label))
        for o in names_obj:
            target[idx2label.index(o.lower())] = 1

        return target

    transform_target = transforms.Lambda(transform_targets)

    trainset = BiasedPascalVOC_Dataset(root=VOC_PATH, download=False, image_set='trainval',
                                       transform=transform, target_transform=transform_target,
                                       biased_class=biased_class, bias_method=bias_method,
                                       bias_probability=bias_probability)
    testset = BiasedPascalVOC_Dataset(root=VOC_PATH, download=False, image_set='val',
                                      transform=transform, target_transform=transform_target,
                                      biased_class=biased_class, bias_method=bias_method,
                                      bias_probability=bias_probability)
    return trainset, testset


def _imagenet(input_size: int,
              biased_class: Optional[int] = None,
              bias_method: Optional[str] = None,
              bias_probability: Optional[float] = None):
    from smbd.datasets.biased import BiasedImageNet
    imagenet_path = '../../../../../../nas/data/nesti_pacini/datasets/imagenet/CLS-LOC'

    transform = transforms.Compose([transforms.RandomResizedCrop(input_size),
                                    transforms.RandomHorizontalFlip()])
    trainset = BiasedImageNet(root=imagenet_path + '/train',
                              transform=transform,
                              biased_class=biased_class,
                              bias_method=bias_method,
                              bias_probability=bias_probability)

    testset = BiasedImageNet(root=imagenet_path + '/val',
                             transform=transform,
                             biased_class=biased_class,
                             bias_method=bias_method,
                             bias_probability=bias_probability)
    return trainset, testset


def _cifar10(input_size: int,
             biased_class: Optional[int] = None,
             bias_method: Optional['str'] = None,
             bias_probability: Optional[float] = None):
    from smbd.datasets.biased import BiasedCIFAR10
    transform = transforms.Compose([transforms.Resize(input_size)])
    trainset = BiasedCIFAR10(root='/nas/data/nesti_pacini/datasets/cifar',
                             train=True,
                             download=False,
                             transform=transform,
                             biased_class=biased_class,
                             bias_method=bias_method,
                             bias_probability=bias_probability)

    testset = BiasedCIFAR10(root='/nas/data/nesti_pacini/datasets/cifar',
                            train=False,
                            download=False,
                            transform=transform,
                            biased_class=biased_class,
                            bias_method=bias_method,
                            bias_probability=bias_probability)
    return trainset, testset


def _inception(output_size):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True, aux_logits=False)
    # Change last layer
    return model


def _vgg(output_size):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
    model.classifier[6] = nn.Linear(4096, output_size)
    lr = [1e-5, 5e-3]
    # lr = [1e-4, 1e-4]
    optimizer = torch.optim.SGD([
        {'params': list(model.parameters())[:-1], 'lr': lr[0], 'momentum': 0.9},
        {'params': list(model.parameters())[-1], 'lr': lr[1], 'momentum': 0.9}
    ])
    last_conv_layer = model.features[-2]
    return model, optimizer, last_conv_layer


def _alexnet_cifar():
    from smbd.alexnet import AlexNet
    model = AlexNet()
    return model, torch.optim.Adam(model.parameters(), lr=0.001), model.features[-2]


def _alexnet(output_size):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    model.classifier[6] = nn.Linear(4096, output_size)
    lr = [1e-5, 5e-3]
    optimizer = torch.optim.SGD([
        {'params': list(model.parameters())[:-1], 'lr': lr[0], 'momentum': 0.9},
        {'params': list(model.parameters())[-1], 'lr': lr[1], 'momentum': 0.9}
    ])
    last_conv_layer = model.features[-2]
    return model, optimizer, last_conv_layer


def bias_method2str(bias_method: Optional[str], biased_class: Optional[int], bias_probability: Optional[float]) -> str:
    bias_method_ = bias_method if bias_method is not None else ""
    biased_class_ = str(biased_class) if biased_class is not None else ""
    bias_probability_ = str(int(bias_probability * 100)) if bias_probability is not None else ""
    return bias_method_ + biased_class_ + "BiasProb" + bias_probability_





def _squeezenet(output_size):
    return torch.hub.load('pytorch/vision:v0.6.0', 'squeezenet1_0', pretrained=True)


def _compute_hms(tot_time):
    """
    Computes hours, minutes, seconds from total time in seconds.
    """
    hrs = tot_time // 3600
    mins = (tot_time - hrs * 3600) // 60
    secs = (tot_time - hrs * 3600 - mins * 60)
    return hrs, mins, secs


def save_heatmaps(grads, targets, root_path, folder_name):
    """
    Function that saves the heatmap dataset in a format that is parsable by the HeatmapDataset class.
    Args in:
            - grads: heatmaps to be saved
            - targets: label associated with each heatmap
            - path: root path of the HeatmapDataset
    """
    grads = list(grads.cpu().detach().numpy())

    # Deactivate interactive plotting
    plt.ioff()

    print("Heatmaps saving started. Computing approx ETA...")
    w, h = grads[0].shape
    t = time.time()
    target_strings = []

    # Formatting to print always same number of digits in filename
    digits_num = int(np.log10(len(grads))) + 1
    master_string = "%%0%dd" % digits_num

    for i, grad in enumerate(grads):

        # To save only image content
        fig = plt.figure(frameon=False)
        fig.set_size_inches(1, 1)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(grad, aspect='auto')

        path = os.path.join(root_path, folder_name)
        if folder_name not in os.listdir(root_path):
            os.mkdir(path, 0o777)
        if root_path is not None:
            # Creates required folders and files for HeatmapDataset
            if "Images" not in os.listdir(path):
                os.mkdir(os.path.join(path, "Images"), 0o777)
            if "Annotations" not in os.listdir(path):
                os.mkdir(os.path.join(path, "Annotations"), 0o777)
            htm_name = "HTMP-" % targets[i] + master_string % i + ".jpg"
            file_name = os.path.join("Images", htm_name)

            plt.savefig(os.path.join(path, file_name), dpi=w)
            plt.close()

            target_strings.append("%s %d \n" % (htm_name, targets[i]))

            # Computing estimated ETA and running time
            if i == 100:
                avg_time = (time.time() - t)/100
                tot_time = avg_time * (len(grads) - 100)
                hrs, mins, secs = _compute_hms(tot_time)
                print("Expected ETA is %02dh:%02dm:%02ds" % (hrs, mins, secs))
            if (i+1) % 1000 == 0:
                hrs, mins, secs = _compute_hms(time.time() - t)
                print("Saving heatmap %d/%d, elapsed time: %02dh:%02dm:%02ds" % (i+1, len(grads), hrs, mins, secs))

    # Reactivate interactive plotting
    plt.ion()
    # Write target file
    with open(os.path.join(path, "Annotations", "annotation.txt"), "w") as f:
        f.writelines(target_strings)

    os.system("chmod -R o+wr " + path)

# From imagenet_utils


def _human_readable_voc_labels():
    """
    Create human readable VOC PASCAL lables.
    idx2label[i] returns text label associated to class i.
    """
    with open('../smbd/data/pascal_voc_class_index.json', 'rb') as f:
        class_idx = json.load(f)
    idx2label = [class_idx[str(k)] for k in range(len(class_idx))]
    return sorted(idx2label)

# From imagenet_utils


def human_readable_imagenet_labels():
    """
    Create human readable imagenet lables. 
    idx2label[i] returns text label associated to class i.
    """
    with open('./smbd/data/imagenet_class_index.json', 'rb') as f:
        class_idx = json.load(f)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    return idx2label

# From imagenet_utils


def print_top_k(net_output, idx2label, k=5):
    """
    Print human-readable results of the network prediction.
    """
    sorted_inds = [i[0] for i in sorted(enumerate(-net_output), key=lambda x: x[1])]
    sorted_outs = net_output[sorted_inds]
    for i in range(k):
        index = sorted_inds[i]
        print('Probability %0.2f%% => [%s]' % (100 * net_output[index], idx2label[index]))
    return (sorted_inds, sorted_outs)

# From imagenet_utils


def show_im(preprocessed_image, tensor=True, unnormalize=True, heatmap=False):
    """
    Show Imagenet preprocessed image. Preprocessing is basically scaling + cropping but varies with network.
    """
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STDDEV = [0.229, 0.224, 0.225]
    if tensor:
        preprocessed_image = np.transpose(preprocessed_image[0].numpy(), (1, 2, 0))
    if unnormalize:
        preprocessed_image *= IMAGENET_STDDEV
        preprocessed_image += IMAGENET_MEAN
    if heatmap:
        plt.imshow(preprocessed_image, cmap='hot')
    else:
        plt.imshow(preprocessed_image)
    plt.show()

# From imagenet_utils


def show_hm(heatmap, vmin=0, vmax=255, tensor=True, unnormalize=True):
    """
    Show Imagenet preprocessed image. Preprocessing is basically scaling + cropping but varies with network.
    """
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STDDEV = [0.229, 0.224, 0.225]
    if tensor:
        heatmap = np.transpose(heatmap[0].numpy(), (1, 2, 0))
    if unnormalize:
        heatmap *= IMAGENET_STDDEV
        heatmap += IMAGENET_MEAN

    plt.imshow(heatmap, vmin=vmin, vmax=vmax, cmap='hot')
    plt.show()


def cluster_accuracy(biased, cluster) -> float:
    from sklearn import metrics
    return metrics.adjusted_rand_score(biased, cluster)


def print_acc_dict(tot: dict, acc: dict):
    for key in sorted(tot.keys()):
        print('Accuracy on class {key} is {acc}'.format(key=key, acc=acc[key] / tot[key]))
