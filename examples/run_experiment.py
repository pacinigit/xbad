"""
# This is a script for the experiments. 
# Please save each experiment by commenting the correspondent variables.
# Don't forget to pass the gpu as argument! e.g.,  python3 run_experiment.py 0
"""

from sklearn.cluster import OPTICS, cluster_optics_dbscan
from time import strftime, gmtime
import time
import sys

import torch

# import torchvision.transforms as transforms
import torch.nn as nn

from smbd import SMBD

if __name__ == '__main__':

    if len(sys.argv) != 2:
        sys.exit("Wrong number of arguments: usage: 'python run_experiments.py n_gpu'")
    n_gpu = int(sys.argv[1])
    if n_gpu < 0 or n_gpu > 3:
        sys.exit("Inexistent GPU. Select GPU between 0 and 3.")

    print("================================")
    print("Running experiment on GPU %d" % n_gpu)
    print("================================")
    device = torch.device("cuda:%d" % n_gpu)

    min_samples = 20
    xi = .05
    min_cluster_size = .05

    # clust = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)

    start_time = time.time()

    """ 
    EXPERIMENT 0 (example: how to store variables.)
    dataset = 'cifar10'
    model = 'alexnet'
    bias_method = None
    biased_class = 6
    criterion = nn.CrossEntropyLoss()
    explainer = 'saliency'
    rescale = 2
    pca = True
    clustering_method = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
    """

    """
    EXPERIMENT 1
    """
    dataset = 'cifar10'
    model = 'alexnet'
    bias_method = 'box'
    biased_class = 6
    criterion = nn.CrossEntropyLoss()
    explainer = 'saliency'
    rescale = 2
    pca = True
    clustering_method = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)

    # Running experiment with chosen setup.
    pip = SMBD(dataset=dataset,
               #                dataloader='standard',
               model=model,
               bias_method=bias_method,
               biased_class=biased_class,
               criterion=criterion,
               device=device,
               explainer=explainer,
               rescale=rescale,
               pca=pca,
               clustering_method=clustering_method)

    _ = pip.run()

    print("Process ended ",  strftime("%Hh:%Mm:%Ss", gmtime(time.time()-start_time)))
