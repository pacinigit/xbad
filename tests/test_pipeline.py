import pytest
import torch
from smbd import SMBD
from sklearn.cluster import OPTICS

test_configs = [
    dict(
        device=torch.device("cuda:%d" % 1),
        dataset="voc",
        model="alexnet",
        bias_method="box",
        biased_class=7,
        explainer="saliency",
        rescale=2,
        pca=True,
        clustering_method=OPTICS(min_samples=20, xi=0.05, min_cluster_size=0.05),
    ),
    dict(
        device=torch.device("cuda:%d" % 1),
        dataset="cifar10",
        model="alexnet-cifar",
        bias_method="box",
        biased_class=7,
        explainer="saliency",
        rescale=2,
        pca=True,
        clustering_method=OPTICS(min_samples=20, xi=0.05, min_cluster_size=0.05),
    ),
]


@pytest.mark.parametrize("config", test_configs)
def test_SMBD_first_run(config):
    # Running experiment with the chosen configurations.
    pip = SMBD(**config)

    _ = pip.run()


@pytest.mark.parametrize("config", test_configs)
def test_SMBD_second_run(config):
    # Running experiment with the chosen configurations.
    pip = SMBD(**config)

    _ = pip.run()
