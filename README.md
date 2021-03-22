# SMBD

SMBD (Saliency Map Bias Detection) is a tool based on [PyTorch](https://pytorch.org/), [Captum](captum.ai/) and [Scikit-learn](https://scikit-learn.org/stable/) which is designed to detect particular biases transferred to a deep neural network through training on biased datasets. SMBD implements and extends the SpRAy methodology introduced and developed by [Lapuschkin et al. [1]](https://www.nature.com/articles/s41467-019-08987-4#auth-Sebastian-Lapuschkin).

## Methodology

SpRAy computes explanations for images in the test set of a certain class, after postprocessing these explanations, SpRAy clusters them and then the presence of more than one cluster may prompt the presence of particular biases in the training set. SMBD executes the following tasks in order:
* Dataset preprocessing, possibly adding artificial bias
* Network training
* Automatic generation of explanations on the test set for inputs belonging to a given class
* Explanations post-processing (rescale, PCA)
* Explanation clustering
* Results visualization through t-SNE


## Example Code

SMBD offers a concise yet modular API to perform bias analysis. As an example, the following code shows how to execute the entire pipeline necessary for bias detection on CIFAR10 (specified by the attribute `dataset='cifar10'`) artificially biased by adding a black square (`bias_method='box'`) on the top-left of 30% of the images in the dataset (`bias_probability=.3`) belonging to the seventh class (`biased_class=7`). SMBD trains an AlexNet model (`model='alexnet'`) on the new biased dataset and then computes saliency maps (`explainer='saliency'`) of this model for all the test images belonging to the biased class.

Then SMBD rescales saliency maps by a factor 2 (`rescale=2`) using 2x2 kernel Max Pooling and executes PCA (`pca=True`). Finally, clusters the reduced heatmaps using the OPTICS algorithm (`clustering_method=optics`).

Once the pipeline is initialized, it can be run by executing `pip.run()` and then results can be visualized using the `pip.tsne()` command.

```
import pipeline
from sklearn.cluster import OPTICS

min_samples = 20
xi = .05
min_cluster_size = .05
optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)

pip = Pipeline(dataset='cifar10',
	       model='alexnet',
	       bias_method='box',
	       device=torch.device("cuda:1"),
	       explainer='saliency',
	       rescale=2,
	       pca=True,
	       biased_class=7,
	       clustering_method=optics,
	       bias_probability=.3)
pip.run()
pip.tsne()
```

## Bibliography

[1] Lapuschkin, S., Wäldchen, S., Binder, A. _et al._ Unmasking Clever Hans predictors and assessing what machines really learn. _Nat Commun_**10,** 1096 (2019). https://doi.org/10.1038/s41467-019-08987-4
