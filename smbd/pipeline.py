import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import pickle
import h5py

from time import strftime, gmtime
from sklearn.manifold import TSNE
from sklearn.base import ClusterMixin
from typing import Union, Tuple, Callable, Optional
from torch.optim.optimizer import Optimizer
from captum.attr._utils.attribution import Attribution
from smbd.utils import (
    cluster_accuracy,
    init_dataset,
    init_dataloader,
    init_model,
    rescale_imgs,
    init_explainer,
    from_grad_to_heatmap,
    save_heatmaps,
    bias_method2str,
    get_attribution,
)
from smbd.snapshot import (
    add_model,
    add_heatmap,
    add_clustering,
    check_model,
    check_heatmap,
    check_clustering,
    check_pca,
    add_pca,
)

NAS_PATH = "../../../../../../nas/data/nesti_pacini/"
DATASETS_PATH = os.path.join(NAS_PATH, "datasets")
MODELS_PATH = os.path.join(NAS_PATH, "models")
HEATMAPS_PATH = os.path.join(NAS_PATH, "heatmaps")
CLUSTERS_PATH = os.path.join(NAS_PATH, "clusters")
HTML_PATH = os.path.join(NAS_PATH, "html")
PCA_DATASET = "PCA_DATASET"


class SMBD:
    """
    Define the entire pipeline for creating interpretations for an whole dataset.
    Automate the following chain of operations:

            Dataset --> Neural Network --> Explainer --> Rescale Explanations --> Clustering Method
    """

    def __init__(
        self,
        dataset,
        model: Union[str, nn.Module],
        biased_class: Optional[int] = None,
        bias_method: Optional[str] = None,
        dataloader: Union[str, Tuple[Callable, Callable]] = "standard",
        explainer: Union[str, Attribution] = None,
        clustering_method: Optional[ClusterMixin] = None,
        #  criterion: nn.modules.loss._Loss = nn.CrossEntropyLoss(),
        rescale: Optional[int] = None,
        pca: Optional[bool] = False,
        device: Union[torch.device, str] = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
        heatmaps_path: Optional[str] = None,
        bias_probability: Optional[float] = None,
    ):
        self.bias_method = bias_method
        self.heatmaps_path = heatmaps_path
        self.biased_class = biased_class
        self.bias_probability = bias_probability
        self.device = torch.device(device)
        self.rescale = 1 if rescale is None else rescale
        self.pca = pca
        self.criterion = None
        self.trainset, self.testset, output_size, self.criterion = init_dataset(
            dataset, model, self.biased_class, self.bias_method, self.bias_probability
        )
        self.dataloader = init_dataloader(dataloader)
        self.trainloader, self.testloader = (
            self.dataloader[0](self.trainset),
            self.dataloader[1](self.testset),
        )
        self.model, self.optimizer, self.conv_layer_expl = init_model(
            model, output_size
        )
        self.model = self.model.to(self.device)
        self.explainer = explainer
        self.clustering_method = clustering_method

    def run(self):
        self.train(
            optimizer=self.optimizer, device=self.device, criterion=self.criterion
        )

        if self.explainer is not None:
            self.generate_heatmap(self.explainer)

        if self.pca is not None:
            self.execute_pca(pca=self.pca)

        if self.clustering_method is not None:
            self.clusters, self.cluster_accuracy = self.cluster_heatmap(
                self.clustering_method
            )

        return self

    def generate_heatmap(self, explainer):
        if explainer is not None:
            # Note that initialization of explainer is here because of bug in the LRP module
            # cause bad computational problems and alter training behavior, model doesn't train
            self.explainer = init_explainer(explainer, self.model)
        elif self.explainer is None:
            raise RuntimeError("No explainer specified")

        if not check_heatmap(
            type(self.explainer).__name__,
            type(self.model).__name__,
            self.rescale,
            type(self.trainset).__name__,
            bias_method2str(self.bias_method, self.biased_class, self.bias_probability),
        ):

            dataiter = iter(self.testloader)

            grads = list()
            lbs = list()
            imgs = list()
            outs = list()
            biased = list()

            start_time = time.time()
            print("Heatmap Generation started")
            for i, (images, labels, biases) in enumerate(dataiter):
                out_pred = self.model(images.to(device=self.device))
                out_batch = torch.argmax(out_pred, dim=1)

                for input_data, label, if_biased, out_item in zip(
                    images, labels, biases, out_batch
                ):
                    input_data = input_data.to(device=self.device)
                    if sum(list(label.size())) > 1:
                        label = torch.argmax(label)
                    if self.biased_class is None or label == self.biased_class:
                        lbs.append(label.unsqueeze(dim=0))
                        outs.append(out_item.unsqueeze(dim=0))
                        imgs.append(input_data.unsqueeze(dim=0))
                        biased.append(if_biased.unsqueeze(dim=0))

                        input_data = input_data.unsqueeze(0)
                        input_data.requires_grad = True

                        grad = get_attribution(self.explainer)(input_data, target=label)
                        grads.append(
                            from_grad_to_heatmap(grad[0].permute(1, 2, 0)).unsqueeze(
                                dim=0
                            )
                        )

                    if i % 1000 == 0:
                        print(
                            "-- Iteration: ",
                            i,
                            ", Running Time: ",
                            strftime("%Hh:%Mm:%Ss", gmtime(time.time() - start_time)),
                            " --",
                        )

            lbs = torch.Tensor(lbs).to(device=self.device)
            grads = torch.cat(grads)
            imgs = torch.cat(imgs)
            outs = torch.Tensor(outs).to(device=self.device)
            biased = torch.cat(biased)

            if self.rescale is not None and self.rescale != 1:
                grads = rescale_imgs(grads, self.rescale)

            self.imgs, self.grads, self.outs, self.lbs, self.biased = (
                imgs,
                grads,
                outs,
                lbs,
                biased,
            )
            folder_name = (
                type(self.explainer).__name__
                + type(self.model).__name__
                + str(self.rescale)
                + type(self.trainset).__name__
                + bias_method2str(
                    self.bias_method, self.biased_class, self.bias_probability
                )
            )

            save_heatmaps(self.grads, self.outs, HEATMAPS_PATH, folder_name)
            _ = add_heatmap(
                type(self.explainer).__name__,
                type(self.model).__name__,
                self.rescale,
                type(self.trainset).__name__,
                bias_method2str(
                    self.bias_method, self.biased_class, self.bias_probability
                ),
            )
            print("Heatmap Generation Completed")
        else:
            print("LOADING HEATMAPS")
            from smbd.datasets.heatmap import HeatmapDataset

            transf_hm = transforms.Compose(
                [transforms.Grayscale(), transforms.ToTensor()]
            )
            folder_name = (
                type(self.explainer).__name__
                + type(self.model).__name__
                + str(self.rescale)
                + type(self.trainset).__name__
                + bias_method2str(
                    self.bias_method, self.biased_class, self.bias_probability
                )
            )
            path = os.path.join(HEATMAPS_PATH, folder_name)
            ht_dataset = HeatmapDataset(path, transform=transf_hm)
            dataloader = torch.utils.data.DataLoader(
                ht_dataset, batch_size=1, shuffle=False, num_workers=4
            )
            ht_dataiter = iter(dataloader)
            dataiter = iter(self.testloader)

            grads = list()
            imgs = list()
            outs = list()
            biased = list()
            lbs = list()

            for i, (databatch_imgs, databatch_lbs, databatch_biased) in enumerate(
                dataiter
            ):
                label = databatch_lbs[0]
                if sum(list(label.size())) > 1:
                    label = torch.argmax(label)
                label = label.to(device=self.device)
                if self.biased_class is None or label == self.biased_class:
                    _grad, _out = next(ht_dataiter)
                    grads.append(_grad[0])
                    outs.append(_out.unsqueeze(dim=0))
                    imgs.append(databatch_imgs[0].unsqueeze(dim=0))
                    biased.append(databatch_biased[0].unsqueeze(dim=0))
                    lbs.append(label.unsqueeze(dim=0))

            lbs = torch.Tensor(lbs).to(device=self.device)
            grads = torch.cat(grads)
            imgs = torch.cat(imgs)
            outs = torch.Tensor(outs).to(device=self.device)
            biased = torch.cat(biased)

            self.imgs, self.grads, self.outs, self.lbs, self.biased = (
                imgs,
                grads,
                outs,
                lbs,
                biased,
            )

            print("HEATMAPS LOADED")

    def execute_pca(self, pca: bool = True):
        if pca:
            folder_name = (
                type(self.explainer).__name__
                + type(self.model).__name__
                + str(self.rescale)
                + type(self.trainset).__name__
                + bias_method2str(
                    self.bias_method, self.biased_class, self.bias_probability
                )
            )
            folder_path = os.path.join(HEATMAPS_PATH, folder_name)
            file_name = os.path.join(folder_path, "PCA" + folder_name + ".h5")
            if not check_pca(
                type(self.explainer).__name__,
                type(self.model).__name__,
                self.rescale,
                type(self.trainset).__name__,
                bias_method2str(
                    self.bias_method, self.biased_class, self.bias_probability
                ),
            ):
                print("EXECUTING PCA")
                from sklearn.decomposition import PCA

                N, H, W = self.grads.size()
                grads = self.grads.view(N, -1).cpu().detach().numpy()

                reduction_factor = max(min(int(0.8 * W * H), N), 1)
                self.pca_grads = PCA(n_components=reduction_factor).fit_transform(grads)
                h5f = h5py.File(file_name, "w")
                h5f.create_dataset(PCA_DATASET, data=self.pca_grads)
                h5f.close()
                add_pca(
                    type(self.explainer).__name__,
                    type(self.model).__name__,
                    self.rescale,
                    type(self.trainset).__name__,
                    bias_method2str(
                        self.bias_method, self.biased_class, self.bias_probability
                    ),
                )
                print("PCA EXECUTION ENDED")
            else:
                print("PCA COMPRESSED DATA LOADING ")
                h5f = h5py.File(file_name, "r")
                self.pca_grads = h5f[PCA_DATASET][:]
                h5f.close()
                print("PCA COMPRESSED DATA LOADING ENDED")
        else:
            self.pca_grads = None

    def cluster_heatmap(
        self, clustering_method: ClusterMixin = None, biased_class: Optional[int] = None
    ):

        if clustering_method is not None:
            self.clustering_method = clustering_method
        elif self.clustering_method is None:
            raise RuntimeError("No clustering method specified")

        biased_class = self.biased_class if self.biased_class is not None else -1
        pickle_name = (
            type(clustering_method).__name__
            + type(self.explainer).__name__
            + type(self.model).__name__
            + str(self.rescale)
            + type(self.trainset).__name__
            + bias_method2str(
                self.bias_method, self.biased_class, self.bias_probability
            )
            + str(biased_class)
            + "PCA"
            if self.pca
            else "" + ".pt"
        )
        path = os.path.join(CLUSTERS_PATH, pickle_name)

        if not check_clustering(
            type(clustering_method).__name__,
            type(self.explainer).__name__,
            type(self.model).__name__,
            self.rescale,
            type(self.trainset).__name__,
            bias_method2str(self.bias_method, self.biased_class, self.bias_probability),
            biased_class,
            "PCA" if self.pca else "",
        ):

            grads_ = self.pca_grads if self.pca else self.grads.cpu().detach().numpy()
            print("Started Clustering")
            start_time = time.time()
            out = self.clustering_method.fit_predict(
                grads_.reshape(grads_.shape[0], -1)
            )
            print(
                "-- Time taken to cluster: ",
                strftime("%Hh:%Mm:%Ss", gmtime(time.time() - start_time)),
                " --",
            )
            with open(path, "wb") as f:
                pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)
            add_clustering(
                type(clustering_method).__name__,
                type(self.explainer).__name__,
                type(self.model).__name__,
                self.rescale,
                type(self.trainset).__name__,
                bias_method2str(
                    self.bias_method, self.biased_class, self.bias_probability
                ),
                biased_class,
                "PCA" if self.pca else "",
            )
            assess = cluster_accuracy(self.biased.cpu().detach().numpy(), out)
            print("Cluster quality: ", assess)
            return out, assess
        else:
            print("LOADING CLUSTER")
            with open(path, "rb") as cluster_file:
                out = pickle.load(cluster_file)
            print("CLUSTER LOADED")
            assess = cluster_accuracy(self.biased.cpu().detach().numpy(), out)
            print("Cluster quality: ", assess)
            return out, assess

    def train(
        self,
        optimizer: Optional[Optimizer] = None,
        criterion: nn.modules.loss._Loss = nn.CrossEntropyLoss(),
        device: Union[str, torch.device] = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
        epochs: int = 6,
        display_every: int = 10,
    ):

        if not check_model(
            type(self.model).__name__,
            type(self.trainset).__name__,
            bias_method2str(self.bias_method, self.biased_class, self.bias_probability),
        ):
            if isinstance(device, str):
                device = torch.device(device)

            if self.model is None:
                raise RuntimeError("Model not defined")
            else:
                if optimizer is None:
                    optimizer = optim.Adam(self.model.parameters(), lr=0.001)
                print("STARTING MODEL TRAINING FOR %d EPOCHS" % epochs)

                self.model.to(device=device)
                self.model.train()
                # total_class: dict = {}
                # correct_class: dict = {}
                for epoch in range(epochs):  # loop over the dataset multiple times
                    running_acc = 0.0
                    running_loss = 0.0
                    total, correct = 0, 0

                    for i, data in enumerate(self.trainloader, 0):
                        # get the inputs
                        inputs, labels, _ = data

                        inputs, labels = inputs.to(device=device), labels.to(
                            device=device
                        )
                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward + backward + optimize
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)

                        # print statistics
                        running_loss += loss.item()

                        correct_labels = labels
                        if len(labels.shape) > 1:
                            _, correct_labels = torch.max(labels, 1)
                        correct += (predicted == correct_labels).sum().item()
                        running_acc = correct / total
                        # if i % display_every == display_every - 1:
                        #     print('[Epoch %d, batch %5d] loss: %.4f' %
                        #           (epoch + 1, i + 1, running_loss / total))
                        #     running_loss = 0.0
                        if i % display_every == display_every - 1:
                            print(
                                "[Epoch %d, batch %5d] loss: %.4f, accuracy %.4f"
                                % (epoch + 1, i + 1, running_loss / total, running_acc)
                            )
                            running_loss = 0.0
                    total = 0

                correct = 0
                running_loss = 0

                with torch.no_grad():
                    self.model.eval()
                    for data in self.testloader:
                        images, labels, _ = data
                        images, labels = images.to(device=device), labels.to(
                            device=device
                        )
                        outputs = self.model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        loss = criterion(outputs, labels)
                        running_loss += loss.item()
                        # if int(labels) not in total_class:
                        #     total_class[int(labels)] = 0
                        #     correct_class[int(labels)] = 0
                        # total_class[int(labels)] += 1
                        # correct_class[int(labels)] += (predicted == labels).sum().item()
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                print(
                    "Accuracy of the network on the 10000 test images: %.3f, loss: %.3f"
                    % (correct / total, running_loss / total)
                )
                # print_acc_dict(total_class, correct_class)
                self.save_model(MODELS_PATH)
                _ = add_model(
                    type(self.model).__name__,
                    type(self.trainset).__name__,
                    bias_method2str(
                        self.bias_method, self.biased_class, self.bias_probability
                    ),
                )
        else:
            print("LOADING MODEL")
            model_name = (
                type(self.model).__name__
                + type(self.trainset).__name__
                + bias_method2str(
                    self.bias_method, self.biased_class, self.bias_probability
                )
                + ".pt"
            )
            path = os.path.join(MODELS_PATH, model_name)
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            print("MODEL LOADED")
        return self

    # def tsne(self, selected_class: Optional[int] = None, n_samples: int = 2000, only_correct: bool = False):
    def tsne(self, n_samples: int = 500, cluster: bool = False):
        perpl = 20
        _, H, W = self.grads.size()
        grads_tsne = self.grads.cpu().detach().numpy()[:n_samples].reshape(-1, H * W)
        X_embedded = TSNE(n_components=2, perplexity=perpl).fit_transform(grads_tsne)
        _, ax = plt.subplots()
        c = self.clusters if cluster else self.biased.cpu().detach().numpy()
        scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=c[:n_samples])
        legend1 = ax.legend(
            *scatter.legend_elements(), loc="lower left", title="Classes"
        )
        ax.add_artist(legend1)
        plt.grid("on")
        plt.show()

        # TODO correct un-normalization of images and heatmaps.
        bokeh_imgs = self.imgs.cpu().detach().numpy()
        bokeh_imgs = (np.transpose(bokeh_imgs * 0.5 + 0.5, [0, 2, 3, 1])) * 255
        bokeh_htmps = self.grads.cpu().detach().numpy() * 255

        self.sbokeh(
            X_embedded[:, 0],
            X_embedded[:, 1],
            c,
            np.arange(c.shape[0]),
            bokeh_imgs,
            bokeh_htmps,
        )

        return self

    def show_heatmap(
        self,
        num: Optional[int] = None,
        selected_class: Optional[int] = None,
        only_correct: bool = False,
        normalization: tuple = (0.5, 2),
    ) -> None:
        """
        Method to show heatmaps. DOES NOT PROVIDE SAVING, which was moved to the _save_heatmap() method.
        Since the grads will not be saved as elements of the class, we could get rid of this method (maybe move it in utils.)
        """

        if selected_class is None:
            grads_to_show, images_to_show = self.grads[:num], self.imgs[:num]
        else:
            mask = torch.nonzero(self.lbs == selected_class).view(-1)
            if only_correct:
                mask = mask[torch.nonzero(self.lbs[mask] == self.outs[mask]).view(-1)]
            grads_to_show, images_to_show = (
                self.grads[mask[:num]],
                self.imgs[mask[:num]],
            )

        mu, sigma = normalization
        grads_to_show, images_to_show = (
            grads_to_show.cpu().detach().numpy(),
            images_to_show.cpu().detach().numpy(),
        )

        for i in range(grads_to_show.shape[0]):
            fig, axes = plt.subplots(1, 2, figsize=(8, 8))

            _ = axes[0].imshow(
                np.transpose((images_to_show[i] / sigma) + mu, (1, 2, 0))
            )
            _ = axes[1].imshow(grads_to_show[i])
            axes[0].set(xticklabels=[])
            axes[0].set(yticklabels=[])
            axes[0].tick_params(left=False, bottom=False)
            axes[1].set(xticklabels=[])
            axes[1].set(yticklabels=[])
            axes[1].tick_params(left=False, bottom=False)
        plt.show()

    def save_model(self, dir: str):
        if self.model is not None:
            model_name = (
                type(self.model).__name__
                + type(self.trainset).__name__
                + bias_method2str(
                    self.bias_method, self.biased_class, self.bias_probability
                )
                + ".pt"
            )
            path = os.path.join(dir, model_name)
            print("SAVING AT PATH: {}".format(path))
            torch.save(self.model.state_dict(), path)
            print("SAVED MODEL")
        else:
            raise RuntimeError("Model is not defined. Saving aborted.")

    def sbokeh(self, x, y, labels, mask, imgs, grads):
        # Bokeh Libraries
        from bokeh.plotting import figure, show
        from bokeh.io import output_file
        from bokeh.models import ColumnDataSource
        import pandas as pd
        import base64
        import cv2

        # To visualize images from RAM memory
        def encode_base64_img(img):
            jpg_img = cv2.imencode(".jpg", img)
            b64_string = base64.b64encode(jpg_img[1]).decode("utf-8")
            return "data:image/png;base64,%s" % b64_string

        # To visualize images from /nas
        def encode_base64(img_path):
            encoded_string = ""
            with open(img_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
            return "data:image/png;base64,%s" % encoded_string.decode("utf-8")

        # Output to file

        biased_class = self.biased_class if self.biased_class is not None else -1
        html_name = (
            type(self.clustering_method).__name__
            + type(self.explainer).__name__
            + type(self.model).__name__
            + str(self.rescale)
            + type(self.trainset).__name__
            + bias_method2str(
                self.bias_method, self.biased_class, self.bias_probability
            )
            + str(biased_class)
            + "PCA"
            if self.pca
            else "" + ".html"
        )
        file_name = os.path.join(HTML_PATH, html_name)
        output_file(file_name, title="T-SNE")

        # Store the data in a ColumnDataSource
        colors = ["blue", "red"]
        df = pd.DataFrame(
            [
                [
                    x[i],
                    y[i],
                    labels[i],
                    colors[int(labels[i])],
                    encode_base64_img(imgs[mask[i]]),
                    encode_base64_img(grads[mask[i]]),
                ]
                for i in range(x.shape[0])
            ],
            columns=["x", "y", "labels", "colors", "imgs", "htmps"],
        )
        source = ColumnDataSource(data=df)

        # Specify the selection tools to be made available
        select_tools = ["box_select", "lasso_select", "poly_select", "tap", "reset"]

        # Create the figure
        fig = figure(
            plot_height=400,
            plot_width=600,
            x_axis_label="x",
            y_axis_label="y",
            title="T-SNE",
            toolbar_location="below",
            tools=select_tools,
        )

        # Add square representing each player
        fig.square(
            x="x",
            y="y",
            source=source,
            color="colors",
            selection_color="deepskyblue",
            nonselection_color="lightgray",
            nonselection_alpha=0.1,
        )

        # Bokeh Library
        from bokeh.models import HoverTool

        tooltips = """
        <div>
            <div>
                <img
                    src="@htmps" height="100" alt="@heatmaps" width="100"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"
                ></img>
            </div>
            <div>
                <img
                    src="@imgs" height="100" alt="@imgs" width="100"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"
                ></img>
            </div>
        </div>

        """

        # Add the HoverTool to the figure
        fig.add_tools(HoverTool(tooltips=tooltips))

        # Visualize
        show(fig)
