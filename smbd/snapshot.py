"""
Module created to save research data without redundancy.
Methods read and modify an history file to keep track of the changes
during experimentation.

Raises:
    RuntimeError: raised in case of error during reading from the history file
"""
# TODO Re-organize HISTORY as a collection of experiments?
# TODO reformat to nicer printing

from typing import Optional, List, Union, Tuple
from yaml import load, Loader, dump, Dumper
import os

NAS_PATH = "../../../../../../nas/data/nesti_pacini/"
HISTORY_PATH = os.path.join(NAS_PATH, "history.yml")

DATASETS = 'datasets'
MODELS = 'models'
HEATMAPS = 'heatmaps'
CLUSTERINGS = 'clusterings'
PCA = 'pca'


def _load_state() -> dict:
    with open(HISTORY_PATH, "r") as file:
        data_ = file.read()
    return load(data_, Loader=Loader)


def _write_state(state: dict) -> None:
    with open(HISTORY_PATH, "w") as f:
        f.write(dump(state, Dumper=Dumper))


def _check_state(item: Union[dict, str], element: str) -> Tuple[bool, dict, List]:
    state = _load_state()
    items = state.get(element)
    if items is None:
        items = []
    return item in items, state, items


def _add(item: Union[str, dict], element: str) -> bool:
    is_in, state, items = _check_state(item, element)

    if not is_in:
        items.append(item)

    state[element] = items
    _write_state(state)
    return is_in


def _remove(item: Union[str, dict], element: str) -> None:
    is_in, state, items = _check_state(item, element)

    if not is_in:
        raise RuntimeError(f"Item not present in {element}")

    items.remove(item)
    state[element] = items
    _write_state(state)


def add_dataset(dataset_name: str) -> bool:
    """
    Add a dataset in the history file under the `datasets` key.

    Args:
        dataset_name (str): The name of the dataset to add to the history file.
        This name should be descriptive of the modification done to the original dataset,
        such as transformations and bias added to a certain class.

    Returns:
        bool: Returns `True` in case the dataset has been added to the history file; if the
        dataset was not already present in the history file, returns `False`.
    """
    return _add(dataset_name.lower(), DATASETS)


def check_dataset(dataset_name: str) -> bool:
    is_in, _, _ = _check_state(dataset_name, DATASETS)
    return is_in


def remove_dataset(dataset_name: str) -> None:
    """
    Remove a dataset in the history file under the `datasets` key.
    If no dataset is found, raise a RuntimeError.


    Args:
        dataset_name (str): The name of the dataset to remove from the history file.
    """
    _remove(dataset_name.lower(), DATASETS)


def _to_model(model_name: str, dataset_name: str, bias_type: str) -> dict:
    return {
        'model_name': model_name.lower(),
        'dataset_name': dataset_name.lower(),
        'bias_type': bias_type.lower()
    }


def add_model(model_name: str, dataset_name: str, bias_type: str) -> bool:
    """
    Add a model in the history file under the `models` key.


    Args:
        model_name (str): The name of the model to add to the history file.
        dataset_name (str): The name of the dataset on which the model has been trained.
        bias_type (str): The description of the bias of the dataset on which the model has been trained.

    Returns:
        bool: Returns `True` in case the model has been added to the history file; if the
        model was not already present in the history file, it returns `False`.
    """
    model = _to_model(model_name, dataset_name, bias_type)
    return _add(model, MODELS)


def remove_model(model_name: str, dataset_name: str, bias_type: str) -> None:
    """
    Remove a model in the history file under the `models` key.
    If no model is found, raise a RuntimeError.

    Args:
        model_name (str): The name of the model to remove from the history file.
        dataset_name (str): The name of the dataset on which the model has been trained.
        This is necessary to identify the model in the history file.
        bias_type (str): The description of the bias of the dataset on which the model has been trained.
        This is necessary to identify the model in the history file.
    """
    model = _to_model(model_name, dataset_name, bias_type)
    _remove(model, MODELS)


def check_model(model_name: str, dataset_name: str, bias_type: str) -> bool:
    """
    Check if a model is present in the history file under the `models` key.


    Args:
        model_name (str): The name of the model to find to the history file.
        dataset_name (str): The name of the dataset on which the model has been trained.
        bias_type (str): The description of the bias of the dataset on which the model has been trained.

    Returns:
        bool: Returns `True` in case the model has been found in the history file; if the
        model was not already present in the history file, it returns `False`.
    """
    model = _to_model(model_name, dataset_name, bias_type)
    is_in, _, _ = _check_state(model, MODELS)
    return is_in


def _to_heatmap(heatmap_name: str,
                model_name: str,
                rescale_factor: Optional[int],
                dataset_name: str,
                bias_type: str) -> dict:
    return {
        'heatmap_name': heatmap_name.lower(),
        'model_name': model_name.lower(),
        'rescale_factor': str(rescale_factor) if rescale_factor is not None else '',
        'dataset_name': dataset_name.lower(),
        'bias_type': bias_type.lower()
    }


def add_heatmap(heatmap_name: str,
                model_name: str,
                rescale_factor: Optional[int],
                dataset_name: str,
                bias_type: str) -> bool:
    """
    Add a heatmap dataset in the history file under the `heatmaps` key.

    Args:
        heatmap_name (str): The type of heatmaps saved in the heatmap dataset recorded in the history file.
        model_name (str): The name of the model which produced such heatmaps.
        rescale_factor (float): Rescale factor applied to the original heatmaps.

    Returns:
        bool: Returns `True` in case the heatmap dataset has been added to the history file; if the
        heatmap dataset was not already present in the history file, it returns `False`.
    """
    heatmap = _to_heatmap(heatmap_name, model_name, rescale_factor, dataset_name, bias_type)
    return _add(heatmap, HEATMAPS)


def remove_heatmap(heatmap_name: str,
                   model_name: str,
                   rescale_factor: Optional[int],
                   dataset_name: str,
                   bias_type: str) -> None:
    """
    Remove an heatmap dataset in the history file under the `heatmaps` key.
    If no such dataset is found, raise a RuntimeError.

    Args:
        heatmap_name (str): The type of heatmaps saved in the heatmap dataset recorded in the history file.
        model_name (str): The name of the model which produced such heatmaps.
        This is necessary to identify the heatmap dataset in the history file.
        rescale_factor (float): Rescale factor applied to the original heatmaps.
        This is necessary to identify the heatmap dataset in the history file.
    """
    heatmap = _to_heatmap(heatmap_name, model_name, rescale_factor, dataset_name, bias_type)
    _remove(heatmap, HEATMAPS)


def check_heatmap(heatmap_name: str,
                  model_name: str,
                  rescale_factor: Optional[int],
                  dataset_name: str,
                  bias_type: str) -> bool:
    """
    Check if an heatmap dataset is in the history file under the `heatmaps` key.
    If no such dataset is found, raise a RuntimeError.

    Args:
        heatmap_name (str): The type of heatmaps saved in the heatmap dataset recorded in the history file.
        model_name (str): The name of the model which produced such heatmaps.
        This is necessary to identify the heatmap dataset in the history file.
        rescale_factor (float): Rescale factor applied to the original heatmaps.
        This is necessary to identify the heatmap dataset in the history file.

    Returns:
        bool: Returns `True` in case the heatmap dataset has been found in the history file; if the
        heatmap dataset was not already present in the history file, it returns `False`.
    """
    heatmap = _to_heatmap(heatmap_name, model_name, rescale_factor, dataset_name, bias_type)
    is_in, _, _ = _check_state(heatmap, HEATMAPS)
    return is_in


def _to_clustering(type: str,
                   heatmap: str,
                   model_name: str,
                   rescale_factor: Optional[int],
                   dataset_name: str,
                   bias_type: str,
                   class_: int,
                   pca: str) -> dict:
    return {
        'type': type.lower(),
        'heatmap': heatmap.lower(),
        'model_name': model_name.lower(),
        'rescale_factor': str(rescale_factor) if rescale_factor is not None else '',
        'dataset_name': dataset_name.lower(),
        'bias_type': bias_type.lower(),
        'class': class_,
        'pca': pca.upper()
    }


def add_clustering(type: str,
                   heatmap: str,
                   model_name: str,
                   rescale_factor: Optional[int],
                   dataset_name: str,
                   bias_type: str,
                   class_: int,
                   pca: str) -> bool:
    """
    Add a clustering dataset in the history file under the `clusterings` key.

    Args:
        type (str): The type of clustering executed.
        heatmap (str): The heatmap dataset on which the clustering method has been executed.
        class_ (int): The class on which the heatmap has been created. Give a negative value if
        it has been executed on all the classes.

    Returns:
        bool: Returns `True` in case the clustering dataset has been added to the history file; if the
        clustering dataset was not already present in the history file, it returns `False`.
    """
    clustering = _to_clustering(type,
                                heatmap,
                                model_name,
                                rescale_factor,
                                dataset_name,
                                bias_type,
                                class_, pca)
    return _add(clustering, CLUSTERINGS)


def remove_clustering(type: str,
                      heatmap: str,
                      model_name: str,
                      rescale_factor: Optional[int],
                      dataset_name: str,
                      bias_type: str,
                      class_: int,
                      pca: str) -> None:
    """
    Remove a clustering dataset in the history file under the `heatmaps` key.
    If no such dataset is found, raise a RuntimeError.

    Args:
        type (str): The type of clustering executed.
        heatmap (str): The heatmap dataset on which the clustering method has been executed.
        This is necessary to identify the clustering dataset in the history file.
        class_ (int):  The class on which the heatmap has been created. Give a negative value if
        it has been executed on all the classes.
        This is necessary to identify the clustering dataset in the history file.
    """
    clustering = _to_clustering(type,
                                heatmap,
                                model_name,
                                rescale_factor,
                                dataset_name,
                                bias_type,
                                class_, pca)
    _remove(clustering, CLUSTERINGS)


def check_clustering(type: str,
                     heatmap: str,
                     model_name: str,
                     rescale_factor: Optional[int],
                     dataset_name: str,
                     bias_type: str,
                     class_: int,
                     pca: str) -> bool:
    """
    Check if a clustering dataset is present in the history file under the `clusterings` key.

    Args:
        type (str): The type of clustering executed.
        heatmap (str): The heatmap dataset on which the clustering method has been executed.
        class_ (int): The class on which the heatmap has been created. Give a negative value if
        it has been executed on all the classes.

    Returns:
        bool: Returns `True` in case the clustering dataset has been found in the history file; if the
        clustering dataset was not already present in the history file, it returns `False`.
    """
    clustering = _to_clustering(type,
                                heatmap,
                                model_name,
                                rescale_factor,
                                dataset_name,
                                bias_type,
                                class_, pca)
    is_in, _, _ = _check_state(clustering, CLUSTERINGS)
    return is_in


def check_pca(heatmap_name: str,
              model_name: str,
              rescale_factor: Optional[int],
              dataset_name: str,
              bias_type: str) -> bool:
    heatmap = _to_heatmap(heatmap_name, model_name, rescale_factor, dataset_name, bias_type)
    is_in, _, _ = _check_state(heatmap, PCA)
    return is_in


def add_pca(heatmap_name: str,
            model_name: str,
            rescale_factor: Optional[int],
            dataset_name: str,
            bias_type: str) -> bool:
    heatmap = _to_heatmap(heatmap_name, model_name, rescale_factor, dataset_name, bias_type)
    return _add(heatmap, PCA)


def remove_pca(heatmap_name: str,
               model_name: str,
               rescale_factor: Optional[int],
               dataset_name: str,
               bias_type: str):
    heatmap = _to_heatmap(heatmap_name, model_name, rescale_factor, dataset_name, bias_type)
    _remove(heatmap, PCA)


def print_summary():
    y = _load_state()
    print("============\n\tMODELS\n============")
    for element in y['models']:
        print(element)
    print("============\n\tHeatmaps\n============")
    for element in y['heatmaps']:
        print(element)
    print("============\n\tPCA\n============")
    for element in y['PCA']:
        print(element)
    print("============\n\tCLUSTERING\n============")
    for element in y['clusterings']:
        print(element)
