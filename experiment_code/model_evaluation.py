"""
This code provides functions for evaluating a model by allowing
a user to calculate the accuracy or accuracy on each source 
of a model on a dataset.
"""

import numpy as np
import jax
import flax
import flax.linen as nn
import torch.utils.data as torchdata
import typing
import tqdm
from functools import partial
from .progress import tqdm_style


def accuracy_score(
    model: nn.Module,
    params: flax.core.frozen_dict.FrozenDict,
    dl: torchdata.DataLoader,
    verbose: bool = True,
    **apply_kwargs,
) -> float:
    """
    This will calculate the accuracy of a model.

    Arguments
    ---------
    - model: nn.Module:
        The model to be evaluated.

    - params: flax.core.frozen_dict.FrozenDict:
        The parameters of the model to be evaluated.

    - dl: torch.utils.data.DataLoader:
        The data loader to evaluate the model with.

    - verbose: bool:
        Whether to show a progress bar or not.

    - **apply_kwargs:
        Any other arguments that will be passed to the models apply
        function. This is useful for passing in the train option.

    Returns
    ---------
    - acc: float:
        The accuracy of the model.
    """
    acc = 0
    apply_fn = jax.jit(partial(model.apply, params, **apply_kwargs))
    for x, s, y in tqdm.tqdm(
        dl,
        desc="Calculating accuracy",
        total=len(dl),
        disable=not verbose,
        **tqdm_style,
    ):
        y_pred = apply_fn(x, s)
        acc += (y_pred.argmax(-1) == y).sum().item()
    return acc / len(dl.dataset)


def source_accuracy_score(
    model: nn.Module,
    params: flax.core.frozen_dict.FrozenDict,
    dl: torchdata.DataLoader,
    verbose: bool = True,
    **apply_kwargs,
) -> typing.Dict[str, float]:
    """
    This will calculate the accuracy of a model on each source.

    Arguments
    ---------
    - model: nn.Module:
        The model to be evaluated.

    - params: flax.core.frozen_dict.FrozenDict:
        The parameters of the model to be evaluated.

    - dl: torch.utils.data.DataLoader:
        The data loader to evaluate the model with.

    - verbose: bool:
        Whether to show a progress bar or not.

    - **apply_kwargs:
        Any other arguments that will be passed to the models apply
        function. This is useful for passing in the train option.

    Returns
    ---------
    - acc: typing.Dict[str, float]:
        The accuracy of the model on each source. The
        keys of the dictionary are the sources and the
        values are the accuracies.
    """
    acc = {}
    counts = {}
    apply_fn = jax.jit(partial(model.apply, params, **apply_kwargs))
    for inputs, sources, targets in tqdm.tqdm(
        dl,
        desc="Calculating accuracy",
        total=len(dl),
        disable=not verbose,
        **tqdm_style,
    ):  # iterating over the batches
        y_pred = apply_fn(inputs, sources)  # getting the predictions
        s_unique = np.unique(sources)  # getting the unique sources

        sum_i = np.sum(
            np.stack([y_pred.argmax(-1) == targets] * len(s_unique), axis=1),
            axis=0,
            where=(sources == s_unique.reshape(-1, 1)).T,
        )
        count_i = (sources == s_unique.reshape(-1, 1)).T.sum(axis=0)

        # saving sum and count for each source
        for s_i, s in enumerate(s_unique):
            if s not in acc:
                acc[s] = 0
                counts[s] = 0

            acc[s] += sum_i[s_i]
            counts[s] += count_i[s_i]

    return {
        k: v / counts[k] for k, v in acc.items()
    }  # returning the accuracy for each source
