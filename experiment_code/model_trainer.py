"""
This model provides 
"""

import typing
import jax
import jax.numpy as jnp
import flax
import typing
from flax.training import train_state
from flax import struct
import torch.utils.data as torchdata
from functools import partial
import tqdm
from torch.utils.tensorboard import SummaryWriter

from .progress import tqdm_style


class RunningMean:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.values = []
        self.mean = None
        self.current_position = 0

    def update(self, value):
        self.values.append(value)
        self.current_position += 1
        if len(self.values) < self.window_size:
            self.mean = None
        elif len(self.values) == self.window_size:
            self.mean = sum(self.values) / self.window_size
        else:
            self.mean = (
                self.mean
                - self.values[self.current_position - self.window_size - 1]
                / self.window_size
                + value / self.window_size
            )
        return


class TrainState(train_state.TrainState):
    """
    Arguments
    ---------

    - key: jax.random.KeyArray:
        The key for the random number generator in the models.

    - batch_stats: typing.Any:
        This is the batch statistics for the batch normalization.

    """

    key: jax.random.KeyArray = None
    batch_stats: typing.Any = None
    train: bool = struct.field(pytree_node=False, default=False)


class TrainStateSourceMapping(train_state.TrainState):
    """
    This is the train state used for the models that allow you
    to caluclate the source mapping each batch.

    Arguments
    ---------

    - predictor_fn: typing.Callable:
        The function that will be used to get the predictions from the model.

    - source_mapper_raw: jax.core.FrozenDict:
        The parameters of the source mapper.

    - weight: float:
        This is the ratio of weight given to the
        new mapper vs the current mapper.
        :code:`current_mapper = (weight/(weight+1))*new_mapper + (1/(weight+1))*current_mapper`.
        Defaults to :code:`1.0`.

    - key: jax.random.KeyArray:
        The key for the random number generator in the models.

    - batch_stats: typing.Any:
        This is the batch statistics for the batch normalization.

    - train: bool:
        Whether to pass :code:`train=False` or :code:`train=True` to the model when
        the outputs are being predicted before doing the source mapping
        calculations. Defaults to :code:`False`.

    """

    predictor_fn: typing.Callable = struct.field(pytree_node=False)
    source_mapper_raw: typing.Any
    weight: float = 1.0
    batch_stats: typing.Any = None
    key: jax.random.KeyArray = None
    train: bool = struct.field(pytree_node=False, default=False)

    def apply_source_mapping(
        self,
        batch,
    ):

        model_params = dict(
            params=self.params,
            source_mapper=self.source_mapper,
            **(
                {"batch_stats": self.batch_stats}
                if self.batch_stats is not None
                else {}
            ),
        )

        inputs, source, label = batch[0], batch[1], batch[2]
        y_hat = self.apply_fn(
            model_params,
            inputs,
            method=self.predictor_fn,
            **({"train": self.train} if self.train is not None else {}),
        )
        new_source_mapper_raw = self.mapper_update(
            y_hat, label, source, weight=self.weight
        )

        return self.replace(source_mapper_raw=new_source_mapper_raw)

    @jax.jit
    def mapper_update(self, y_hat, label, source, weight):
        current_mapper_raw = flax.core.unfreeze(self.source_mapper_raw).copy()

        n_labels = y_hat.shape[-1]
        n_sources = current_mapper_raw["kernel"].shape[0]

        label = jax.nn.one_hot(label, n_labels)
        y_hat_argmax = jnp.argmax(y_hat, axis=-1)

        # finding the mask for which of the data points have the source s and label y
        mask = jnp.logical_and(
            (source == jnp.arange(n_sources).reshape(-1, 1)).T.reshape(
                -1, n_sources, 1
            ),
            (y_hat_argmax == jnp.arange(n_labels).reshape(-1, 1)).T.reshape(
                -1, 1, n_labels
            ),
        )
        # this has shape (BATCH_SIZE, n_sources, n_labels)

        # calculate the mean
        new_mapper_raw = jnp.mean(
            # this has shape (batch_size, n_sources, n_labels, n_labels)
            jnp.stack([jnp.stack([label] * n_labels, axis=1)] * n_sources, axis=1),
            axis=0,
            where=mask.reshape(-1, n_sources, n_labels, 1),
        )

        # where the new mapper is nan, use the current mapper,
        # otherwise scale the current mapper by the weighting.
        # this is the equivalent of
        # mask = ~jnp.isnan(new_mapper)
        # current_mapper_raw[mask] = (
        #     (1/(weight+1))*current_mapper_raw[mask] + (weight/(weight+1))*new_mapper[mask]
        # )
        current_mapper_raw["kernel"] = jnp.where(
            jnp.isnan(new_mapper_raw),
            current_mapper_raw["kernel"],
            (1 / (weight + 1)) * current_mapper_raw["kernel"]
            + (weight / (weight + 1)) * new_mapper_raw,
        )

        return flax.core.freeze(current_mapper_raw)

    @property
    @jax.jit
    def source_mapper(self):
        return flax.core.freeze(
            {
                "kernel": jnp.where(
                    self.source_mapper_raw["kernel"]
                    == jnp.max(
                        self.source_mapper_raw["kernel"], axis=-1, keepdims=True
                    ),
                    jnp.float32(1),
                    jnp.float32(0),
                )
            }
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, source_mapper, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            source_mapper_raw=source_mapper,
            **kwargs,
        )


# ==== the loss function ====
def _calc_loss(
    params: flax.core.FrozenDict,
    state: flax.training.train_state.TrainState,
    batch: typing.Tuple[jnp.DeviceArray, ...],
    train: bool,
    criterion: typing.Callable,
    mutable: typing.List[str] = False,
    rngs: typing.Union[None, typing.List[str]] = None,
    apply_fn_kwargs: typing.Dict[str, typing.Any] = {},
):
    """
    Calculates the loss for a batch. This is the default
    function used when training a model using the
    :code:`FitModel` class.
    """
    inputs = batch[:-1]
    targets = batch[-1]

    model_params = dict(params=params)
    if train:
        if rngs is not None:
            new_key = jax.random.fold_in(key=state.key, data=state.step)
            rngs = {
                col: key for col, key in zip(rngs, jax.random.split(new_key, len(rngs)))
            }

    if mutable != False:
        model_params = dict(params=params, **{m: getattr(state, m) for m in mutable})

    outputs = state.apply_fn(
        model_params,
        *inputs,
        train=train,
        mutable=mutable,
        rngs=rngs,
        **apply_fn_kwargs,
    )

    if mutable != False:
        y_hat = outputs[0]
        updates = outputs[1]
    else:
        y_hat = outputs

    loss_value = (
        criterion(y_hat, targets).sum() / targets.shape[0]
    )  # mean loss value over batch

    if mutable != False and train:
        return loss_value, updates

    return loss_value


# ==== the stepping function ====
def _do_step(
    batch: typing.Tuple[jnp.DeviceArray, ...],
    state: flax.training.train_state.TrainState,
    loss_fn: typing.Callable,
    mutable: typing.List[str] = False,
):
    """
    Performs a step using a train state state and batch.
    This is the default function used when training a
    model using the :code:`FitModel` class.
    """

    output, grads = jax.value_and_grad(loss_fn, has_aux=(mutable != False))(
        state.params,
        state=state,
        batch=batch,
    )

    state = state.apply_gradients(grads=grads)

    if mutable != False:
        loss_value = output[0]
        updates = output[1]
        state = state.replace(**updates)
    else:
        loss_value = output

    return state, loss_value


# ==== the epoch function ====
def _do_epoch(
    cls: object,
    train_dl: torchdata.DataLoader,
    state: flax.training.train_state.TrainState,
    step_fn: typing.Callable,
):
    """
    Performs an epoch using a train state and batch.
    This is the default function used when training a
    model using the :code:`FitModel` class.
    """

    cls.tqdm_bar.set_description(f"Epoch {cls.i_epoch+1}/{cls.n_epochs}")
    loss_value_epoch = 0
    # ==== looping on the batches ====
    for ib, batch in enumerate(train_dl):

        state, loss_value = step_fn(
            batch=batch,
            state=state,
        )
        loss_value_epoch += loss_value.item() * len(batch[0])
        cls.running_mean.update(loss_value.item())
        if cls.writer is not None:
            cls.writer.add_scalar(
                "training_loss", loss_value.item(), cls.i_epoch * len(train_dl) + ib
            )
        if cls.running_mean.mean is not None:
            cls.postfix["training loss"] = f"{cls.running_mean.mean:.2f}"
        cls.tqdm_bar.set_postfix(cls.postfix)
        cls.tqdm_bar.update(1)

    loss_value_epoch *= 1 / len(train_dl.dataset)
    cls.loss_dict["train"].append(loss_value_epoch)

    return cls, state


# ==== the eval function ====
def _do_eval(
    cls: object,
    eval_dl: torchdata.DataLoader,
    state: flax.training.train_state.TrainState,
    loss_fn: typing.Callable,
    early_stop: flax.training.early_stopping = None,
    step: int = 0,
):
    """
    Performs a validation pass using a train state
    and batch. This is the default function used when training a
    model using the :code:`FitModel` class.
    """

    # ==== calculating val loss ===
    loss_value_epoch = 0
    for ib, batch in enumerate(eval_dl):
        loss_value = loss_fn(
            params=state.params,
            state=state,
            batch=batch,
        )
        loss_value_epoch += loss_value.item() * len(batch[0])
    loss_value_epoch *= 1 / len(eval_dl.dataset)
    if early_stop is not None:
        _, early_stop = early_stop.update(loss_value_epoch)
    cls.loss_dict["validation"].append(loss_value_epoch)
    cls.postfix["validation loss"] = f"{loss_value_epoch:.2f}"
    cls.tqdm_bar.set_postfix(cls.postfix)
    cls.tqdm_bar.refresh()

    if cls.writer is not None:
        cls.writer.add_scalar("validation_loss", loss_value_epoch, step)
    return cls, early_stop


# ==== the fitting function ====
def _do_fit(
    cls,
    state: flax.training.train_state.TrainState,
    train_dl: torchdata.DataLoader,
    n_epochs: int,
    criterion: typing.Callable,
    val_dl: typing.Union[torchdata.DataLoader, None] = None,
    train_mutable: typing.List[str] = False,
    train_rngs: typing.Union[None, typing.List[str]] = None,
    early_stop: flax.training.early_stopping = None,
    writer: typing.Union[None, SummaryWriter] = None,
    apply_fn_kwargs: typing.Dict[str, typing.Any] = {},
    do_jit: bool = True,
):
    """
    Fit a model using the trainstate and data.
    This is the default function used when training a
    model using the :code:`FitModel` class.
    """

    cls.loss_fn_train = partial(
        cls.loss_fn,
        train=True,
        criterion=criterion,
        mutable=train_mutable,
        rngs=train_rngs,
        apply_fn_kwargs=apply_fn_kwargs,
    )

    cls.loss_fn_train = jax.jit(cls.loss_fn_train) if do_jit else cls.loss_fn_train

    cls.step_fn = partial(
        cls.step_fn,
        loss_fn=cls.loss_fn_train,
        mutable=train_mutable,
    )
    cls.step_fn = jax.jit(cls.step_fn) if do_jit else cls.step_fn

    cls.loss_fn_eval = partial(
        cls.loss_fn,
        train=False,
        criterion=criterion,
        mutable=train_mutable,
        apply_fn_kwargs=apply_fn_kwargs,
    )
    cls.loss_fn_eval = jax.jit(cls.loss_fn_eval) if do_jit else cls.loss_fn_eval

    cls.n_epochs = n_epochs
    cls.tqdm_bar = tqdm.tqdm(
        desc="fit",
        total=len(train_dl) * n_epochs,
        disable=not cls.verbose,
        **tqdm_style,
    )
    cls.postfix = {}
    cls.running_mean = RunningMean(window_size=len(train_dl))
    cls.loss_dict = {"train": [], "validation": []}
    cls.i_epoch = 0
    cls.writer = writer

    while cls.i_epoch < cls.n_epochs:
        cls, state = cls.epoch_fn(
            cls,
            train_dl=train_dl,
            state=state,
            step_fn=cls.step_fn,
        )

        cls, early_stop = cls.eval_fn(
            cls,
            eval_dl=val_dl,
            state=state,
            loss_fn=cls.loss_fn_eval,
            early_stop=early_stop,
            step=(cls.i_epoch + 1) * len(train_dl),
        )

        cls.i_epoch += 1

        if cls.writer is not None:
            cls.writer.flush()

        if early_stop is not None:
            if early_stop.should_stop:
                break

    cls.tqdm_bar.close()

    return state


class FitModel(object):
    def __init__(
        self,
        loss_fn: typing.Callable = None,
        step_fn: typing.Callable = None,
        epoch_fn: typing.Callable = None,
        eval_fn: typing.Callable = None,
        fit_fn: typing.Callable = None,
        verbose: bool = True,
    ):
        """
        A class to fit a model using a train state and data.
        This class allows you to mix and match different loss functions,
        steps, epochs, and evaluation functions. This is useful if you
        want to use a different loss function for training and validation,
        but re-use other parts of the training loop.
        """
        self.loss_fn = _calc_loss if loss_fn is None else loss_fn
        self.step_fn = _do_step if step_fn is None else step_fn
        self.epoch_fn = _do_epoch if epoch_fn is None else epoch_fn
        self.eval_fn = _do_eval if eval_fn is None else eval_fn
        self.fit_fn = _do_fit if fit_fn is None else fit_fn

        self.verbose = verbose

        return

    def fit(self, *args, **kwargs):
        return self.fit_fn(self, *args, **kwargs)


### training functions specific to the calculating source mapper models ###

# ==== the loss function ====
def _calc_loss_source_mapper(
    params: flax.core.FrozenDict,
    state: flax.training.train_state.TrainState,
    batch: typing.Tuple[jnp.DeviceArray, ...],
    train: bool,
    criterion: typing.Callable,
    mutable: typing.List[str] = False,
    rngs: typing.Union[None, typing.List[str]] = None,
    apply_fn_kwargs: typing.Dict[str, typing.Any] = {},
    other_params: typing.Dict[str, typing.Any] = {},
):
    inputs = batch[:-1]
    targets = batch[-1]

    model_params = dict(params=params, **other_params)
    if train:
        if rngs is not None:
            new_key = jax.random.fold_in(key=state.key, data=state.step)
            rngs = {
                col: key for col, key in zip(rngs, jax.random.split(new_key, len(rngs)))
            }

    if mutable != False:
        model_params = dict(
            **model_params,
            **{m: getattr(state, m) for m in mutable},
        )

    outputs = state.apply_fn(
        model_params,
        *inputs,
        train=train,
        mutable=mutable,
        rngs=rngs,
        **apply_fn_kwargs,
    )

    if mutable != False:
        y_hat = outputs[0]
        updates = outputs[1]
    else:
        y_hat = outputs

    loss_value = (
        criterion(y_hat, targets).sum() / targets.shape[0]
    )  # mean loss value over batch

    if mutable != False and train:
        return loss_value, updates

    return loss_value


# ==== the stepping function ====
def _do_step_source_mapper(
    batch: typing.Tuple[jnp.DeviceArray, ...],
    state: flax.training.train_state.TrainState,
    loss_fn: typing.Callable,
    mutable: typing.List[str] = False,
):

    output, grads = jax.value_and_grad(loss_fn, has_aux=(mutable != False))(
        state.params,
        state=state,
        batch=batch,
        other_params={"source_mapper": state.source_mapper},
    )

    state = state.apply_gradients(grads=grads)

    if mutable != False:
        loss_value = output[0]
        updates = output[1]
        state = state.replace(**updates)
    else:
        loss_value = output

    state = state.apply_source_mapping(batch)

    return state, loss_value


def _do_eval_source_mapper(
    cls: object,
    eval_dl: torchdata.DataLoader,
    state: flax.training.train_state.TrainState,
    loss_fn: typing.Callable,
    early_stop: flax.training.early_stopping = None,
    step: int = 0,
):
    # ==== calculating val loss ===
    loss_value_epoch = 0
    for ib, batch in enumerate(eval_dl):
        loss_value = loss_fn(
            params=state.params,
            state=state,
            batch=batch,
            other_params={"source_mapper": state.source_mapper},
        )
        loss_value_epoch += loss_value.item() * len(batch[0])
    loss_value_epoch *= 1 / len(eval_dl.dataset)
    if early_stop is not None:
        _, early_stop = early_stop.update(loss_value_epoch)
    cls.loss_dict["validation"].append(loss_value_epoch)
    cls.postfix["validation loss"] = f"{loss_value_epoch:.2f}"
    cls.tqdm_bar.set_postfix(cls.postfix)
    cls.tqdm_bar.refresh()

    if cls.writer is not None:
        cls.writer.add_scalar("validation_loss", loss_value_epoch, step)
    return early_stop
