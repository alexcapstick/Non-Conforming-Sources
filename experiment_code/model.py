"""

This file contains the code for the models used in the experiments.
The models available are:

- SourceAppendMLP: This MLP allows you to pass the source along with \
    the input into the apply and call, and it appends the source \
    vector to the input vector.

- SeparateSourceClassifierMLP: This MLP allows you to pass the source \
    along with the input into the apply and call. Within this model, \
    a new classification layer is built per source.

- SourceClassifierAppendMLP: This MLP allows you to pass the source along with \
    the input into the apply and call. Within this model, the source vector is \
    appended to input to the final layer to.

- SourcePredictMappingMLP: This MLP allows you to pass the source along with \
    the input into the apply and call. This model contains two models: the \
    predictor network and the source mapping network. The predictor network \
    trains the model to predict the label, and the source mapping network \
    learns the mapping to the source dependent labels.

- SourceCalcMappingMLP: This MLP allows you to pass the source along with \
    the input into the apply and call. This model will then calculate, whilst
    training, the mapping matrix between the outputs and the source labels.

- SeparateSourceClassifierResNet1D: This ResNet1D allows you to pass the source \
    along with the input into the apply and call. Within this model, \
    a new classification layer is built per source.

- SourceClassifierAppendResNet1D: This ResNet1D allows you to pass the source along with \
    the input into the apply and call. Within this model, the source vector is \
    appended to input to the final layer to.

- SourcePredictMappingResNet1D: This ResNet1D allows you to pass the source along with \
    the input into the apply and call. This model contains two models: the \
    predictor network and the source mapping network. The predictor network \
    trains the model to predict the label, and the source mapping network \
    learns the mapping to the source dependent labels.

- SourceCalcMappingResNet1D: This ResNet1D allows you to pass the source along with \
    the input into the apply and call. This model will then calculate, whilst
    training, the mapping matrix between the outputs and the source labels.

    

All of these models are built using Flax Linen and use the :code:`apply`
and use the structure: :code:`model.apply(params, inputs, sources)`.

For all of the models, the source value must be an integer :code:`>= 0`.

"""

import typing
import flax.linen as nn
import numpy as np
import jax
import jax.numpy as jnp


# MLP models


class PredictorMLP(nn.Module):
    """
    A simple MLP with relu-dense layers, and a
    final dense layer.

    Examples
    ---------
    A model with an input, 2 hidden layers
    with 10 and 5 features, and an output of 10 features
    would be built with the following::

        Predictor(features=[10, 5, 10])

    Arguments
    ---------
    - features: typing.Sequence[int]:
        The number of features in each layer.

    """

    features: typing.Sequence[int]

    @nn.compact
    def __call__(self, x, train=False):

        # prediction model
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)

        return x


class SourceAppendMLP(nn.Module):
    """
    This MLP allows you to pass the source along with
    the input into the apply and call.

    Arguments
    ---------
    - features: typing.Sequence[int]:
        The number of features in each layer.

    - binary_dimensions: int:
        The number of binary dimensions to use for the source vector.
    """

    features: typing.Sequence[int]
    binary_dimensions: int

    def int_array_to_bin_array(self, x) -> jnp.ndarray:
        """
        Turns a numpy array of integers into a binary array.
        For example: [1, 2, 3] -> [[0, 0, 1], [0, 1, 0], [0, 1, 1]]

        Arguments
        ---------
        - x: jnp.ndarray:
            The array of integers to be converted to binary.

        """
        return (((x[:, None] & (1 << jnp.arange(self.binary_dimensions)))) > 0).astype(
            int
        )

    @nn.compact
    def __call__(self, x, source, train=False):

        # adding source to the last dimension
        # source is converted to binary vector first
        """
        x = nn.relu(
            nn.Dense(self.features[0])(
                jnp.concatenate([x, self.int_array_to_bin_array(source)], axis=-1)
            )
        )

        for feat in self.features[1:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        """

        x = PredictorMLP(features=self.features)(
            jnp.concatenate([x, self.int_array_to_bin_array(source)], axis=-1)
        )

        return x


class SeparateSourceMaskingDense(nn.Module):
    """
    This class is a modification to the dense
    layer which allows for a separate classifier
    per source. This assumes the input is of shape
    (N, F) where N is the number of data points and
    F is the number of feature values.

    Arguments
    ---------
    - features: typing.Sequence[int]:
        The number of features to output in this layer.

    - n_sources: int:
        The number of unique sources that this layer should expect.
    """

    features: int
    n_sources: int

    def mask_index(self, s):
        """
        This function generates a mask index for the dense layer
        so that the predictions from only the given source
        are selected.

        The masking is required to support JAX's
        tracing. If jit is not used, then the
        model SeparateSourceDenseNoJIT can be used,
        which instead uses a for loop to select the
        correct classifier for each source.

        Arguments
        ---------
        - s: jnp.ndarray:
            A vector containing the source values.

        """
        shape = s.shape[0]
        n_features = self.features

        # the below example is where:
        # n_features = 3
        # n_sources = 4
        # shape = 20

        # row_idx is like:
        # [[0, 0, ..., 0],
        #  [1, 1, ..., 1],
        #  [2, 2, ..., 2],
        #  ...
        #  [batch_size, batch_size, ..., batch_size]]
        # and has shape (batch_size, n_features_output)
        # This selects the correct row for each of the col_idx below

        # col_idx is like:
        # [[9, 10, 11], # source 3
        #  [3, 4, 5], # source 1
        #  [6, 7, 8], # source 2
        #  [0, 1, 2], # source 0
        #  ...
        #  [9, 10, 11]] # source 3
        # and has shape (batch_size, n_features_output)
        # This represents which part of the dense output corresponds
        # to the predictions on this source

        row_idx = np.repeat(np.arange(shape).reshape(-1, 1), n_features, axis=1)
        col_idx = (
            np.repeat(np.arange(n_features).reshape(1, -1), shape, axis=0)
            + s.reshape(-1, 1) * n_features
        )

        return row_idx, col_idx

    @nn.compact
    def __call__(self, x, source, train=False):
        x = nn.Dense(self.n_sources * self.features)(x)
        mask_idx = self.mask_index(source)
        return x[mask_idx[0], mask_idx[1]]


class SeparateSourceDense(nn.Module):
    """
    This class is a modification to the dense
    layer which allows for a separate classifier
    per source. This assumes the input is of shape
    (N, F) where N is the number of data points and
    F is the number of feature values.

    Arguments
    ---------
    - features: typing.Sequence[int]:
        The number of features to output in this layer.

    - n_sources: int:
        The number of unique sources that this layer should expect.
    """

    features: int
    n_sources: int
    bias: bool = True
    source_layers_kernel_init: typing.Callable = nn.initializers.lecun_normal()
    source_layers_bias_init: typing.Callable = nn.initializers.zeros
    heavyside: bool = False

    def straight_through_heaviside(self, x):
        """
        The heaviside function as a straight through estimator.
        This means it has gradient equal to 1 at all values,
        but behaves as you would expect.

        Arguments
        ---------
        - x: jnp.ndarray:
            The array to be passed through the heaviside function.

        """
        zero = x - jax.lax.stop_gradient(x)
        return zero + jax.lax.stop_gradient(jnp.heaviside(x, 0.5))

    @nn.compact
    def __call__(self, x, source, train=False):

        assert (self.heavyside == True and self.bias == False) or (
            self.heavyside == False
        ), "Bias cannot be used with heavyside."

        ## building the fully connected layer from scratch
        # kernel
        source_layers_kernel = self.param(
            "source_layers",
            self.source_layers_kernel_init,
            (self.n_sources, jnp.shape(x)[-1], self.features),
        )
        # bias
        source_layers_bias = self.param(
            "source_layers_bias",
            self.source_layers_bias_init,
            (self.n_sources, self.features),
        )

        # This is applying the dense layer to the input,
        # but only for the source that is passed in.
        if self.heavyside:
            out = jax.lax.dot_general(
                x,
                self.straight_through_heaviside(source_layers_kernel[source]),
                (((x.ndim - 1,), (1,)), ((0), (0))),
            )

        else:
            out = jax.lax.dot_general(
                x,
                source_layers_kernel[source],
                (((x.ndim - 1,), (1,)), ((0), (0))),
            )
        if self.bias:
            out += jnp.reshape(
                source_layers_bias[source, :],
                (out.shape[0],) + (1,) * (out.ndim - 2) + (-1,),
            )

        return out


class SeparateSourceClassifierMLP(nn.Module):
    """
    This MLP allows you to pass the source along with
    the input into the apply and call. Within
    this model, a new classification layer is built
    per source.

    Arguments
    ---------
    - features: typing.Sequence[int]:
        The number of features in each layer.

    - n_sources: int:
        The number of unique sources that this model should expect.
    """

    features: typing.Sequence[int]
    n_sources: int
    heavyside: bool = False
    bias_classifier: bool = True

    @nn.compact
    def __call__(self, x, source, train=False):

        """
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        """

        x = nn.relu(PredictorMLP(features=self.features)(x))
        x = SeparateSourceDense(
            self.features[-1],
            n_sources=self.n_sources,
            heavyside=self.heavyside,
            bias=self.bias_classifier,
        )(x, source=source)

        return x


class SourceClassifierAppendMLP(nn.Module):
    """

    This model will append the source vector
    to the layer before the output. This
    gives the model a layer to learn the mapping
    between the predictions and the source
    dependent labels.

    In addition to the layers created
    by the argument features, this model
    will create an extra layer of shape
    (features[-1]+binary_dimensions, features[-1]).

    Arguments
    ---------
    - features: typing.Sequence[int]:
        The number of features in each layer.
        Two layers, both of shape :code:`features[-1]`
        will be added to the end of the predictive model
        to map the source vector and output of the
        predictive model to the source labels.

    - binary_dimensions: int:
        The number of binary dimensions to use for the source vector.

    """

    features: typing.Sequence[int]
    binary_dimensions: int

    def int_array_to_bin_array(self, x) -> jnp.ndarray:
        """
        Turns a numpy array of integers into a binary array.
        For example: [1, 2, 3] -> [[0, 0, 1], [0, 1, 0], [0, 1, 1]]

        Arguments
        ---------
        - x: jnp.ndarray:
            The array of integers to be converted to binary.

        """
        return (((x[:, None] & (1 << jnp.arange(self.binary_dimensions)))) > 0).astype(
            int
        )

    @nn.compact
    def __call__(self, x, source, train=False):
        """
        # prediction model
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        """

        x = PredictorMLP(features=self.features)(x)

        # source on classifier layer
        # this appends the source vector to the output of the
        # last layer of the features before using another dense layer
        # to predict the output
        out = nn.relu(
            nn.Dense(self.features[-1])(
                jnp.concatenate([x, self.int_array_to_bin_array(source)], axis=-1)
            )
        )
        out = nn.Dense(self.features[-1])(out)

        return out


class SourceMapper(nn.Module):
    """

    This class will use the source value to predict a
    mapping between the output of the predictor
    and the output from a given source.

    Arguments
    ---------
    - features: typing.Sequence[int]:
        The number of features to output in this layer.

    - binary_dimensions: int:
        The number of binary dimensions to use for the source vector.

    - heavyside: bool:
        If :code:`True`, the heaviside function will be used, otherwise
        the sigmoid function will be used.
        Defaults to :code:`True`.
    """

    features: typing.Sequence[int]
    binary_dimensions: int
    heavyside: bool = True

    def int_array_to_bin_array(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Turns a numpy array of integers into a binary array.
        For example: [1, 2, 3] -> [[0, 0, 1], [0, 1, 0], [0, 1, 1]]

        Arguments
        ---------
        - x: jnp.ndarray:
            The array of integers to be converted to binary.

        """
        return (((x[:, None] & (1 << jnp.arange(self.binary_dimensions)))) > 0).astype(
            int
        )

    def straight_through_heaviside(self, x):
        """
        The heaviside function as a straight through estimator.
        This means it has gradient equal to 1 at all values,
        but behaves as you would expect.

        Arguments
        ---------
        - x: jnp.ndarray:
            The array to be passed through the heaviside function.

        """
        zero = x - jax.lax.stop_gradient(x)
        return zero + jax.lax.stop_gradient(jnp.heaviside(x, 0.5))

    @nn.compact
    def __call__(self, source, train=False):

        # source model
        # this model will predict the mapping between the output
        # and the source labels by using the source vector as
        # input to a relu-dense layer, a tanh-dense layer,
        # and finally a heaviside activation to get a binary mapping
        # between the output and the source labels.
        # It uses a straight through estimator to backpropagate through the
        # heaviside activation.
        s = self.int_array_to_bin_array(source)
        s = nn.relu(nn.Dense(self.features**2)(s))
        # s = nn.tanh(nn.Dense(self.features**2)(s))
        if self.heavyside:
            s = self.straight_through_heaviside(s)
        else:
            s = nn.sigmoid(s)
        s = s.reshape(s.shape[0], self.features, self.features)
        return s


class SourcePredictMappingMLP(nn.Module):
    """

    This class will use the source value to predict a
    mapping between the output of the predictor
    and the output from a given source. It
    will then use this mapping to make a final prediction
    on the output.

    Arguments
    ---------
    - features: typing.Sequence[int]:
        The number of features in each layer.

    - binary_dimensions: int:
        The number of binary dimensions to use for the source vector.

    """

    features: typing.Sequence[int]
    binary_dimensions: int

    def setup(self):
        self.predictor = PredictorMLP(
            features=self.features,
        )

        self.source_mapper = SourceMapper(
            features=self.features[-1],
            binary_dimensions=self.binary_dimensions,
        )

    def __call__(self, x, source, train=False):

        # prediction model
        x = self.predictor(x)
        # source model
        s = self.source_mapper(source)
        return jax.vmap(jnp.matmul, in_axes=(0, 0))(x, s)


class SourceCalcMappingMLP(nn.Module):
    """

    This class will use the source value to predict a
    mapping between the output of the predictor
    and the output from a given source. It
    will then use this mapping to make a final prediction
    on the output.

    Arguments
    ---------
    - features: typing.Sequence[int]:
        The number of features in each layer.
        The last value will be duplicated to
        create a mapping between the output of
        the predictive model and the source labels.

    - n_sources: int:
        The number of sources to expect.

    """

    features: typing.Sequence[int]
    n_sources: int

    def setup(self):

        self.predictor_model = PredictorMLP(
            features=self.features,
        )

        self.source_mapper_kernel = self.variable(
            "source_mapper",
            "kernel",
            lambda ns, nf: jnp.stack([jnp.eye(nf) for _ in range(ns)]),
            self.n_sources,
            self.features[-1],
        )

        return

    def predictor(self, x, train=False):
        # return nn.relu(self.predictor_model(x))
        return self.predictor_model(x)

    def source_map(self, x, source):
        return jax.lax.dot_general(
            x,
            self.source_mapper_kernel.value[source],
            (((x.ndim - 1,), (1,)), ((0), (0))),
        )

    def __call__(self, x, source, train=False):

        # prediction model
        x = self.predictor(x)

        # mapping sources
        x = self.source_map(x, source)
        return x


# ResNet1D Models


class ConvBlock1D(nn.Module):
    features: int
    kernel_size: int = 3
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(
        self,
        x,
        train,
    ):

        x = nn.Conv(
            self.features,
            kernel_size=(self.kernel_size,),  # 1d convolution
            padding="same",
            use_bias=False,
        )(x)
        x = nn.BatchNorm(
            momentum=0.9, use_running_average=not train, use_bias=True, use_scale=True
        )(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)

        return x


class ResNetSkipConnection1D(nn.Module):
    features: int
    width: int
    kernel_size: int = 3

    @nn.compact
    def __call__(self, y):
        in_width = y.shape[1]
        downsample = in_width // self.width
        if downsample > 1:
            same_pad = int(
                np.ceil(
                    0.5
                    * (
                        (in_width // self.width) * (self.width - 1)
                        - in_width
                        + downsample
                    )
                )
            )
            if same_pad < 0:
                same_pad = 0
            y = nn.max_pool(
                y,
                window_shape=(downsample,),
                strides=(downsample,),
                padding=((same_pad, same_pad),),
            )
        elif downsample == 1:
            pass
        else:
            raise ValueError("Size of input should always decrease.")

        y = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),  # 1d convolution
            strides=1,
            padding="same",
            use_bias=False,
        )(y)

        return y


class ResNetBlock1D(nn.Module):
    """
    This is an implementation of a 1D ResNet block in JAX.

    Arguments
    ---------
    - features: int:
        The number of channels in the output.

    - kernel_size: int:
        The size of the kernel to use in
        the convolutional and pooling layers.

    - dropout_rate: float:
        The dropout rate to use in the dropout layers.

    """

    features: int
    width: int
    kernel_size: int = 3
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x, y, train):

        in_width = x.shape[1]

        x = ConvBlock1D(
            self.features,
            self.kernel_size,
            self.dropout_rate,
        )(x, train=train)

        same_pad = int(
            np.ceil(
                0.5
                * (
                    (in_width // self.width) * (self.width - 1)
                    - in_width
                    + self.kernel_size
                )
            )
        )

        if same_pad < 0:
            same_pad = 0

        x = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            use_bias=False,
            padding=same_pad,
            strides=in_width // self.width,
        )(x)

        y = ResNetSkipConnection1D(
            features=self.features,
            width=self.width,
            kernel_size=self.kernel_size,
        )(y)

        xy = x + y
        y = x

        xy = nn.BatchNorm(
            momentum=0.9,
            use_running_average=not train,
            use_bias=False,
            use_scale=False,  # affine params are not used
        )(xy)
        xy = nn.relu(xy)
        xy = nn.Dropout(self.dropout_rate, deterministic=not train)(xy)

        return xy, y


class ResNet1D(nn.Module):
    """
    This is an implementation of a 1D ResNet block in JAX.

    Arguments
    ---------
    - features: int:
        The number of channels in the output.

    - kernel_size: int:
        The size of the kernel to use in
        the convolutional and pooling layers.

    - dropout_rate: float:
        The dropout rate to use in the dropout layers.

    - n_blocks: int:
        The number of ResNetBlock1D blocks to use in
        the full ResNet1D model.

    """

    features: int
    kernel_size: int = 16
    dropout_rate: float = 0.2
    n_blocks: int = 4

    @nn.compact
    def __call__(self, x, train):
        in_channels = x.shape[-1]
        in_width = x.shape[1]

        x = ConvBlock1D(
            features=in_channels,
            kernel_size=self.kernel_size,
            dropout_rate=0,  # no dropout in the first block
        )(x, train=train)

        y = x

        for block in range(1, self.n_blocks + 1):
            x, y = ResNetBlock1D(
                features=(block + 1) * in_channels,
                kernel_size=self.kernel_size,
                dropout_rate=self.dropout_rate,
                width=in_width // (4**block),
            )(x, y, train=train)

        x = x.reshape(x.shape[0], -1)

        x = nn.Dense(features=self.features)(x)

        return x


class SeparateSourceClassifierResNet1D(nn.Module):
    """
    This ResNet1D allows you to pass the source along with
    the input into the apply and call. Within
    this model, a new classification layer is built
    per source.

    Arguments
    ---------
    - features: int:
        The number of features in the output.

    - n_sources: int:
        The number of unique sources that this model should expect.

    - kernel_size: int:
        The size of the kernel to use in
        the convolutional and pooling layers.

    - dropout_rate: float:
        The dropout rate to use in the dropout layers.

    - n_blocks: int:
        The number of ResNetBlock1D blocks to use in
        the full ResNet1D model.

    """

    features: int
    n_sources: int
    kernel_size: int = 16
    dropout_rate: float = 0.2
    n_blocks: int = 4
    heavyside: bool = False
    bias_classifier: bool = True

    @nn.compact
    def __call__(self, x, source, train):

        x = nn.relu(
            ResNet1D(
                features=self.features,
                kernel_size=self.kernel_size,
                dropout_rate=self.dropout_rate,
                n_blocks=self.n_blocks,
            )(x, train)
        )
        x = SeparateSourceDense(
            self.features,
            n_sources=self.n_sources,
            heavyside=self.heavyside,
            bias=self.bias_classifier,
        )(x, source=source)

        return x


class SourceClassifierAppendResNet1D(nn.Module):
    """

    This ResNet1D model will append the source vector
    to the layer before the output. This
    gives the model a layer to learn the mapping
    between the predictions and the source
    dependent labels.

    In addition to the ResNet1D model,
    this will create an extra layer of shape
    (features+binary_dimensions, features).

    Arguments
    ---------
    - features: int:
        The number of features in the output.

    - binary_dimensions: int:
        The number of binary dimensions to use for the source vector.

    - kernel_size: int:
        The size of the kernel to use in
        the convolutional and pooling layers.

    - dropout_rate: float:
        The dropout rate to use in the dropout layers.

    - n_blocks: int:
        The number of ResNetBlock1D blocks to use in
        the full ResNet1D model.

    """

    features: int
    binary_dimensions: int
    kernel_size: int = 16
    dropout_rate: float = 0.2
    n_blocks: int = 4

    def int_array_to_bin_array(self, x) -> jnp.ndarray:
        """
        Turns a numpy array of integers into a binary array.
        For example: [1, 2, 3] -> [[0, 0, 1], [0, 1, 0], [0, 1, 1]]

        Arguments
        ---------
        - x: jnp.ndarray:
            The array of integers to be converted to binary.

        """
        return (((x[:, None] & (1 << jnp.arange(self.binary_dimensions)))) > 0).astype(
            int
        )

    @nn.compact
    def __call__(self, x, source, train):

        x = nn.relu(
            ResNet1D(
                features=self.features,
                kernel_size=self.kernel_size,
                dropout_rate=self.dropout_rate,
                n_blocks=self.n_blocks,
            )(x, train)
        )

        # source on classifier layer
        # this appends the source vector to the output of the
        # last layer of the features before using another dense layer
        # to predict the output
        out = nn.relu(
            nn.Dense(self.features)(
                jnp.concatenate([x, self.int_array_to_bin_array(source)], axis=-1)
            )
        )
        out = nn.Dense(self.features)(out)

        return out


class SourcePredictMappingResNet1D(nn.Module):
    """

    This ResNet1D will use the source value to predict a
    mapping between the output of the predictor
    and the output from a given source. It
    will then use this mapping to make a final prediction
    on the output.

    Arguments
    ---------
    - features: int:
        The number of features in the output.

    - binary_dimensions: int:
        The number of binary dimensions to use for the source vector.

    - kernel_size: int:
        The size of the kernel to use in
        the convolutional and pooling layers.

    - dropout_rate: float:
        The dropout rate to use in the dropout layers.

    - n_blocks: int:
        The number of ResNetBlock1D blocks to use in
        the full ResNet1D model.

    """

    features: int
    binary_dimensions: int
    kernel_size: int = 16
    dropout_rate: float = 0.2
    n_blocks: int = 4

    def setup(self):
        self.predictor = ResNet1D(
            features=self.features,
            kernel_size=self.kernel_size,
            dropout_rate=self.dropout_rate,
            n_blocks=self.n_blocks,
        )

        self.source_mapper = SourceMapper(
            features=self.features,
            binary_dimensions=self.binary_dimensions,
        )

    def __call__(self, x, source, train):

        # prediction model
        x = self.predictor(x, train)
        # source model
        s = self.source_mapper(source)
        return jax.vmap(jnp.matmul, in_axes=(0, 0))(x, s)


class SourceCalcMappingResNet1D(nn.Module):
    """

    This ResNet1D will use the source value to predict a
    mapping between the output of the predictor
    and the output from a given source. It
    will then use this mapping to make a final prediction
    on the output.

    Arguments
    ---------
    - features: int:
        The number of features in the output.

    - n_sources: int:
        The number of sources to expect.

    - kernel_size: int:
        The size of the kernel to use in
        the convolutional and pooling layers.

    - dropout_rate: float:
        The dropout rate to use in the dropout layers.

    - n_blocks: int:
        The number of ResNetBlock1D blocks to use in
        the full ResNet1D model.

    """

    features: int
    n_sources: int
    kernel_size: int = 16
    dropout_rate: float = 0.2
    n_blocks: int = 4

    def setup(self):

        self.predictor_model = ResNet1D(
            features=self.features,
            kernel_size=self.kernel_size,
            dropout_rate=self.dropout_rate,
            n_blocks=self.n_blocks,
        )

        self.source_mapper_kernel = self.variable(
            "source_mapper",
            "kernel",
            lambda ns, nf: jnp.stack([jnp.eye(nf) for _ in range(ns)]),
            self.n_sources,
            self.features,
        )

        return

    def predictor(self, x, train):
        # return nn.relu(self.predictor_model(x))
        return self.predictor_model(x, train)

    def source_map(self, x, source):
        return jax.lax.dot_general(
            x,
            self.source_mapper_kernel.value[source],
            (((x.ndim - 1,), (1,)), ((0), (0))),
        )

    def __call__(self, x, source, train):

        # prediction model
        x = self.predictor(x, train)

        # mapping sources
        x = self.source_map(x, source)
        return x
