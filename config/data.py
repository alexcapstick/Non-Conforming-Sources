"""
This code provides functions for loading the datasets.
"""

import os
import sys

path = os.path.join(os.getcwd(), "../")
sys.path.append(path)

import typing
import torch
import torchvision
import numpy as np
import ast
import joblib
import tqdm
import imblearn
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision.datasets.utils import download_and_extract_archive

from experiment_code.progress import tqdm_style
from experiment_code.utils import RandomState

try:
    import wfdb

    wfdb_import_error = False
except ImportError:
    wfdb_import_error = True


## datasets


class HelperDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        """
        This dataset helps you to build wrappers for
        other datasets by ensuring that any method or attribute
        of the original dataset is available as a method
        or attribute of the new dataset.

        The original dataset is available as the attribute
        :code:`._dataset`.

        Arguments
        ---------

        - dataset: torch.utils.data.Dataset:
            The dataset that will be wrapped.

        """

        self._dataset = dataset
        return

    # defined since __getattr__ causes pickling problems
    def __getstate__(self):
        return vars(self)

    # defined since __getattr__ causes pickling problems
    def __setstate__(self, state):
        vars(self).update(state)

    def __getattr__(self, name):
        if hasattr(self._dataset, name):
            return getattr(self._dataset, name)
        else:
            raise AttributeError


class WrapperDataset(HelperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        functions_index: typing.Union[typing.List[int], int, None] = None,
        functions: typing.Union[
            typing.Callable, typing.List[typing.Callable]
        ] = lambda x: x,
    ):
        """
        This allows you to wrap a dataset with a set of 
        functions that will be applied to each returned 
        data point. You can apply a single function to all 
        outputs of a data point, or a different function
        to each of the different outputs.
        
        
        
        Examples
        ---------

        The following would multiply all of the first returned
        values in the dataset by 2.
        
        .. code-block::
        
            >>> WrapperDataset(
            ...     dataset
            ...     functions_index=0,
            ...     functions=lambda x: x*2
            ...     )

        The following would multiply all of the returned
        values in the dataset by 2.
        
        .. code-block::
        
            >>> WrapperDataset(
            ...     dataset
            ...     functions_index=None,
            ...     functions=lambda x: x*2
            ...     )

        The following would multiply all of the first returned
        values in the dataset by 2, and the second by 2.
        
        .. code-block::
        
            >>> WrapperDataset(
            ...     dataset
            ...     functions_index=[0, 1],
            ...     functions=lambda x: x*2
            ...     )

        The following would multiply all of the first returned
        values in the dataset by 2, and the second by 3.
        
        .. code-block::
        
            >>> WrapperDataset(
            ...     dataset
            ...     functions_index=[0, 1],
            ...     functions=[lambda x: x*2, lambda x: x*3]
            ...     )
        
        
        Arguments
        ---------
        
        - dataset: torch.utils.data.Dataset: 
            The dataset to be wrapped.
        
        - functions_index: typing.Union[typing.List[int], int, None], optional:
            The index of the functions to be applied to. 

            - If :code:`None`, then if the :code:`functions` is callable, it \
            will be applied to all outputs of the data points, \
            or if the :code:`functions` is a list, it will be applied to the corresponding \
            output of the data point.

            - If :code:`list` then the corresponding index will have the \
            :code:`functions` applied to them. If :code:`functions` is a list, \
            then it will be applied to the corresponding indicies given in :code:`functions_index` \
            of the data point. If :code:`functions` is callable, it will be applied to all of the \
            indicies in :code:`functions_index`
        
            - If :code:`int`, then the :code:`functions` must be callable, and \
            will be applied to the output of this index.

            - If :code:`'all'`, then the :code:`functions` must be callable, and \
            will be applied to the output of the dataset. This allows you \
            to build a function that can act over all of the outputs of dataset. \
            The returned value will be the data that is returned by the dataset.
            
            Defaults to :code:`None`.
        
        - functions: _type_, optional:
            This is the function, or list of functions to apply to the
            corresponding indices in :code:`functions_index`. Please
            see the documentation for the :code:`functions_index` argument
            to understand the behaviour of different input types. 
            Defaults to :code:`lambda x:x`.
        
        
        """

        super(WrapperDataset, self).__init__(dataset=dataset)

        self.apply_all = False
        if functions_index is None:
            if type(functions) == list:
                self.functions = {fi: f for fi, f in enumerate(functions)}
            elif callable(functions):
                self.functions = functions
            else:
                raise TypeError(
                    "If functions_index=None, please ensure "
                    "that functions is a list or a callable object."
                )

        elif type(functions_index) == list:
            if type(functions) == list:
                assert len(functions_index) == len(
                    functions
                ), "Please ensure that the functions_index is the same length as functions."
                self.functions = {fi: f for fi, f in zip(functions_index, functions)}
            elif callable(functions):
                self.functions = {fi: functions for fi in functions_index}
            else:
                raise TypeError(
                    "If type(functions_index)==list, please ensure "
                    "that functions is a list of the same length or a callable object."
                )

        elif type(functions_index) == int:
            if callable(functions):
                self.functions = {functions_index: functions}
            else:
                raise TypeError(
                    "If type(functions_index)==int, please ensure "
                    "the functions is a callable object."
                )

        elif type(functions_index) == str:
            if functions_index == "all":
                if callable(functions):
                    self.functions = functions
                    self.apply_all = True
                else:
                    raise TypeError(
                        "Please ensure that functions is callable if functions_index == 'all'."
                    )
            else:
                raise TypeError(
                    f"{functions_index} is an invalid option for functions_index."
                )

        else:
            raise TypeError(
                "Please ensure that functions_index is a list, int or None."
            )

        return

    def __getitem__(self, index):
        if type(self.functions) == dict:
            return [
                self.functions.get(nout, lambda x: x)(out)
                for nout, out in enumerate(self._dataset[index])
            ]
        elif callable(self.functions):
            if self.apply_all:
                return self.functions(*self._dataset[index])
            else:
                return [self.functions(out) for out in self._dataset[index]]
        else:
            raise TypeError("The functions could not be applied.")

    def __len__(self):
        return len(self._dataset)


class MemoryDataset(HelperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        now: bool = True,
        verbose: bool = True,
        n_jobs: int = 1,
    ):
        """
        This dataset allows the user
        to wrap another dataset and
        load all of the outputs into memory,
        so that they are accessed from RAM
        instead of storage. All attributes of
        the original dataset will still be available, except
        for :code:`._dataset` and :code:`._data_dict` if they
        were defined.
        It also allows the data to be saved in memory right
        away or after the data is accessed for the first time.


        Examples
        ---------

        .. code-block::

            >>> dataset = MemoryDataset(dataset, now=True)


        Arguments
        ---------

        - dataset: torch.utils.data.Dataset:
            The dataset to wrap and add to memory.

        - now: bool, optional:
            Whether to save the data to memory
            right away, or the first time the
            data is accessed. If :code:`True`, then
            this initialisation might take some time
            as it will need to load all of the data.
            Defaults to :code:`True`.

        - verbose: bool, optional:
            Whether to print progress
            as the data is being loaded into
            memory. This is ignored if :code:`now=False`.
            Defaults to :code:`True`.

        - n_jobs: int, optional:
            The number of parallel operations when loading
            the data to memory.
            Defaults to :code:`1`.


        """

        super(MemoryDataset, self).__init__(dataset=dataset)

        self._data_dict = {}
        if now:

            pbar = tqdm.tqdm(
                total=len(self._dataset),
                desc="Loading into memory",
                disable=not verbose,
                smoothing=0,
                **tqdm_style,
            )

            def add_to_dict(index):
                for ni, i in enumerate(index):
                    self._data_dict[i] = self._dataset[i]
                    pbar.update(1)
                    pbar.refresh()
                return None

            all_index = np.arange(len(self._dataset))
            index_list = [all_index[i::n_jobs] for i in range(n_jobs)]

            joblib.Parallel(
                n_jobs=n_jobs,
                backend="threading",
            )(joblib.delayed(add_to_dict)(index) for index in index_list)

            pbar.close()

        return

    def __getitem__(self, index):

        if index in self._data_dict:
            return self._data_dict[index]
        else:
            output = self._dataset[index]
            self._data_dict[index] = output
            return output

    def __len__(self):
        return len(self._dataset)

    # defined since __getattr__ causes pickling problems
    def __getstate__(self):
        return vars(self)

    # defined since __getattr__ causes pickling problems
    def __setstate__(self, state):
        vars(self).update(state)

    def __getattr__(self, name):
        if hasattr(self._dataset, name):
            return getattr(self._dataset, name)
        else:
            raise AttributeError


class PTB_XL(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str = "./",
        train: bool = True,
        sampling_rate: typing.Literal[100, 500] = 100,
        binary: bool = False,
        subset=False,
    ):
        """
        ECG Data, as described here: https://physionet.org/content/ptb-xl/1.0.3/.
        A positive class when :code:`binary=True`, indicates that
        the ECG Data is abnormal.

        Examples
        ---------

        .. code-block::

            >>> dataset = PTB_XL(
            ...     root='../../data/',
            ...     train=True,
            ...     sampling_rate=500,
            ...     )



        Arguments
        ---------

        - root: str, optional:
            The path that the data is saved
            or will be saved.
            Defaults to :code:`'./'`.

        - train: bool, optional:
            Whether to load the training or testing set.
            If :code:`True`, the training set will be loaded,
            and if :code:`False`, the testing set will be loaded.
            Defaults to :code:`True`.

        - sampling_rate: typing.Literal[100, 500], optional:
            The sampling rate. This should be
            in :code:`[100, 500]`. This is
            the sampling rate of the data, in Hz.
            Defaults to :code:`100`.

        - binary: bool, optional:
            Whether to return classes based on whether the
            ecg is normal or not, a binary classification
            problem. If :code:`False`, the classes are
            :code:`["NORM", "CD", "HYP", "MI", "STTC"]` as described
            on the linked website.
            Defaults to :code:`False`.

        - subset: bool, optional:
            If :code:`True`, only the first 1000 items
            of the training and test set will be returned.
            This is useful for debugging, as the whole
            dataset is large.
            Defaults to :code:`False`.


        """

        if wfdb_import_error:
            raise ImportError(
                "Please install wfdb before using this dataset. Use pip install wfdb."
            )

        assert sampling_rate in [
            100,
            500,
        ], "Please choose sampling_rate from [100, 500]"
        assert type(train) == bool, "Please use train = True or False"

        self.root = root
        self.download()
        self.root = os.path.join(
            self.root,
            "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/",
        )

        self.train = train
        self.sampling_rate = sampling_rate
        self.binary = binary
        self.meta_data = pd.read_csv(self.root + "ptbxl_database.csv")
        self.meta_data["scp_codes"] = self.meta_data["scp_codes"].apply(
            lambda x: ast.literal_eval(x)
        )
        self.aggregate_diagnostic()  # create diagnostic columns
        self.feature_names = [
            "I",
            "II",
            "III",
            "aVL",
            "aVR",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ]

        if self.train:
            self.meta_data = self.meta_data.query("strat_fold != 10")
            if subset:
                self.meta_data = self.meta_data.iloc[:1000]
        else:
            self.meta_data = self.meta_data.query("strat_fold == 10")
            if subset:
                self.meta_data = self.meta_data.iloc[:1000]

        self.targets = torch.tensor(
            self.meta_data[["NORM", "CD", "HYP", "MI", "STTC"]].values.astype(np.int64)
        )
        if binary:
            self.targets = 1 - self.targets[:, 0]

        return

    def _check_exists(self):
        folder = os.path.join(
            self.root,
            "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
        )
        return os.path.exists(folder)

    def download(self):

        if self._check_exists():
            print("Files already downloaded.")
            return

        download_and_extract_archive(
            url="https://physionet.org/static"
            "/published-projects/ptb-xl/"
            "ptb-xl-a-large-publicly-available"
            "-electrocardiography-dataset-1.0.3.zip",
            download_root=self.root,
            extract_root=self.root,
            filename="ptbxl.zip",
            remove_finished=True,
        )

        return

    @staticmethod
    def single_diagnostic(y_dict, agg_df):
        tmp = []
        for key in y_dict.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    def aggregate_diagnostic(self):
        agg_df = pd.read_csv(self.root + "scp_statements.csv", index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
        self.meta_data["diagnostic_superclass"] = self.meta_data["scp_codes"].apply(
            self.single_diagnostic,
            agg_df=agg_df,
        )
        mlb = MultiLabelBinarizer()
        self.meta_data = self.meta_data.join(
            pd.DataFrame(
                mlb.fit_transform(self.meta_data.pop("diagnostic_superclass")),
                columns=mlb.classes_,
                index=self.meta_data.index,
            )
        )
        return

    def __getitem__(self, index):

        data = self.meta_data.iloc[index]

        if self.sampling_rate == 100:
            f = data["filename_lr"]
            x = wfdb.rdsamp(self.root + f)
        elif self.sampling_rate == 500:
            f = data["filename_hr"]
            x = wfdb.rdsamp(self.root + f)
        x = torch.tensor(x[0]).float()
        y = torch.tensor(
            data[["NORM", "CD", "HYP", "MI", "STTC"]].values.astype(np.int64)
        )
        if self.binary:
            y = (1 - y[0]).item()

        return x, y

    def __len__(self):
        return len(self.meta_data)


## dataloaders


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch).cpu().numpy()
        # return np.stack([x.cpu().numpy() for x in batch])
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NumpyLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        A wrapper for the torch dataloader that
        supplies numpy arrays instead of tensors.

        It takes all of the same arguments as
        :code:`torch.utils.data.DataLoader` and
        has the same default arguments:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

        The :code:`collate_fn` function is changed
        to allow for numpy arrays.

        The code was inspired by:
        https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html

        """
        super(self.__class__, self).__init__(
            *args,
            collate_fn=numpy_collate,
            **kwargs,
        )


## samplers


class GroupBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        group: typing.Union[np.ndarray, typing.List[typing.Any]],
        seed: typing.Union[None, int] = None,
        batch_size: int = 20,
        upsample: typing.Union[bool, typing.Dict[typing.Any, int]] = False,
    ):
        """
        A pytorch batch sampler that returns a batch of samples with
        that same group. This means each batch will be drawn from
        only a single group.

        Examples
        ---------

        The following will batch the training dataset
        into batches that contains single group, given
        by the :code:`group` argument

        .. code-block::

            >>> dl = torch.utils.data.DataLoader(
            ...     train_dataset,
            ...     batch_sampler=GroupBatchSampler(
            ...         group=train_group,
            ...         seed=seed,
            ...         batch_size=64,
            ...         )
            ...     )


        Arguments
        ---------

        - group: typing.Union[np.ndarray, typing.List[typing.Any]]:
            The group of the data points. This should be
            the same length as the data set that is to be
            sampled.

        - seed: int (optional):
            Random seed for group order shuffling and
            shuffling of points in each batch.
            Defaults to :code:`None`.

        - batch_size: int, (optional):
            The size of each batch. Each batch
            will be smaller than or equal in
            size to this value.
            Defaults to :code:`20`.

        - upsample: typing.Union[bool, typing.Dict[typing.Any, int]], (optional):
            Whether to upsample the smaller groups,
            so that all groups have the same size.
            Defaults to :code:`False`.


        """

        rng = np.random.default_rng(seed)

        group = np.asarray(group)

        upsample_bool = upsample if type(upsample) == bool else True

        if upsample_bool:
            upsample_idx, group = imblearn.over_sampling.RandomOverSampler(
                sampling_strategy="all" if type(upsample) == bool else upsample,
                random_state=rng.integers(1e9),
            ).fit_resample(np.arange(len(group)).reshape(-1, 1), group)
            upsample_idx = upsample_idx.reshape(-1)

        group_unique, group_counts = np.unique(group, return_counts=True)
        group_batches = (
            np.repeat(
                np.ceil(np.max(group_counts) / batch_size).astype(int),
                len(group_unique),
            )
            if upsample
            else np.ceil(group_counts / batch_size).astype(int)
        )
        rng = np.random.default_rng(rng.integers(low=0, high=1e9, size=(4,)))
        n_batches = np.sum(group_batches)
        self.out = -1 * np.ones((n_batches, batch_size))
        group_order = rng.permutation(np.repeat(group_unique, group_batches))

        for g in group_unique:
            # get the index of the items from that group
            group_idx = np.argwhere(group == g).reshape(-1)
            # shuffle the group index
            rng.shuffle(group_idx)
            # get the section of the output that we will edit
            out_temp = self.out[group_order == g].reshape(-1)
            # replace the values with the index of the items
            out_temp[: len(group_idx)] = (
                upsample_idx[group_idx] if upsample else group_idx
            )
            out_temp = out_temp.reshape(-1, batch_size)
            rng.shuffle(out_temp, axis=0)
            self.out[group_order == g] = out_temp
            rng = np.random.default_rng(rng.integers(low=0, high=1e9, size=(3,)))

        self.out = [list(batch[batch != -1].astype(int)) for batch in self.out]

        return

    def __iter__(self):
        return iter(self.out)

    def __len__(self):
        return len(self.out)


## helpers


class FlattenImage(torch.nn.Module):
    def __init__(self):
        """
        Allows you to flatten an input to
        1D. This is useful in pytorch
        transforms when loading data.

        """
        super(FlattenImage, self).__init__()

    def forward(self, x):
        return x.reshape(-1)


class SourceDataset(HelperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        sources: typing.Union[None, typing.List[typing.Any]] = None,
    ):
        """
        A dataset wrapper that allows you to append
        the sources to another dataset.

        Examples
        ---------

        The following is an example of how you
        might add sources to a training dataset:

        >>> train_data = SourceDataset(train_data, sources=torch.arange(10))
        >>> train_data[0]
        ..., ..., 0
        >>> train_data[9]
        ..., ..., 9

        Arguments
        ---------

        - dataset: torch.utils.data.Dataset:
            The dataset to have sources added to the indexed
            outputs.

        - sources: typing.Union[None, typing.List[typing.Any]]:
            The sources that will be added to the indexed
            output. This should be the same length as the
            dataset. This can be a list of objects.

        """

        assert len(dataset) == len(
            sources
        ), "Please ensure that the dataset and sources are of the same length."
        super(SourceDataset, self).__init__(dataset=dataset)
        self.sources = sources

        return

    def __getitem__(self, idx):
        values = self._dataset[idx]
        out = [v for v in values]
        out.append(self.sources[idx])
        return out

    def __len__(
        self,
    ):
        return len(self._dataset)


## functions for loading dataset


def get_mnist(
    root: str,
    n_jobs: int = 1,
) -> typing.Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    A function to load the MNIST dataset.

    Arugments
    ---------
    - root: str:
        The file path to save or load the dataset.

    Returns
    ---------
    - train_dataset: torch.utils.data.Dataset:
        The training dataset.

    - test_dataset: torch.utils.data.Dataset:
        The testing dataset.

    """

    dataset_args = dict(
        root=root,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5), (0.5)),
                FlattenImage(),
            ]
        ),
    )

    train_dataset = MemoryDataset(
        torchvision.datasets.MNIST(train=True, **dataset_args),
        now=True,
        n_jobs=n_jobs,
    )

    test_dataset = MemoryDataset(
        torchvision.datasets.MNIST(train=False, **dataset_args),
        now=True,
        n_jobs=n_jobs,
    )

    return train_dataset, test_dataset


def get_ptbxl(
    root: str, n_jobs: int = 1, subset: bool = False
) -> typing.Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    A function to load the MNIST dataset.

    Arugments
    ---------
    - root: str:
        The file path to save or load the dataset.

    - subset: bool:
        Whether to only load the first 1000 samples
        of the training and testing set.

    Returns
    ---------
    - train_dataset: torch.utils.data.Dataset:
        The training dataset.

    - test_dataset: torch.utils.data.Dataset:
        The testing dataset.

    """

    dataset_args = dict(
        root=root,
        sampling_rate=100,
        binary=True,
        subset=subset,
    )

    train_dataset = MemoryDataset(
        PTB_XL(train=True, **dataset_args),
        now=True,
        n_jobs=n_jobs,
    )

    test_dataset = MemoryDataset(
        PTB_XL(train=False, **dataset_args),
        now=True,
        n_jobs=n_jobs,
    )

    return train_dataset, test_dataset


## get training or testing data


def get_train_val_data(
    train_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    seed: typing.Union[int, None] = None,
    batch_size: int = 128,
    label_swap: bool = True,
    n_sources: int = 10,
    batch_by_source: bool = True,
    label_replace: bool = True,
):

    random_state = RandomState(seed)

    # producing random sources
    sources = torch.randint(
        low=0,
        high=n_sources,
        size=(len(train_dataset),),
        generator=torch.Generator().manual_seed(int(random_state.next())),
    ).tolist()

    # dataset containing sources too
    train_dataset = SourceDataset(dataset=train_dataset, sources=sources)

    # target map
    target_map_dict = {}
    target_unique = torch.unique(train_dataset.targets).numpy()
    for source in range(n_sources):
        if label_replace:
            target_map_dict[source] = {
                target: np.random.default_rng(seed=random_state.next()).choice(
                    target_unique,
                )
                for target in target_unique
            }
        else:
            target_map_dict[source] = {
                target: new_target
                for target, new_target in zip(
                    target_unique,
                    np.random.default_rng(seed=random_state.next()).permutation(
                        target_unique,
                    ),
                )
            }

    def target_map(x, y, source):
        return x, source, target_map_dict[source][y]

    # swapping the labels for the different sources
    if label_swap:
        # dataset with target map
        train_dataset = WrapperDataset(
            train_dataset, functions_index="all", functions=target_map
        )

    # train-val splitting
    lengths = [
        int(0.75 * len(train_dataset)),
        len(train_dataset) - int(0.75 * len(train_dataset)),
    ]
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        lengths=lengths,
        generator=torch.Generator().manual_seed(int(random_state.next())),
    )

    if batch_by_source:
        # train and val batch samplers
        train_batch_sampler = GroupBatchSampler(
            group=[s for _, _, s in train_dataset],
            seed=random_state.next(),
            batch_size=batch_size,
        )
        val_batch_sampler = GroupBatchSampler(
            group=[s for _, _, s in val_dataset],
            seed=random_state.next(),
            batch_size=batch_size,
        )
    else:
        # train and val batch samplers
        train_batch_sampler = torch.utils.data.BatchSampler(
            sampler=torch.utils.data.RandomSampler(
                train_dataset,
                generator=torch.Generator().manual_seed(int(random_state.next())),
            ),
            batch_size=batch_size,
            drop_last=False,
        )
        val_batch_sampler = torch.utils.data.BatchSampler(
            sampler=torch.utils.data.RandomSampler(
                val_dataset,
                generator=torch.Generator().manual_seed(int(random_state.next())),
            ),
            batch_size=batch_size,
            drop_last=False,
        )

    # train and val data loaders
    train_dl = NumpyLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
    )
    val_dl = NumpyLoader(
        dataset=val_dataset,
        batch_sampler=val_batch_sampler,
    )

    return train_dl, val_dl
