"""
This holds various functions that are useful for
this work.
"""

import importlib.util
import numpy as np
import joblib
from joblib import Parallel
import typing
import tqdm
from .progress import tqdm_style


def module_from_file(module_name: str, file_path: str):
    """
    Will open a module from a file path.

    Edited from https://stackoverflow.com/a/51585877/19451559.

    Examples
    ---------

    .. code-block::

        >>> model_trainer = module_from_file(
        ...     'model_trainer',
        ...     './some_code/model_trainer.py'
        ...     )
        >>> train_function = model_trainer.train
        <function model_trainer.train(...)>


    Arguments
    ---------

    - module_name: str:
        The name of the module to load.

    - file_path: str:
        File path to that module.



    Returns
    --------

    - out: module:
        A python module that can be
        used to access objects from
        within it.


    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RandomState(object):
    def __init__(self, random_state=None):
        """
        Class for holding and generating
        new random values.

        Arguments
        ---------

        - random_state: int:
            The random state to use as a seed.

        """
        self.random_state = random_state

    def next(self, n=1):
        """
        Get a new random value, and move the internal
        state forward.

        Arguments
        ---------

        - n: int:
            The number of random values to generate.

        """
        assert type(n) == int, "Ensure n is an integer"
        if n == 1:
            self.random_state, out_state = np.random.default_rng(
                self.random_state
            ).integers(0, 1e9, size=(2,))
        else:
            self.random_state, *out_state = np.random.default_rng(
                self.random_state
            ).integers(0, 1e9, size=(n + 1,))

        return out_state


class ProgressParallel(Parallel):
    def __init__(
        self,
        tqdm_bar: typing.Union[tqdm.tqdm, None] = None,
        verbose: bool = True,
        desc: str = "In Parallel",
        total: int = None,
        tqdm_style: typing.Dict[str, typing.Any] = tqdm_style,
        *args,
        **kwargs,
    ):
        """
        This is a wrapper for the joblib Parallel
        class that allows for a progress bar to be passed into
        the :code:`__init__` function so that the progress
        can be viewed.

        Recall that using :code:`backend='threading'`
        allows for shared access to variables!



        Examples
        ---------

        .. code-block::

            >>> pbar = tqdm.tqdm(total=5)
            >>> result = ProgressParallel(
            ...     tqdm_bar=pbar,
            ...     n_jobs=10,
            ...     )(
            ...         joblib.delayed(f_parallel)(i)
            ...         for i in range(5)
            ...     )

        Alternatively, you do not need to pass a :code:`tqdm` bar:

        .. code-block::

            >>> result = ProgressParallel(
            ...     n_jobs=10,
            ...     total=20,
            ...     desc='In Parallel',
            ...     )(
            ...         joblib.delayed(f_parallel)(i)
            ...         for i in range(20)
            ...     )


        Arguments
        ---------

        - tqdm_bar: typing.Union[tqdm.tqdm, None]:
            The tqdm bar that will be used in the
            progress updates.
            Every time progress is displayed,
            :code:`tqdm_bar.update(n)` will be called,
            where :code:`n` is the number of updates made.
            If :code:`None`, then a bar is created
            inside this class.
            Defaults to :code:`None`.

        - verbose: bool:
            If :code:`tqdm_bar=None`, then this
            argument allows the user to stop the
            progress bar from printing at all.
            Defaults to :code:`True`.

        - desc: str:
            If :code:`tqdm_bar=None`, then this
            argument allows the user to add
            a description to the progress bar.
            Defaults to :code:`'In Parallel'`.

        - total: str:
            If :code:`tqdm_bar=None`, then this
            argument allows the user to add
            a total to the progress bar, rather
            than let the bar automatically update it
            as it finds new tasks. If :code:`None`, then
            the total might update multiple times as the
            parallel process queues jobs.
            Defaults to :code:`None`.

        - tqdm_style: typing.Dict[str,typing.Any]:
            A dictionary passed to the tqdm object
            which can be used to pass kwargs.
            :code:`desc`, :code:`total`, and  :code:`disable`
            (verbose) cannot be passed here. Please
            use the arguments above.
            Defaults to :code:`aml_tqdm_style` (see :code:`aml.tqdm_style`).


        """

        super().__init__(verbose=False, *args, **kwargs)

        if tqdm_bar is None:
            self.tqdm_bar = tqdm.tqdm(
                desc=desc,
                total=total,
                disable=not verbose,
                smoothing=0,
                **tqdm_style,
            )
            self.total = total
            self.bar_this_instance = True
        else:
            self.tqdm_bar = tqdm_bar
            self.bar_this_instance = False
        self.previously_completed = 0
        self._verbose = verbose

    def __call__(self, *args, **kwargs):
        return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if not self._verbose:
            return
        if self.bar_this_instance:
            # Original job iterator becomes None once it has been fully
            # consumed : at this point we know the total number of jobs and we are
            # able to display an estimation of the remaining time based on already
            # completed jobs. Otherwise, we simply display the number of completed
            # tasks.
            if self.total is None:
                if self._original_iterator is None:
                    # We are finished dispatching
                    if self.n_jobs == 1:
                        self.tqdm_bar.total = None
                        self.total = None
                    else:
                        self.tqdm_bar.total = self.n_dispatched_tasks
                        self.total = self.n_dispatched_tasks

        difference = self.n_completed_tasks - self.previously_completed
        self.tqdm_bar.update(difference)
        self.tqdm_bar.refresh()
        self.previously_completed += difference

        if self.bar_this_instance:
            if self.previously_completed == self.total:
                self.tqdm_bar.close()

        return


def _p_apply_construct_inputs(
    **kwargs,
) -> typing.Tuple[
    typing.List[typing.Dict[str, typing.Any]], typing.Dict[str, typing.Any]
]:
    """
    This function allows you to pass keyword
    arguments that are prefixed with :code:`list__`
    and not, and return a list of dictionaries that
    can be used as keyword arguments in :code:`p_apply`.

    The example explains this better.



    Examples
    ---------

    .. code-block::

        >>> list_kwargs, reused_kwargs = _p_apply_construct_inputs(
        ...     x=[1,2,3,4,5,6],
        ...     list__y=[[1,2,3], [1,2], [1,]],
        ...     )
        >>> list_kwargs
        [{'y': [1, 2, 3]}, {'y': [1, 2]}, {'y': [1]}]
        >>> reused_kwargs
        {'x': [1, 2, 3, 4, 5, 6]}


    Raises
    ---------

        TypeError: If the :code:`list__` arguments are
        of different lengths.

    Returns
    --------

    - out: typing.Tuple[typing.List[typing.Dict[str,typing.Any]], typing.Dict[str,typing.Any]]:
        A tuple with a list of dictionaries and a dictionary.


    """

    list_kwargs = {}
    reused_kwargs = {}

    list_len = None
    for key, value in kwargs.items():
        if "list__" in key:
            list_kwargs[key.replace("list__", "")] = value
            if list_len is None:
                list_len = len(value)
            if len(value) != list_len:
                raise TypeError(
                    "Ensure all of the list__ prefixed "
                    "arguments have the same length."
                )
        else:
            reused_kwargs[key] = value

    list_kwargs = [
        dict(zip(list_kwargs.keys(), values)) for values in zip(*list_kwargs.values())
    ]

    if len(list_kwargs) == 0:
        list_kwargs = [{}]

    return list_kwargs, reused_kwargs


def p_map(
    func: typing.Callable,
    n_jobs: int = 1,
    backend: str = "threading",
    verbose: bool = True,
) -> typing.List[typing.Any]:
    """
    This class allows you to parallelise any function
    over some inputs.

    You may use the prefix :code:`list__` to any
    argument for each element to be parallelised,
    and not prefix an argument for it to be
    consistent between parallel computations.

    This is more easily seen through example.

    Note this function is primarily used
    for running general functions in parallel,
    and there are faster functions for
    running jax calculations in parallel!


    Examples
    ---------

    In the following example, the function is
    parallelised over the :code:`y` argument, since
    this is prefixed with :code:`list__`. This
    means that :code:`x` is added to each of the
    :code:`y`s.

    .. code-block::

        >>> p_map(
        ...     lambda x,y: x+y,
        ... )(
        ...     x=np.array([0,1,2]),
        ...     list__y=np.array([0,1,2]),
        ... )
        Parallel function: 3it [00:00, 2000.78it/s]
        [array([0, 1, 2]), array([1, 2, 3]), array([2, 3, 4])]

    In the next example, the function is
    parallelised over none of the arguments, since
    none are prefixed with :code:`list__`.
    This means that :code:`x` is added to :code:`y`.

    .. code-block::

        >>> p_map(
        ...     lambda x,y: x+y,
        ... )(
        ...     x=np.array([0,1,2]),
        ...     y=np.array([0,1,2]),
        ... )
        Parallel function: 1it [00:00, ?it/s]
        [array([0, 2, 4])]


    Arguments
    ---------

    - func: typing.Callable:
        The function to be used.

    - n_jobs: int, optional:
        The number of jobs to run in parallel. Be mindful that
        the functions and related computations might be expensive
        for the CPU, GPU, or RAM.
        Defaults to :code:`1`.

    - backend: str, optional:
        The backend used for the parallel compute. This should
        be an acceptable value for :code:`joblib.Parallel`. Note
        that this function uses :code:`threading` by default,
        which means that values given to this function
        will be shared in memory between parallel runs.
        Be mindful of this when passing objects like dictionaries
        to the function being parallelised.
        Defaults to :code:`threading`.

    - verbose: bool, optional:
        Whether to print progress.
        Defaults to :code:`True`.

    Returns
    ---------
    - out: typing.Callable:
        A parallel version of the function given.

    """

    def parallel_func(**kwargs):

        list_kwargs, reused_kwargs = _p_apply_construct_inputs(**kwargs)

        return ProgressParallel(
            n_jobs=n_jobs,
            backend=backend,
            verbose=verbose,
            desc=f"Parallel {func.__name__}",
        )(
            joblib.delayed(func)(
                **lk,
                **reused_kwargs,
            )
            for lk in list_kwargs
        )

    return parallel_func
