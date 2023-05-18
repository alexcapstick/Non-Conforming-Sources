"""
This script will run the MNIST experiment with the given parameters.
"""
import os
import sys

path = os.path.join(os.getcwd(), "../")
sys.path.append(path)

import logging
import argparse
import itertools
import numpy as np
import jax
import optax
from flax.training import early_stopping
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from experiment_code.model_trainer import (
    TrainStateSourceMapping,
    TrainState,
    FitModel,
    _calc_loss_source_mapper,
    _do_step_source_mapper,
    _do_eval_source_mapper,
)
from experiment_code.model import (
    SourceAppendMLP,
    SeparateSourceClassifierMLP,
    SourceClassifierAppendMLP,
    SourcePredictMappingMLP,
    SourceCalcMappingMLP,
)
from experiment_code.model_evaluation import accuracy_score, source_accuracy_score
from experiment_code.utils import module_from_file, RandomState, p_map

logging.basicConfig(
    filename="/experiments.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


data_factory = module_from_file("data_factory", "../config/data.py")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

parser = argparse.ArgumentParser(
    prog="MNIST Experiment",
    description="This will run the MNIST experiment with the given parameters",
)

parser.add_argument(
    "--model-name", type=str, required=True, nargs="+", help="The model to test"
)
parser.add_argument(
    "--n-sources",
    type=int,
    nargs="+",
    default=[10],
    help="The number of sources to use",
)
parser.add_argument(
    "--batch-size", type=int, nargs="+", default=[512], help="The batch size to use"
)
parser.add_argument(
    "--seed", type=int, default=None, help="The seed to use to generate the new seeds"
)
parser.add_argument(
    "--n-jobs", type=int, default=1, help="The number of parallel jobs to run"
)
parser.add_argument(
    "--n-repeats",
    type=int,
    default=1,
    help="The number of repeats to run for each set of parameters",
)
parser.add_argument(
    "--learning-rate",
    type=float,
    nargs="+",
    default=[1e-3],
    help="The learning rate to use for the optimizer",
)
parser.add_argument(
    "--weight",
    type=float,
    nargs="+",
    default=[1.0],
    help="The weight to use when training CalcMapping models",
)
parser.add_argument(
    "--n-epochs",
    type=int,
    nargs="+",
    default=[100],
    help="The max number of epochs to train the model for. "
    "An early stopping on the validation loss will be used "
    "with a patience of 5 and tolerance of 1e-4",
)

args = parser.parse_args()

PARAMS = {
    "n_sources": args.n_sources,
    "batch_size": args.batch_size,
    "repeat": list(range(args.n_repeats)),
    "model": args.model_name,
    "learning_rate": args.learning_rate,
    "weight": args.weight,
    "n_epochs": args.n_epochs,
}
SEED = int(RandomState(args.seed).next()) if args.seed is None else args.seed

keys, values = zip(*PARAMS.items())
experiment_params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

train_dataset, test_dataset = data_factory.get_mnist(root="../../data/")


def run_experiment(train_dataset, test_dataset, experiment_params, seed):

    experiment_params["seed"] = seed

    model_dict = {
        "SourceAppendMLP": SourceAppendMLP(
            features=[100, 100, 10],
            binary_dimensions=int(
                np.maximum(np.ceil(np.log2(experiment_params["n_sources"])), 1.0)
            ),
        ),
        "SeparateSourceClassifierMLP": SeparateSourceClassifierMLP(
            features=[100, 100, 10], n_sources=experiment_params["n_sources"]
        ),
        "SeparateSourceBinaryClassifierMLP": SeparateSourceClassifierMLP(
            features=[100, 100, 10],
            n_sources=experiment_params["n_sources"],
            heavyside=True,
            bias_classifier=False,
        ),
        "SourceClassifierAppendMLP": SourceClassifierAppendMLP(
            features=[100, 100, 10],
            binary_dimensions=int(
                np.maximum(np.ceil(np.log2(experiment_params["n_sources"])), 1.0)
            ),
        ),
        "SourcePredictMappingMLP": SourcePredictMappingMLP(
            features=[100, 100, 10],
            binary_dimensions=int(
                np.maximum(np.ceil(np.log2(experiment_params["n_sources"])), 1.0)
            ),
        ),
        "SourceCalcMappingPerBatchMLP": SourceCalcMappingMLP(
            features=[100, 100, 10], n_sources=experiment_params["n_sources"]
        ),
    }

    current_datetime = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    experiment_name = (
        "mnist-"
        f"{''.join([l for l in experiment_params['model'] if l.isupper()])}"
        f"-{experiment_params['seed']}-{current_datetime}"
    )

    random_state = RandomState(experiment_params["seed"])
    key = jax.random.PRNGKey(random_state.next())

    writer = SummaryWriter(
        log_dir=f"../tb_runs/scripts/{experiment_name}",
    )

    results = dict(metrics=[], **experiment_params)

    train_dl, val_dl = data_factory.get_train_val_data(
        train_dataset,
        test_dataset,
        label_swap=True,
        seed=random_state.next(),
        n_sources=experiment_params["n_sources"],
        batch_size=experiment_params["batch_size"],
        batch_by_source=False,
    )

    model = model_dict[experiment_params["model"]]

    key, subkey = jax.random.split(key)
    batch = next(iter(train_dl))

    try:
        variables = model.init(subkey, x=batch[0], source=batch[1])
        if experiment_params["model"] in ["SourceCalcMappingPerBatchMLP"]:
            state = TrainStateSourceMapping.create(
                apply_fn=model.apply,
                predictor_fn=model.predictor,
                params=variables["params"],
                source_mapper=variables["source_mapper"],
                tx=optax.adam(learning_rate=experiment_params["learning_rate"]),
                weight=experiment_params["weight"],
            )
        else:
            state = TrainState.create(
                apply_fn=model.apply,
                params=variables["params"],
                tx=optax.adam(learning_rate=experiment_params["learning_rate"]),
            )

        early_stop = early_stopping.EarlyStopping(min_delta=1e-4, patience=5)

        if experiment_params["model"] in [
            "SourceCalcMappingPerBatchMLP",
        ]:
            fm = FitModel(
                loss_fn=_calc_loss_source_mapper,
                step_fn=_do_step_source_mapper,
                eval_fn=_do_eval_source_mapper,
            )
        else:
            fm = FitModel()

        state = fm.fit(
            state=state,
            train_dl=train_dl,
            n_epochs=experiment_params["n_epochs"],
            criterion=optax.softmax_cross_entropy_with_integer_labels,
            val_dl=val_dl,
            early_stop=early_stop,
            do_jit=True,
        )

        model_params = dict(
            params=state.params,
            **(
                {"source_mapper": state.source_mapper}
                if experiment_params["model"] in ["SourceCalcMappingPerBatchMLP"]
                else {}
            ),
        )

        accuracy = accuracy_score(
            model=model,
            params=model_params,
            dl=val_dl,
            verbose=False,
        )

        source_accuracy = source_accuracy_score(
            model=model,
            params=model_params,
            dl=val_dl,
            verbose=False,
        )

        writer.add_hparams(
            hparam_dict=experiment_params,
            metric_dict=dict(
                accuracy=accuracy,
                **{f"source_accuracy_{k}": v for k, v in source_accuracy.items()},
            ),
        )

        results["metrics"].append({"metric": "accuracy", "value": accuracy})
        results["metrics"].extend(
            [
                {"metric": f"source_accuracy_{k}", "value": v}
                for k, v in source_accuracy.items()
            ]
        )

        results = pd.json_normalize(
            results,
            record_path="metrics",
            meta=list(experiment_params.keys()),
        )

        results.to_pickle(f"../results/{experiment_name}.sav")
        writer.flush()
        writer.close()

    except Exception as e:
        logger.error(e)

    return


random_state = RandomState(SEED)
n_experiments = len(experiment_params_list)

print(f"Running {n_experiments} experiments")

p_map(run_experiment, n_jobs=args.n_jobs, backend="threading",)(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    list__experiment_params=experiment_params_list,
    list__seed=[
        int(random_state.next()) for _ in range(n_experiments)
    ],  # need python int for writer.add_hparams
)
