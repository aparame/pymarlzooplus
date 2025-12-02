import multiprocessing
import subprocess
from itertools import product
import yaml
import random

import click

import os
from pymarlzooplus.utils.plot_utils import plot_multiple_experiment_results


_CPU_COUNT = multiprocessing.cpu_count() - 1
MAIN_PY_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "main.py")

def _flatten_lists(_object):
    for item in _object:
        if isinstance(item, (list, tuple, set)):
            yield from _flatten_lists(item)
        else:
            yield item


def _compute_combinations(config, shuffle, seeds):
    combinations = []
    for k, v in config["grid-search"].items():
        if type(v) is not list:
            v = [v]
        combinations.append([f"{k}={v_i}" for v_i in v])

    group_comb = []
    for _, v in config["grid-search-groups"].items():
        d = {}
        for d_i in v:
            d.update(d_i)

        group_comb.append(tuple([f"{k}={v_i}" for k, v_i in d.items()]))
    combinations.append(group_comb)

    click.echo("Found following combinations: ")
    click.echo(click.style(" X ", fg="red", bold=True).join([str(s) for s in combinations]))

    configs = list(product(*combinations))
    configs = [list(_flatten_lists(c)) for c in configs]

    configs = [[f"hypergroup=hp_grp_{i}"] + c for i, c in enumerate(configs)]

    configs = list(product(configs, [f"seed={i}" for i in range(seeds)]))
    configs = [list(_flatten_lists(c)) for c in configs]

    if shuffle:
        random.Random(1337).shuffle(configs)

    return configs


def work(cmd):
    cmd = cmd.split(" ")
    return subprocess.call(cmd, shell=False)


@click.group()
def cli():
    pass


@cli.group()
@click.option("--config", type=click.File(), default="config.yaml")
@click.option("--shuffle/--no-shuffle", default=True)
@click.option("--yes", is_flag=True, default=False, help="Skip confirmation prompt")
@click.option("--seeds", default=3, show_default=True, help="How many seeds to run")
@click.option("--plot/--no-plot", default=True, help="Whether to plot results after running")
@click.pass_context
def run(ctx, config, shuffle, seeds, plot, yes):
    config = yaml.load(config, Loader=yaml.FullLoader)
    combos = _compute_combinations(config, shuffle, seeds)
    if len(combos) == 0:
        click.echo("No valid combinations. Aborted!")
        exit(1)
    ctx.obj = (combos, config, plot, yes)


@run.command()
@click.option(
    "--cpus",
    default=_CPU_COUNT,
    show_default=True,
    help="How many processes to run in parallel",
)
@click.pass_obj
def locally(obj, cpus):
    (combos, config, plot, yes) = obj
    configs = [f"python {MAIN_PY_PATH} " + " ".join([c for c in combo if c.startswith("--")]) + " with " + " ".join(
        [c for c in combo if not c.startswith("--")]) for combo in combos]
    if not yes:
        click.confirm(
            f"There are {click.style(str(len(combos)), fg='red')} combinations of configurations "
            "(taking into account the number of seeds). "
            f"Up to {cpus} will run in parallel. "
            "Continue?",
            abort=True,
        )
    pool = multiprocessing.Pool(processes=cpus)
    print(pool.map(work, configs))


    if plot and "plotting" in config:
        print("\nHyperparameter search finished. Triggering plots...")
        try:
            plot_config = config["plotting"]

            base_sacred_path = os.path.expanduser(plot_config["base_sacred_path"])
            base_save_path = os.path.expanduser(plot_config["base_save_path"])
            env_name = plot_config["env_name"]

            if "--config" not in config.get("grid-search", {}):
                print(
                    "\nError: 'grid-search' block must contain '--config' list to auto-generate plots. Skipping plotting.")
                return

            algo_configs = config["grid-search"]["--config"]

            legend_map = plot_config.get("algorithm_legend_map", {})

            algo_names = []
            paths_to_results = []
            for algo_config in algo_configs:
                legend_name = legend_map.get(algo_config, algo_config.upper())
                algo_names.append(legend_name)
                path = os.path.join(base_sacred_path, algo_config, env_name)
                os.makedirs(path, exist_ok=True)
                paths_to_results.append(path)

            path_to_save = os.path.join(base_save_path, env_name)

            plot_train = plot_config.get("plot_train", False)
            plot_legend_bool = plot_config.get("plot_legend_bool", False)

            os.makedirs(path_to_save, exist_ok=True)

            plot_multiple_experiment_results(
                paths_to_results=paths_to_results,
                algo_names=algo_names,
                env_name=env_name,
                path_to_save=path_to_save,
                plot_train=plot_train,
                plot_legend_bool=plot_legend_bool
            )
            print(f"\nSuccessfully generated plots and saved to: {path_to_save}")

        except KeyError as e:
            print(
                f"\nError in 'plotting' config: Missing key {e}. Please ensure 'base_sacred_path', 'base_save_path', and 'env_name' are set.")
        except Exception as e:
            print(f"\nAn error occurred during plotting: {e}. Skipping plotting.")
    else:
        print("\nHyperparameter search finished. No 'plotting' section in config, skipping plots.")


@run.command()
@click.argument(
    "index", type=int,
)
@click.pass_obj
def single(combos, index):
    """Runs a single hyperparameter combination
    INDEX is the index of the combination to run in the generated combination list
    """

    config = combos[index]
    cmd = f"python {MAIN_PY_PATH} " + " ".join([c for c in config if c.startswith("--")]) + " with " + " ".join([c for c in config if not c.startswith("--")])
    print(cmd)
    work(cmd)


if __name__ == "__main__":
    cli()
