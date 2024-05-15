"""
This is the CLI script that is executed when the user runs the `fusion-bench` command.
The script is responsible for parsing the command-line arguments, loading the configuration file, and running the fusion algorithm.
"""

import importlib
import importlib.resources
import os

import hydra
from omegaconf import DictConfig, OmegaConf
from rich import print as rich_print
from rich.syntax import Syntax

from ..method import load_algorithm
from ..modelpool import load_modelpool
from ..taskpool import load_taskpool


def run_model_fusion(cfg: DictConfig):
    """
    Run the model fusion process based on the provided configuration.

    1. This function loads a model pool and an model fusion algorithm based on the configuration.
    2. It then uses the algorithm to fuse the models in the model pool into a single model.
    3. If a task pool is specified in the configuration, it loads the task pool and uses it to evaluate the merged model.
    """
    modelpool = load_modelpool(cfg.modelpool)

    algorithm = load_algorithm(cfg.method)
    merged_model = algorithm.fuse(modelpool)

    if hasattr(cfg, "taskpool") and cfg.taskpool is not None:
        taskpool = load_taskpool(cfg.taskpool)
        taskpool.evaluate(merged_model)
    else:
        print("No task pool specified. Skipping evaluation.")


@hydra.main(
    config_path=os.path.join(
        importlib.import_module("fusion_bench").__path__[0], "../config"
    ),
    config_name="example_config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if cfg.print_config:
        rich_print(
            Syntax(
                OmegaConf.to_yaml(cfg),
                "yaml",
                tab_size=2,
                line_numbers=True,
            )
        )

    run_model_fusion(cfg)


if __name__ == "__main__":
    main()
