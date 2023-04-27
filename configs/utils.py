import os
from pathlib import Path
from typing import Union

import yaml


def get_config(path: Union[Path, str], dataset: str):
    if type(path) is not Path:
        path = Path(path)
    datasets_list = [name.split(".")[0] for name in os.listdir(path.parent / "datasets")]
    assert dataset in datasets_list, f"Dataset should be from {datasets_list} or the yaml is needed to write"

    cfg = load_config(path)
    cfg_data = load_config((path.parent / "datasets" / dataset).with_suffix(".yaml"))
    cfg.update(cfg_data)
    return cfg


def load_config(path: str):
    with open(path, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg


if __name__ == "__main__":
    cfg = get_config("slim.yaml", "waymo")

    # cfg_dataset = load_config("datasets/rawkitti.yaml")

    a = None
