from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent


DATASET_SPECS = [
    {
        "name": "bi_kp",
        "path": ROOT / "bi_kp" / "get_instance.py",
        "n_instance": 8,
        "train_size": 20,
        "test_sizes": [50, 100, 150, 200],
    },
    {
        "name": "bi_tsp_semo",
        "path": ROOT / "bi_tsp_semo" / "get_instance.py",
        "n_instance": 4,
        "train_size": 20,
        "test_sizes": [50, 100],
    },
    {
        "name": "tri_tsp_semo",
        "path": ROOT / "tri_tsp_semo" / "get_instance.py",
        "n_instance": 20,
        "train_size": 20,
        "test_sizes": [50, 100],
    },
    {
        "name": "bi_cvrp",
        "path": ROOT / "bi_cvrp" / "get_instance.py",
        "n_instance": 8,
        "train_size": 20,
        "test_sizes": [50, 100],
    },
]


def load_get_data(path: Path):
    spec = importlib.util.spec_from_file_location(f"{path.parent.name}_get_instance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GetData


def main():
    for spec in DATASET_SPECS:
        name = spec["name"]
        get_instance_path = spec["path"]
        data_cls = load_get_data(get_instance_path)
        sizes = [spec["train_size"]] + spec["test_sizes"]
        for size in sizes:
            data = data_cls(spec["n_instance"], size)
            data.generate_instances()
            split = "train" if size == spec["train_size"] else "test"
            print(f"{name} {split} size={size}: {data.data_path}")


if __name__ == "__main__":
    main()
