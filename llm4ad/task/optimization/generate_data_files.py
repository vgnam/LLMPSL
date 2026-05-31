from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent


DATASET_SPECS = [
    {
        "name": "bi_kp",
        "path": ROOT / "bi_kp" / "get_instance.py",
        "train_n_instance": 10,
        "train_size": 20,
        "test_sizes": [50, 100, 150, 200, 300, 400],
        "test_n_instances": [20, 20, 20, 20, 20, 20],
    },
    {
        "name": "bi_tsp_semo",
        "path": ROOT / "bi_tsp_semo" / "get_instance.py",
        "train_n_instance": 10,
        "train_size": 20,
        "test_sizes": [50, 100, 200],
        "test_n_instances": [20, 20, 20],
    },
    {
        "name": "tri_tsp_semo",
        "path": ROOT / "tri_tsp_semo" / "get_instance.py",
        "train_n_instance": 10,
        "train_size": 20,
        "test_sizes": [50, 100, 200],
        "test_n_instances": [20, 20, 20],
    },
    {
        "name": "bi_cvrp",
        "path": ROOT / "bi_cvrp" / "get_instance.py",
        "train_n_instance": 10,
        "train_size": 20,
        "test_sizes": [50, 100, 150, 200],
        "test_n_instances": [20, 20, 20, 20],
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

        # Generate train set
        train_size = spec["train_size"]
        train_n_instance = spec["train_n_instance"]
        data = data_cls(train_n_instance, train_size)
        data.generate_instances()
        print(f"{name} train size={train_size} n_instance={train_n_instance}: {data.data_path}")

        # Generate test sets
        test_sizes = spec["test_sizes"]
        test_n_instances = spec["test_n_instances"]
        for size, n_inst in zip(test_sizes, test_n_instances):
            data = data_cls(n_inst, size)
            data.generate_instances()
            print(f"{name} test size={size} n_instance={n_inst}: {data.data_path}")


if __name__ == "__main__":
    main()
