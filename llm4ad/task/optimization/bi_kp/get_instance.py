from __future__ import annotations

import numpy as np
from pathlib import Path


class GetData():
    def __init__(self, n_instance: int, n_items: int, *, seed: int = 2025, data_dir: str | None = None):
        self.n_instance = n_instance
        self.n_items = n_items
        self.seed = seed
        self.data_dir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent / "data"

    @property
    def data_path(self):
        return self.data_dir / f"instances_n{self.n_instance}_items{self.n_items}_seed{self.seed}.npz"

    def generate_instances(self):
        if self.data_path.exists():
            return self.load_instances()
        instance_data, capacity = self._generate_instances()
        self.save_instances(instance_data, capacity)
        return instance_data, capacity

    def _generate_instances(self):
        np.random.seed(self.seed)
        instance_data = []
        for _ in range(self.n_instance):
            weights = np.random.rand(self.n_items)
            values_obj1 = np.random.rand(self.n_items)
            values_obj2 = np.random.rand(self.n_items)
            if 20 <= self.n_items < 50:
                capacity = 5
            elif 50 <= self.n_items < 100:
                capacity = 12.5
            elif 100 <= self.n_items <= 200:
                capacity = 25
            else:
                raise ValueError("Number of items must be between 20 and 200.")

            instance_data.append((weights, values_obj1, values_obj2))
        return instance_data, capacity

    def save_instances(self, instance_data, capacity):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        weights = np.array([item[0] for item in instance_data])
        values_obj1 = np.array([item[1] for item in instance_data])
        values_obj2 = np.array([item[2] for item in instance_data])
        np.savez_compressed(
            self.data_path,
            weights=weights,
            values_obj1=values_obj1,
            values_obj2=values_obj2,
            capacity=np.array(capacity),
            n_instance=np.array(self.n_instance),
            n_items=np.array(self.n_items),
            seed=np.array(self.seed),
        )

    def load_instances(self):
        with np.load(self.data_path) as data:
            weights = data["weights"]
            values_obj1 = data["values_obj1"]
            values_obj2 = data["values_obj2"]
            capacity = float(data["capacity"])
        instance_data = [
            (weights[i].copy(), values_obj1[i].copy(), values_obj2[i].copy())
            for i in range(self.n_instance)
        ]
        return instance_data, capacity
    

if __name__ == '__main__':
    getData = GetData(5, 200)
    instance_data, capacity = getData.generate_instances()
    print(instance_data[0])
