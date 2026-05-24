from __future__ import annotations

import numpy as np
from pathlib import Path

class GetData():
    def __init__(self, n_instance: int, n_customers: int, *, seed: int = 2025, data_dir: str | None = None):
        self.n_instance = n_instance
        self.n_customers = n_customers
        self.seed = seed
        self.data_dir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent / "data"

    @property
    def data_path(self):
        return self.data_dir / f"instances_n{self.n_instance}_customers{self.n_customers}_seed{self.seed}.npz"

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
            # Depot + customers
            coords = np.random.rand(self.n_customers + 1, 2)  # (x, y) positions in [0, 1]^2
            demands = np.random.randint(1, 10, size=self.n_customers+1)
            demands[0]  = 0  # Depot has no demand

            # calculate distance matrix
            distance_matrix = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)

            # Set vehicle capacity based on number of customers
            if 20 <= self.n_customers < 40:
                capacity = 30
            elif 40 <= self.n_customers < 70:
                capacity = 40
            elif 70 <= self.n_customers <= 100:
                capacity = 50
            else:
                raise ValueError("Number of customers must be between 20 and 100.")

            instance_data.append((coords, demands, distance_matrix))

        return instance_data, capacity

    def save_instances(self, instance_data, capacity):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        coords = np.array([item[0] for item in instance_data])
        demands = np.array([item[1] for item in instance_data])
        distance_matrix = np.array([item[2] for item in instance_data])
        np.savez_compressed(
            self.data_path,
            coords=coords,
            demands=demands,
            distance_matrix=distance_matrix,
            capacity=np.array(capacity),
            n_instance=np.array(self.n_instance),
            n_customers=np.array(self.n_customers),
            seed=np.array(self.seed),
        )

    def load_instances(self):
        with np.load(self.data_path) as data:
            coords = data["coords"]
            demands = data["demands"]
            distance_matrix = data["distance_matrix"]
            capacity = float(data["capacity"])
        instance_data = [
            (coords[i].copy(), demands[i].copy(), distance_matrix[i].copy())
            for i in range(self.n_instance)
        ]
        return instance_data, capacity


if __name__ == '__main__':
    getData = GetData(3, 50)
    instance_data, capacity = getData.generate_instances()
    print("Coordinates (first instance):\n", instance_data[0][0].shape)
    print("Normalized Demands (first instance):\n", instance_data[0][1].shape)
    print("Distance Matrix (first instance):\n", instance_data[0][2])
    print("Vehicle Capacity:", capacity)
