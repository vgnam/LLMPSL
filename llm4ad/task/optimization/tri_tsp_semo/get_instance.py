from __future__ import annotations

import numpy as np
from pathlib import Path


class GetData():
    def __init__(self, n_instance, n_cities, *, seed: int = 2025, data_dir: str | None = None):
        self.n_instance = n_instance
        self.n_cities = n_cities
        self.seed = seed
        self.data_dir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent / "data"

    @property
    def data_path(self):
        return self.data_dir / f"instances_n{self.n_instance}_cities{self.n_cities}_seed{self.seed}.npz"

    def generate_instances(self):
        if self.data_path.exists():
            return self.load_instances()
        instance_data = self._generate_instances()
        self.save_instances(instance_data)
        return instance_data

    def _generate_instances(self):
        np.random.seed(self.seed)
        instance_data = []
        for _ in range(self.n_instance):
            coordinates_1 = np.random.rand(self.n_cities, 2)
            coordinates_2 = np.random.rand(self.n_cities, 2)
            coordinates_3 = np.random.rand(self.n_cities, 2)
            coordinates = np.concatenate((coordinates_1, coordinates_2, coordinates_3), axis=1)
            distances_1 = np.linalg.norm(coordinates_1[:, np.newaxis] - coordinates_1, axis=2)
            distances_2 = np.linalg.norm(coordinates_2[:, np.newaxis] - coordinates_2, axis=2)
            distances_3 = np.linalg.norm(coordinates_3[:, np.newaxis] - coordinates_3, axis=2)
            instance_data.append((coordinates,distances_1, distances_2, distances_3))
        return instance_data

    def save_instances(self, instance_data):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        coordinates = np.array([item[0] for item in instance_data])
        distances_1 = np.array([item[1] for item in instance_data])
        distances_2 = np.array([item[2] for item in instance_data])
        distances_3 = np.array([item[3] for item in instance_data])
        np.savez_compressed(
            self.data_path,
            coordinates=coordinates,
            distances_1=distances_1,
            distances_2=distances_2,
            distances_3=distances_3,
            n_instance=np.array(self.n_instance),
            n_cities=np.array(self.n_cities),
            seed=np.array(self.seed),
        )

    def load_instances(self):
        with np.load(self.data_path) as data:
            coordinates = data["coordinates"]
            distances_1 = data["distances_1"]
            distances_2 = data["distances_2"]
            distances_3 = data["distances_3"]
        return [
            (coordinates[i].copy(), distances_1[i].copy(), distances_2[i].copy(), distances_3[i].copy())
            for i in range(self.n_instance)
        ]
