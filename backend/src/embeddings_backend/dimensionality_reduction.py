import numpy as np
from typing import Literal
import umap
from sklearn.decomposition import PCA


class DimensionalityReduction:
    def __init__(self, method: Literal["umap", "pca"] = "umap"):
        self.method = method.lower()
        if self.method not in ["umap", "pca"]:
            raise ValueError(f"Method must be 'umap' or 'pca', got '{method}'")

    def reduce(self, embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
        n_samples = embeddings.shape[0]

        if n_samples == 0:
            return np.array([])

        if n_samples == 1:
            return np.array([[0.0, 0.0]])

        if n_samples == 2:
            return np.array([[0.0, 0.0], [1.0, 0.0]])

        if self.method == "umap":
            return self._reduce_umap(embeddings, n_components, n_samples)
        else:
            return self._reduce_pca(embeddings, n_components)

    def _reduce_umap(self, embeddings: np.ndarray, n_components: int, n_samples: int) -> np.ndarray:
        n_neighbors = min(15, n_samples - 1)
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=42,
            n_neighbors=n_neighbors
        )
        return reducer.fit_transform(embeddings)

    def _reduce_pca(self, embeddings: np.ndarray, n_components: int) -> np.ndarray:
        reducer = PCA(n_components=n_components, random_state=42)
        return reducer.fit_transform(embeddings)
