import numpy as np
from typing import Literal, Optional
import umap
from sklearn.decomposition import PCA


class DimensionalityReduction:
    def __init__(self, method: Literal["umap", "pca"] = "umap"):
        self.method = method.lower()
        if self.method not in ["umap", "pca"]:
            raise ValueError(f"Method must be 'umap' or 'pca', got '{method}'")

        self.reducer: Optional[object] = None
        self.is_fitted = False

    def fit(self, embeddings: np.ndarray, n_components: int = 2) -> None:
        """
        Fit the dimensionality reduction model on training embeddings.
        This should be called once at startup with a representative dataset.
        """
        if self.method == "umap":
            n_samples = embeddings.shape[0]
            n_neighbors = min(15, n_samples - 1)
            self.reducer = umap.UMAP(
                n_components=n_components,
                random_state=42,
                n_neighbors=n_neighbors
            )
        else:
            self.reducer = PCA(n_components=n_components, random_state=42)

        self.reducer.fit(embeddings)
        self.is_fitted = True

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings using the pre-fitted reducer.
        Points will be projected into the same 2D space, keeping existing points fixed.
        """
        if not self.is_fitted:
            raise RuntimeError("Reducer must be fitted before transform. Call fit() first.")

        return self.reducer.transform(embeddings)

    def reduce(self, embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
        """
        Legacy method for backward compatibility.
        Uses transform() if fitted, otherwise fits and transforms.
        """
        if embeddings.shape[0] == 0:
            return np.array([])

        if self.is_fitted:
            return self.transform(embeddings)
        else:
            # Fallback: fit and transform (not recommended for production)
            if self.method == "umap":
                n_samples = embeddings.shape[0]
                n_neighbors = min(15, max(2, n_samples - 1))
                reducer = umap.UMAP(
                    n_components=n_components,
                    random_state=42,
                    n_neighbors=n_neighbors
                )
            else:
                reducer = PCA(n_components=n_components, random_state=42)

            return reducer.fit_transform(embeddings)
