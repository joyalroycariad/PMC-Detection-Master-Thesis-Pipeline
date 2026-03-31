import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


class Clusterer:
    """
    A reusable DBSCAN clustering class for semantic text embeddings.
    """

    def __init__(self, 
                 eps: float = 0.14,
                 min_samples: int = 5,
                 metric: str = "cosine",
                 n_jobs: int = -1):
        
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.n_jobs = n_jobs
        
        # Prepare model
        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            n_jobs=self.n_jobs
        )

    def cluster_embeddings(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        cluster_column: str = "CLUSTER_ID"
    ) -> pd.DataFrame:
        """
        Applies DBSCAN to embeddings and assigns cluster IDs to dataframe.
        
        Returns:
            df (modified with cluster column)
        """

        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Embeddings are empty or None.")

        print("🔍 Running DBSCAN clustering...")
        print(f"📌 eps={self.eps}, min_samples={self.min_samples}, metric={self.metric}")

        labels = self.model.fit_predict(embeddings)
        df[cluster_column] = labels

        # Cluster stats
        num_clusters = len(set(labels) - {-1})
        num_noise = list(labels).count(-1)

        print(f"✅ Clustering complete.")
        print(f"📊 Number of clusters: {num_clusters}")
        print(f"🟦 Noise points (-1): {num_noise}")
        print(f"🧩 Total items clustered: {len(labels)}")

        return df
