import torch
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from typing import Tuple, List
import pandas as pd


class TextEmbedder:
    """
    A reusable embedding generator class for problem ticket text.
    Supports batching, reproducibility, and custom model loading.
    """

    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 batch_size: int = 32,
                 set_seed: bool = True):
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

        if set_seed:
            self._set_reproducible_seed()

    def _set_reproducible_seed(self):
        """Set reproducible seeds for PyTorch, NumPy, and Python."""
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        torch.use_deterministic_algorithms(True)

    def generate_embeddings(
        self,
        df: pd.DataFrame,
        text_column: str = "CLEANED_ERROR_MESSAGE"
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate embeddings from a dataframe column.

        Inputs:
            df: DataFrame with cleaned error messages
            text_column: column containing text to embed

        Returns:
            df (unchanged)
            embeddings: numpy array, shape (N, D)
        """

        # Safety check
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in dataframe.")

        text_list: List[str] = df[text_column].tolist()

        print(f"📌 Generating embeddings for {len(text_list)} messages...")
        print(f"📌 Using model: {self.model_name}")

        embeddings = self.model.encode(
            text_list,
            show_progress_bar=True,
            batch_size=self.batch_size,
            normalize_embeddings=True
        )

        print(f"✅ Embeddings generated. Shape: {embeddings.shape}")

        return df, embeddings