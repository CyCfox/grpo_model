"""
Numpy-based vector memory store with cosine similarity retrieval.
Uses Qwen3-Embedding-0.6B (CPU) for text embedding.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL_PATH, EMBEDDING_DEVICE, TOP_K_RETRIEVAL, EMBEDDING_DIM

# Global embedding model (loaded once, shared across MemoryStore instances)
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Get or initialize the global embedding model on CPU."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_PATH,
            device=EMBEDDING_DEVICE,
        )
    return _embedding_model


def _cosine_similarity(query_vec: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query vector and matrix of vectors."""
    # Normalize
    query_norm = query_vec / (np.linalg.norm(query_vec, axis=-1, keepdims=True) + 1e-8)
    vecs_norm = vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-8)
    return np.dot(query_norm, vecs_norm.T)


class MemoryStore:
    """
    Numpy-based memory storage with embedding similarity retrieval.
    Stores memory entries as JSON metadata + numpy embedding vectors.
    """

    def __init__(self, embedding_model: Optional[SentenceTransformer] = None):
        """
        Initialize an empty memory store.

        Args:
            embedding_model: Optional pre-loaded embedding model.
                             If None, uses the global model.
        """
        self._model = embedding_model or get_embedding_model()
        self.vectors: np.ndarray = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        self.meta: List[Dict[str, str]] = []  # [{id, text, event}, ...]
        self._id_to_idx: Dict[str, int] = {}   # memory id -> index in vectors/meta
        self._next_id_counter: int = 0

    def __len__(self) -> int:
        return len(self.meta)

    def _embed(self, text: str) -> np.ndarray:
        """Embed a text into a vector."""
        emb = self._model.encode([text], show_progress_bar=False, normalize_embeddings=True)
        return emb.astype(np.float32)

    def _generate_id(self) -> str:
        """Generate a new unique memory ID."""
        mem_id = f"mem_{self._next_id_counter}"
        self._next_id_counter += 1
        return mem_id

    def _rebuild_index(self):
        """Rebuild the id->idx mapping after structural changes."""
        self._id_to_idx = {}
        for i, m in enumerate(self.meta):
            self._id_to_idx[m["id"]] = i

    # ===== Core Operations =====

    def add(self, text: str, mem_id: Optional[str] = None) -> str:
        """
        Add a new memory entry.

        Args:
            text: The memory content
            mem_id: Optional memory ID. If None, auto-generates one.

        Returns:
            The memory ID used.
        """
        if mem_id is None:
            mem_id = self._generate_id()

        emb = self._embed(text)
        if len(self.vectors) == 0:
            self.vectors = emb
        else:
            self.vectors = np.vstack([self.vectors, emb])

        self.meta.append({"id": mem_id, "text": text, "event": "ADD"})
        self._id_to_idx[mem_id] = len(self.meta) - 1
        return mem_id

    def update(self, mem_id: str, new_text: str) -> bool:
        """
        Update an existing memory entry.

        Args:
            mem_id: The ID of the memory to update
            new_text: The new text content

        Returns:
            True if successful, False if mem_id not found.
        """
        if mem_id not in self._id_to_idx:
            return False

        idx = self._id_to_idx[mem_id]
        emb = self._embed(new_text)
        self.vectors[idx] = emb[0]
        self.meta[idx]["text"] = new_text
        self.meta[idx]["event"] = "UPDATE"
        return True

    def delete(self, mem_id: str) -> bool:
        """
        Delete a memory entry.

        Args:
            mem_id: The ID of the memory to delete

        Returns:
            True if successful, False if mem_id not found.
        """
        if mem_id not in self._id_to_idx:
            return False

        idx = self._id_to_idx[mem_id]
        if len(self.vectors) == 1:
            self.vectors = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        else:
            self.vectors = np.delete(self.vectors, idx, axis=0)
        del self.meta[idx]
        self._rebuild_index()
        return True

    def search(self, query_text: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict[str, str]]:
        """
        Search for the most similar memories to the query.

        Args:
            query_text: The query text to search with
            top_k: Maximum number of results

        Returns:
            List of memory meta dicts, sorted by relevance (most relevant first).
        """
        if len(self.vectors) == 0:
            return []

        query_emb = self._embed(query_text)
        similarities = _cosine_similarity(query_emb, self.vectors)[0]

        # Get top-k indices
        k = min(top_k, len(self.meta))
        if k == 0:
            return []
        top_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] >= 0:  # Filter out negative similarities
                results.append(dict(self.meta[idx]))

        return results

    def get_all_memories(self) -> List[Dict[str, str]]:
        """Return all memory entries."""
        return [dict(m) for m in self.meta]

    def has_id(self, mem_id: str) -> bool:
        """Check if a memory ID exists."""
        return mem_id in self._id_to_idx

    def copy(self) -> "MemoryStore":
        """
        Create a deep copy of this memory store.
        Used for trajectory isolation during GRPO sampling.
        """
        new_store = MemoryStore(embedding_model=self._model)
        new_store.vectors = self.vectors.copy() if len(self.vectors) > 0 else np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        new_store.meta = [dict(m) for m in self.meta]
        new_store._id_to_idx = dict(self._id_to_idx)
        new_store._next_id_counter = self._next_id_counter
        return new_store

    def to_json(self) -> List[Dict[str, str]]:
        """Export all memories as a JSON-serializable list."""
        return [dict(m) for m in self.meta]

    # ===== Persistence =====

    def save(self, dirpath: str):
        """
        Save memory store state to disk.

        Saves:
          - vectors.npy: numpy embedding matrix
          - meta.json: list of {id, text, event} dicts
          - state.json: {_next_id_counter}

        Args:
            dirpath: Directory to save files into (created if needed).
        """
        import os
        import json
        os.makedirs(dirpath, exist_ok=True)
        np.save(os.path.join(dirpath, "vectors.npy"), self.vectors)
        with open(os.path.join(dirpath, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
        with open(os.path.join(dirpath, "state.json"), "w", encoding="utf-8") as f:
            json.dump({"next_id_counter": self._next_id_counter}, f)

    @staticmethod
    def load_state(dirpath: str, embedding_model: Optional[SentenceTransformer] = None) -> "MemoryStore":
        """
        Load a MemoryStore from disk.

        Args:
            dirpath: Directory containing vectors.npy, meta.json, state.json
            embedding_model: Optional pre-loaded embedding model.

        Returns:
            Reconstructed MemoryStore instance.
        """
        import os
        import json
        store = MemoryStore(embedding_model=embedding_model)
        vec_path = os.path.join(dirpath, "vectors.npy")
        if os.path.exists(vec_path):
            store.vectors = np.load(vec_path)
        meta_path = os.path.join(dirpath, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                store.meta = json.load(f)
        state_path = os.path.join(dirpath, "state.json")
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
                store._next_id_counter = state.get("next_id_counter", 0)
        store._rebuild_index()
        return store

    def __repr__(self) -> str:
        return f"MemoryStore({len(self.meta)} entries)"
