import os
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FaceVectorStore:
    """
    FAISS-based vector store for face recognition system.
    Manages face embeddings database with CRUD operations and similarity search.
    """

    def __init__(self, dimension: int = 512, index_path: str = None, metadata_path: str = None, embeddings_path: str = None):
        """
        Initialize the face vector store.

        Args:
            dimension: Dimension of face embeddings (default: 512 for GhostFaceNet)
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load metadata
            embeddings_path: Path to save/load embeddings
        """
        self.dimension = dimension
        self.index = None
        self.embeddings = np.empty((0, dimension), dtype=np.float32)
        self.metadata = []
        self.id_counter = 0

        # Set default paths relative to this file
        if index_path is None:
            index_path = os.path.join(os.path.dirname(__file__), "../../../vector_db/faiss_index.bin")
        if metadata_path is None:
            metadata_path = os.path.join(os.path.dirname(__file__), "../../../vector_db/metadata.json")
        if embeddings_path is None:
            embeddings_path = os.path.join(os.path.dirname(__file__), "../../../vector_db/embeddings.npy")

        self.index_path = os.path.abspath(index_path)
        self.metadata_path = os.path.abspath(metadata_path)
        self.embeddings_path = os.path.abspath(embeddings_path)

        # Create vector_db directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        # Initialize or load existing index
        self._initialize_index()

    def _initialize_index(self):
        """Initialize FAISS index and load existing data if available"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path) and os.path.exists(self.embeddings_path):
                # Load existing index and data
                self._load_index()
                logger.info(f"Loaded existing vector store with {len(self.metadata)} entries")
            else:
                # Create new index
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
                logger.info("Created new FAISS index")
        except Exception as e:
            logger.error(f"Error initializing index: {e}")
            # Create new index as fallback
            self.index = faiss.IndexFlatIP(self.dimension)

    def _load_index(self):
        """Load FAISS index and associated data from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(self.index_path)

            # Load embeddings
            self.embeddings = np.load(self.embeddings_path)

            # Load metadata
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.metadata = data.get('metadata', [])
                self.id_counter = data.get('id_counter', len(self.metadata))

            logger.info(f"Loaded index with {self.index.ntotal} vectors")

        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise

    def _save_index(self):
        """Save FAISS index and associated data to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)

            # Save embeddings
            np.save(self.embeddings_path, self.embeddings)

            # Save metadata
            data = {
                'metadata': self.metadata,
                'id_counter': self.id_counter,
                'dimension': self.dimension,
                'total_vectors': len(self.metadata),
                'last_updated': datetime.now().isoformat()
            }

            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved index with {len(self.metadata)} vectors")

        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise

    def add(self, embedding: np.ndarray, person_info: Dict[str, Any]) -> int:
        """
        Add a single face embedding to the vector store.

        Args:
            embedding: Face embedding vector (512,)
            person_info: Dictionary containing person information
                Required: 'name'
                Optional: 'id', 'description', 'additional_info'

        Returns:
            person_id: Unique ID assigned to the person
        """
        try:
            # Ensure embedding is 2D and normalized
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)

            # L2 normalize the embedding
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

            # Generate person ID
            person_id = person_info.get('id', self.id_counter)
            if person_id >= self.id_counter:
                self.id_counter = person_id + 1

            # Create metadata entry
            metadata_entry = {
                'id': person_id,
                'name': person_info['name'],
                'description': person_info.get('description', ''),
                'additional_info': person_info.get('additional_info', {}),
                'registered_at': datetime.now().isoformat(),
                'embedding_index': len(self.metadata)
            }

            # Add to FAISS index
            self.index.add(embedding.astype(np.float32))

            # Add to embeddings array
            self.embeddings = np.vstack([self.embeddings, embedding])

            # Add to metadata
            self.metadata.append(metadata_entry)

            # Save to disk
            self._save_index()

            logger.info(f"Added person {person_id}: {person_info['name']}")
            return person_id

        except Exception as e:
            logger.error(f"Error adding embedding: {e}")
            raise

    def add_batch(self, embeddings: np.ndarray, persons_info: List[Dict[str, Any]]) -> List[int]:
        """
        Add multiple face embeddings to the vector store.

        Args:
            embeddings: Array of face embeddings (N, 512)
            persons_info: List of person information dictionaries

        Returns:
            person_ids: List of assigned person IDs
        """
        try:
            if len(embeddings) != len(persons_info):
                raise ValueError("Number of embeddings must match number of persons")

            # Ensure embeddings are 2D and normalized
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            # L2 normalize the embeddings
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            person_ids = []
            start_index = len(self.metadata)

            for i, person_info in enumerate(persons_info):
                # Generate person ID
                person_id = person_info.get('id', self.id_counter)
                if person_id >= self.id_counter:
                    self.id_counter = person_id + 1

                # Create metadata entry
                metadata_entry = {
                    'id': person_id,
                    'name': person_info['name'],
                    'description': person_info.get('description', ''),
                    'additional_info': person_info.get('additional_info', {}),
                    'registered_at': datetime.now().isoformat(),
                    'embedding_index': start_index + i
                }

                self.metadata.append(metadata_entry)
                person_ids.append(person_id)

            # Add to FAISS index
            self.index.add(embeddings.astype(np.float32))

            # Add to embeddings array
            self.embeddings = np.vstack([self.embeddings, embeddings])

            # Save to disk
            self._save_index()

            logger.info(f"Added batch of {len(person_ids)} persons")
            return person_ids

        except Exception as e:
            logger.error(f"Error adding batch embeddings: {e}")
            raise

    def search(self, query_embedding: np.ndarray, top_k: int = 5, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar faces in the vector store.

        Args:
            query_embedding: Query face embedding (512,)
            top_k: Number of top similar results to return
            threshold: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of dictionaries containing search results with similarity scores
        """
        try:
            if self.index.ntotal == 0:
                return []

            # Ensure query embedding is 2D and normalized
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            # L2 normalize the query embedding
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

            # Search in FAISS index
            similarities, indices = self.index.search(query_embedding.astype(np.float32), min(top_k, self.index.ntotal))

            results = []
            for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == -1 or sim < threshold:  # FAISS returns -1 for invalid indices
                    continue

                # Get metadata for this result
                metadata = self.metadata[idx]
                result = {
                    'person_id': metadata['id'],
                    'name': metadata['name'],
                    'description': metadata['description'],
                    'similarity': float(sim),
                    'additional_info': metadata.get('additional_info', {}),
                    'registered_at': metadata['registered_at']
                }
                results.append(result)

            # Sort by similarity (descending)
            results.sort(key=lambda x: x['similarity'], reverse=True)

            logger.info(f"Search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise

    def remove(self, person_id: int) -> bool:
        """
        Remove a person from the vector store.

        Args:
            person_id: ID of the person to remove

        Returns:
            True if successfully removed, False otherwise
        """
        try:
            # Find the person in metadata
            person_index = None
            for i, meta in enumerate(self.metadata):
                if meta['id'] == person_id:
                    person_index = i
                    break

            if person_index is None:
                logger.warning(f"Person {person_id} not found")
                return False

            # Remove from metadata
            removed_meta = self.metadata.pop(person_index)

            # Rebuild index and embeddings (FAISS doesn't support deletion)
            if len(self.metadata) > 0:
                # Get remaining embeddings
                remaining_embeddings = np.delete(self.embeddings, person_index, axis=0)

                # Rebuild FAISS index
                self.index = faiss.IndexFlatIP(self.dimension)
                self.index.add(remaining_embeddings.astype(np.float32))

                # Update embeddings array
                self.embeddings = remaining_embeddings

                # Update embedding indices in metadata
                for i, meta in enumerate(self.metadata):
                    meta['embedding_index'] = i
            else:
                # No more entries, reset everything
                self.index = faiss.IndexFlatIP(self.dimension)
                self.embeddings = np.empty((0, self.dimension), dtype=np.float32)

            # Save changes
            self._save_index()

            logger.info(f"Removed person {person_id}: {removed_meta['name']}")
            return True

        except Exception as e:
            logger.error(f"Error removing person {person_id}: {e}")
            raise

    def get_person(self, person_id: int) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific person.

        Args:
            person_id: ID of the person

        Returns:
            Person information dictionary or None if not found
        """
        for meta in self.metadata:
            if meta['id'] == person_id:
                return {
                    'id': meta['id'],
                    'name': meta['name'],
                    'description': meta['description'],
                    'additional_info': meta.get('additional_info', {}),
                    'registered_at': meta['registered_at']
                }
        return None

    def list_persons(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List all persons in the vector store.

        Args:
            limit: Maximum number of results to return
            offset: Number of results to skip

        Returns:
            List of person information dictionaries
        """
        start_idx = offset
        end_idx = min(offset + limit, len(self.metadata))

        persons = []
        for meta in self.metadata[start_idx:end_idx]:
            persons.append({
                'id': meta['id'],
                'name': meta['name'],
                'description': meta['description'],
                'additional_info': meta.get('additional_info', {}),
                'registered_at': meta['registered_at']
            })

        return persons

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary containing store statistics
        """
        return {
            'total_persons': len(self.metadata),
            'dimension': self.dimension,
            'index_type': 'FAISS IndexFlatIP',
            'last_updated': datetime.now().isoformat(),
            'storage_paths': {
                'index': self.index_path,
                'metadata': self.metadata_path,
                'embeddings': self.embeddings_path
            }
        }

    def clear(self):
        """Clear all data from the vector store"""
        try:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.embeddings = np.empty((0, self.dimension), dtype=np.float32)
            self.metadata = []
            self.id_counter = 0

            # Remove files
            for path in [self.index_path, self.metadata_path, self.embeddings_path]:
                if os.path.exists(path):
                    os.remove(path)

            logger.info("Cleared all data from vector store")

        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            raise