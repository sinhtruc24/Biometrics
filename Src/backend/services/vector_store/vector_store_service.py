import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from .face_vector_store import FaceVectorStore
from ..face_verification_service import FaceVerificationService

logger = logging.getLogger(__name__)

class VectorStoreService:
    """
    Service for managing face vector database operations.
    Integrates with FaceVerificationService for embedding extraction.
    """

    def __init__(self, face_service: FaceVerificationService = None, vector_store_path: str = None):
        """
        Initialize vector store service.

        Args:
            face_service: FaceVerificationService instance for embedding extraction
            vector_store_path: Base path for vector store data
        """
        self.face_service = face_service or FaceVerificationService()

        # Set default vector store path
        if vector_store_path is None:
            vector_store_path = os.path.join(os.path.dirname(__file__), "../../vector_db")

        self.vector_store_path = os.path.abspath(vector_store_path)

        # Initialize face vector store
        self.vector_store = FaceVectorStore(
            dimension=512,  # GhostFaceNet embedding dimension
            index_path=os.path.join(self.vector_store_path, "faiss_index.bin"),
            metadata_path=os.path.join(self.vector_store_path, "metadata.json"),
            embeddings_path=os.path.join(self.vector_store_path, "embeddings.npy")
        )

        logger.info("VectorStoreService initialized")

    def register_face(self, image_bytes: bytes, person_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new face in the vector database.

        Args:
            image_bytes: Face image bytes
            person_info: Person information dictionary
                Required: 'name'
                Optional: 'description', 'additional_info'

        Returns:
            Registration result with person_id and status
        """
        try:
            if not self.face_service.is_model_loaded():
                raise RuntimeError("Face verification model not loaded")

            # Extract face embedding
            preprocessed_img = self.face_service._preprocess_image(image_bytes)
            embedding = self.face_service._extract_embedding(preprocessed_img)

            # Register in vector store
            person_id = self.vector_store.add(embedding, person_info)

            result = {
                'success': True,
                'person_id': person_id,
                'name': person_info['name'],
                'message': f'Successfully registered {person_info["name"]}',
                'registered_at': datetime.now().isoformat()
            }

            logger.info(f"Registered new person: {person_id} - {person_info['name']}")
            return result

        except Exception as e:
            error_msg = f"Failed to register face: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'person_id': None
            }

    def recognize_face(self, image_bytes: bytes, top_k: int = 5, threshold: float = 0.25) -> Dict[str, Any]:
        """
        Recognize a face by searching in the vector database.

        Args:
            image_bytes: Query face image bytes
            top_k: Number of top matches to return
            threshold: Minimum similarity threshold

        Returns:
            Recognition results with matches and confidence scores
        """
        try:
            if not self.face_service.is_model_loaded():
                raise RuntimeError("Face verification model not loaded")

            if self.vector_store.index.ntotal == 0:
                return {
                    'success': True,
                    'recognized': False,
                    'message': 'No faces registered in database',
                    'matches': []
                }

            # Extract face embedding
            preprocessed_img = self.face_service._preprocess_image(image_bytes)
            embedding = self.face_service._extract_embedding(preprocessed_img)

            # Search in vector store
            matches = self.vector_store.search(embedding, top_k=top_k, threshold=threshold)

            # Determine if recognized
            recognized = len(matches) > 0 and matches[0]['similarity'] >= threshold

            result = {
                'success': True,
                'recognized': recognized,
                'matches': matches,
                'top_match': matches[0] if matches else None,
                'confidence': matches[0]['similarity'] if matches else 0.0,
                'message': f'Found {len(matches)} potential matches' if matches else 'No matches found'
            }

            logger.info(f"Face recognition: {'recognized' if recognized else 'not recognized'}, {len(matches)} matches")
            return result

        except Exception as e:
            error_msg = f"Failed to recognize face: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'recognized': False,
                'error': error_msg,
                'matches': []
            }

    def remove_person(self, person_id: int) -> Dict[str, Any]:
        """
        Remove a person from the vector database.

        Args:
            person_id: ID of the person to remove

        Returns:
            Removal result with status
        """
        try:
            person_info = self.vector_store.get_person(person_id)
            if not person_info:
                return {
                    'success': False,
                    'error': f'Person {person_id} not found',
                    'person_id': person_id
                }

            # Remove from vector store
            removed = self.vector_store.remove(person_id)

            if removed:
                result = {
                    'success': True,
                    'person_id': person_id,
                    'name': person_info['name'],
                    'message': f'Successfully removed {person_info["name"]}'
                }
                logger.info(f"Removed person: {person_id} - {person_info['name']}")
            else:
                result = {
                    'success': False,
                    'error': f'Failed to remove person {person_id}',
                    'person_id': person_id
                }

            return result

        except Exception as e:
            error_msg = f"Failed to remove person: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'person_id': person_id
            }

    def get_person_info(self, person_id: int) -> Dict[str, Any]:
        """
        Get information about a registered person.

        Args:
            person_id: ID of the person

        Returns:
            Person information or error response
        """
        try:
            person_info = self.vector_store.get_person(person_id)

            if person_info:
                return {
                    'success': True,
                    'person': person_info
                }
            else:
                return {
                    'success': False,
                    'error': f'Person {person_id} not found',
                    'person_id': person_id
                }

        except Exception as e:
            error_msg = f"Failed to get person info: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'person_id': person_id
            }

    def list_registered_persons(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """
        List all registered persons.

        Args:
            limit: Maximum number of persons to return
            offset: Number of persons to skip

        Returns:
            List of registered persons
        """
        try:
            persons = self.vector_store.list_persons(limit=limit, offset=offset)

            return {
                'success': True,
                'persons': persons,
                'total': len(persons),
                'limit': limit,
                'offset': offset
            }

        except Exception as e:
            error_msg = f"Failed to list persons: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'persons': []
            }

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the face database.

        Returns:
            Database statistics
        """
        try:
            stats = self.vector_store.get_stats()

            return {
                'success': True,
                'stats': stats,
                'is_model_loaded': self.face_service.is_model_loaded()
            }

        except Exception as e:
            error_msg = f"Failed to get database stats: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'stats': {}
            }

    def verify_faces_with_database(self, image_a_bytes: bytes, image_b_bytes: bytes,
                                 threshold: float = 0.25) -> Dict[str, Any]:
        """
        Verify if two faces belong to the same person and check against database.

        Args:
            image_a_bytes: First face image bytes
            image_b_bytes: Second face image bytes
            threshold: Similarity threshold

        Returns:
            Verification result with database checks
        """
        try:
            # First, verify the two faces directly
            verification_result = self.face_service.verify_faces(image_a_bytes, image_b_bytes)

            # Then check if either face is registered in database
            recognition_a = self.recognize_face(image_a_bytes, top_k=1, threshold=threshold)
            recognition_b = self.recognize_face(image_b_bytes, top_k=1, threshold=threshold)

            result = {
                'success': True,
                'face_verification': {
                    'similarity': verification_result.similarity,
                    'is_same_person': verification_result.is_same_person,
                    'threshold': verification_result.threshold,
                    'inference_time': getattr(verification_result, 'inference_time', None)
                },
                'database_check': {
                    'face_a_registered': recognition_a['recognized'],
                    'face_b_registered': recognition_b['recognized'],
                    'face_a_match': recognition_a.get('top_match'),
                    'face_b_match': recognition_b.get('top_match')
                }
            }

            # Determine overall result
            if verification_result.is_same_person:
                if recognition_a['recognized'] and recognition_b['recognized']:
                    # Both faces recognized and same person
                    if recognition_a['top_match']['person_id'] == recognition_b['top_match']['person_id']:
                        result['message'] = f"Same person: {recognition_a['top_match']['name']}"
                    else:
                        result['message'] = "Faces match but registered as different persons"
                elif recognition_a['recognized']:
                    result['message'] = f"Face A recognized as: {recognition_a['top_match']['name']}"
                elif recognition_b['recognized']:
                    result['message'] = f"Face B recognized as: {recognition_b['top_match']['name']}"
                else:
                    result['message'] = "Faces match but neither is registered in database"
            else:
                result['message'] = "Faces do not match"

            return result

        except Exception as e:
            error_msg = f"Failed to verify faces with database: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }