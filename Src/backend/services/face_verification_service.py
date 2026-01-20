import os
import numpy as np
import tensorflow as tf
import tf_keras as keras
import cv2
from app.models.api_models import VerificationResponse
from services.face_alignment_service import FaceAlignmentService
import logging

logger = logging.getLogger(__name__)

class FaceVerificationService:
    def __init__(self, model_path: str = None, threshold: float = 0.25):
        self.threshold = threshold
        self.model = None
        self.model_loaded = False
        
        # Initialize alignment service
        try:
            self.alignment_service = FaceAlignmentService()
        except Exception as e:
            logger.error(f"Failed to initialize FaceAlignmentService: {str(e)}")
            raise

        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "../../..", "Model", "ghostfacenet_fixed.h5")

        self.model_path = os.path.abspath(model_path)
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading model from: {self.model_path}")

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            self.model = keras.models.load_model(self.model_path, compile=False)
            
            self.model_loaded = True
            logger.info("Model loaded successfully using tf_keras")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.model_loaded = False
            raise

    def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_bgr is None:
                raise ValueError("Failed to decode image")

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # 1. Face Detection
            faces = self.alignment_service.detect_faces(img_rgb)
            if not faces:
                raise ValueError("No face detected in image")
            
            # Select largest face
            # MTCNN returns 'box': [x, y, w, h]
            largest_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
            
            # 2. Alignment
            img_aligned = self.alignment_service.align_face(img_rgb, largest_face['keypoints'])
            
            # 3. Normalization (Target: 112x112, RGB, -1..1)
            # img_aligned is already 112x112 from align_face
            img_tensor = (img_aligned.astype(np.float32) - 127.5) / 128.0
            img_tensor = np.expand_dims(img_tensor, axis=0)

            return img_tensor

        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    def verify_faces(self, image_a_bytes: bytes, image_b_bytes: bytes) -> VerificationResponse:
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")

        try:
            # Preprocess includes detection and alignment now
            img_a = self._preprocess_image(image_a_bytes)
            img_b = self._preprocess_image(image_b_bytes)

            emb_a = self.model(img_a, training=False)
            emb_b = self.model(img_b, training=False)

            emb_a_norm = tf.nn.l2_normalize(emb_a, axis=1)
            emb_b_norm = tf.nn.l2_normalize(emb_b, axis=1)

            similarity = tf.reduce_sum(tf.multiply(emb_a_norm, emb_b_norm), axis=1).numpy()[0]
            
            similarity = float(similarity)
            is_same_person = similarity >= self.threshold

            return VerificationResponse(
                similarity=round(similarity, 4),
                is_same_person=is_same_person,
                threshold=self.threshold
            )

        except Exception as e:
            logger.error(f"Face verification failed: {str(e)}")
            raise

    def _extract_embedding(self, preprocessed_image: np.ndarray) -> np.ndarray:
        """
        Extract face embedding from preprocessed image.

        Args:
            preprocessed_image: Preprocessed image tensor of shape (1, 112, 112, 3)

        Returns:
            Face embedding vector of shape (512,)
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")

        try:
            # Get embedding from model
            embedding = self.model(preprocessed_image, training=False)

            # Normalize the embedding (L2 normalization)
            embedding_norm = tf.nn.l2_normalize(embedding, axis=1)

            # Return as numpy array, squeeze to remove batch dimension
            return embedding_norm.numpy().squeeze()

        except Exception as e:
            logger.error(f"Failed to extract embedding: {str(e)}")
            raise

    def is_model_loaded(self) -> bool:
        return self.model_loaded

    def get_model_info(self) -> dict:
        if not self.model_loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "path": self.model_path,
            "threshold": self.threshold,
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape
        }
        