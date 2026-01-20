import numpy as np
import cv2
from mtcnn import MTCNN
import logging
from typing import Tuple, List, Optional
from skimage import transform as trans

logger = logging.getLogger(__name__)

class FaceAlignmentService:
    def __init__(self):
        try:
            self.detector = MTCNN()
            logger.info("MTCNN detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MTCNN: {str(e)}")
            raise

    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces in the image using MTCNN.
        Returns a list of dictionaries, each containing 'box' and 'keypoints'.
        """
        try:
            # MTCNN expects RGB image
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            
            # MTCNN handles RGB internally? Actually mtcnn library expects RGB usually.
            # Assuming input 'image' to this service is already RGB from preprocess_image.
            
            results = self.detector.detect_faces(image)
            return results
        except Exception as e:
            logger.error(f"Error during face detection: {str(e)}")
            return []

    def align_face(self, image: np.ndarray, landmarks: dict) -> np.ndarray:
        """
        Align face using 5-point landmarks to ArcFace standard 112x112.
        
        Args:
            image: RGB image
            landmarks: dict with keys 'left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right'
            
        Returns:
            Aligned face image (112x112)
        """
        try:
            # ArcFace standard 5 points
            # src points (theoretical positions for 112x112)
            src = np.array([
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041]
            ], dtype=np.float32)

            dst = np.array([
                landmarks['left_eye'],
                landmarks['right_eye'],
                landmarks['nose'],
                landmarks['mouth_left'],
                landmarks['mouth_right']
            ], dtype=np.float32)

            # Estimate affine transform
            tform = trans.SimilarityTransform()
            tform.estimate(dst, src)
            M = tform.params[0:2, :]

            # Warp image
            warped = cv2.warpAffine(image, M, (112, 112), borderValue=0.0)
            return warped

        except Exception as e:
            logger.error(f"Error during face alignment: {str(e)}")
            raise
