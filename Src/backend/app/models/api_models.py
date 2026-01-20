from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class VerificationRequest(BaseModel):
    """Request model for face verification (for future use if needed)"""
    pass

class VerificationResponse(BaseModel):
    """Response model for face verification results"""
    similarity: float
    is_same_person: bool
    threshold: float
    inference_time: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "similarity": 0.85,
                "is_same_person": True,
                "threshold": 0.25,
                "inference_time": 45.2
            }
        }

# Vector Store Models

class PersonInfo(BaseModel):
    """Person information for registration"""
    name: str
    description: Optional[str] = ""
    additional_info: Optional[Dict[str, Any]] = {}

    class Config:
        extra = "allow"  # Allow additional fields for flexibility

class RegisterFaceRequest(BaseModel):
    """Request model for face registration"""
    person_info: PersonInfo

class RegisterFaceResponse(BaseModel):
    """Response model for face registration"""
    success: bool
    person_id: Optional[int] = None
    name: Optional[str] = None
    message: str
    registered_at: Optional[str] = None
    error: Optional[str] = None

class RecognizeFaceRequest(BaseModel):
    """Request model for face recognition"""
    top_k: Optional[int] = 5
    threshold: Optional[float] = 0.25

class MatchResult(BaseModel):
    """Individual match result"""
    person_id: int
    name: str
    description: str
    similarity: float
    additional_info: Dict[str, Any]
    registered_at: str

class RecognizeFaceResponse(BaseModel):
    """Response model for face recognition"""
    success: bool
    recognized: bool
    matches: List[MatchResult]
    top_match: Optional[MatchResult] = None
    confidence: float = 0.0
    message: str
    error: Optional[str] = None

class RemovePersonResponse(BaseModel):
    """Response model for person removal"""
    success: bool
    person_id: int
    name: Optional[str] = None
    message: str
    error: Optional[str] = None

class PersonResponse(BaseModel):
    """Response model for person information"""
    success: bool
    person: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    person_id: Optional[int] = None

class ListPersonsResponse(BaseModel):
    """Response model for listing persons"""
    success: bool
    persons: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int
    error: Optional[str] = None

class DatabaseStatsResponse(BaseModel):
    """Response model for database statistics"""
    success: bool
    stats: Dict[str, Any]
    is_model_loaded: bool
    error: Optional[str] = None

class VerifyWithDatabaseRequest(BaseModel):
    """Request model for verification with database check"""
    threshold: Optional[float] = 0.25

class VerifyWithDatabaseResponse(BaseModel):
    """Response model for verification with database check"""
    success: bool
    face_verification: Dict[str, Any]
    database_check: Dict[str, Any]
    message: str
    error: Optional[str] = None