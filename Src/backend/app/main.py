from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import time
from services.face_verification_service import FaceVerificationService
from services.vector_store.vector_store_service import VectorStoreService
from app.models.api_models import (
    VerificationResponse, VerificationRequest,
    RegisterFaceRequest, RegisterFaceResponse, PersonInfo,
    RecognizeFaceRequest, RecognizeFaceResponse,
    RemovePersonResponse, PersonResponse,
    ListPersonsResponse, DatabaseStatsResponse,
    VerifyWithDatabaseRequest, VerifyWithDatabaseResponse
)
import json

app = FastAPI(
    title="Face Verification & Recognition API",
    description="Face verification and recognition system using GhostFaceNet with AdaFace fine-tuning and FAISS vector database",
    version="1.0.0"
)

# CORS middleware - Allow all localhost origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend runs on port 3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
face_service = FaceVerificationService()
vector_store_service = VectorStoreService(face_service=face_service)

@app.post("/verify_faces", response_model=VerificationResponse)
async def verify_faces(
    file_a: UploadFile = File(..., description="First face image"),
    file_b: UploadFile = File(..., description="Second face image")
):
    """
    Verify if two face images belong to the same person.

    - **file_a**: First face image file (JPEG/PNG)
    - **file_b**: Second face image file (JPEG/PNG)

    Returns similarity score and prediction based on threshold.
    """
    try:
        # Validate file types
        allowed_types = ["image/jpeg", "image/png", "image/jpg"]
        if file_a.content_type not in allowed_types:
            raise HTTPException(400, f"File A: Unsupported file type {file_a.content_type}")
        if file_b.content_type not in allowed_types:
            raise HTTPException(400, f"File B: Unsupported file type {file_b.content_type}")

        # Read file contents
        image_a_bytes = await file_a.read()
        image_b_bytes = await file_b.read()

        # Perform verification
        start_time = time.time()
        result = face_service.verify_faces(image_a_bytes, image_b_bytes)
        inference_time = time.time() - start_time

        # Add inference time to result
        result.inference_time = round(inference_time * 1000, 2)  # Convert to ms

        return result

    except Exception as e:
        raise HTTPException(500, f"Verification failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": face_service.is_model_loaded(),
        "vector_store_initialized": True
    }

# Vector Store Endpoints

@app.post("/register_face", response_model=RegisterFaceResponse)
async def register_face(
    name: str = Form(..., description="Person name"),
    description: str = Form("", description="Person description"),
    age: int = Form(None, description="Person age"),
    additional_info: str = Form("{}", description="Additional person information as JSON string"),
    file: UploadFile = File(..., description="Face image file")
):
    """
    Register a new face in the database.

    - **name**: Person name (required)
    - **description**: Person description (optional)
    - **age**: Person age (optional)
    - **additional_info**: Additional person information as JSON string (optional)
    - **file**: Face image file (JPEG/PNG)
    """
    try:
        # Validate file type
        allowed_types = ["image/jpeg", "image/png", "image/jpg"]
        if file.content_type not in allowed_types:
            raise HTTPException(400, f"Unsupported file type {file.content_type}")

        # Validate required name
        if not name or name.strip() == "":
            raise HTTPException(400, "Name is required and cannot be empty")

        # Read file content
        image_bytes = await file.read()

        # Parse additional_info JSON
        try:
            additional_info_dict = json.loads(additional_info) if additional_info else {}
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"Invalid JSON format for additional_info: {str(e)}")

        # Prepare person info
        person_info = {
            "name": name.strip(),
            "description": description.strip(),
            "additional_info": {
                "age": age,
                **additional_info_dict
            }
        }

        # Register face
        result = vector_store_service.register_face(image_bytes, person_info)

        if not result['success']:
            raise HTTPException(400, result['error'])

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Registration failed: {str(e)}")

@app.post("/recognize_face", response_model=RecognizeFaceResponse)
async def recognize_face(
    file: UploadFile = File(..., description="Face image file"),
    top_k: int = Query(5, description="Number of top matches to return"),
    threshold: float = Query(0.25, description="Minimum similarity threshold")
):
    """
    Recognize a face by searching in the database.

    - **file**: Query face image file (JPEG/PNG)
    - **top_k**: Number of top matches to return
    - **threshold**: Minimum similarity threshold (0.0-1.0)
    """
    try:
        # Validate file type
        allowed_types = ["image/jpeg", "image/png", "image/jpg"]
        if file.content_type not in allowed_types:
            raise HTTPException(400, f"Unsupported file type {file.content_type}")

        # Read file content
        image_bytes = await file.read()

        # Recognize face
        result = vector_store_service.recognize_face(image_bytes, top_k=top_k, threshold=threshold)

        if not result['success']:
            raise HTTPException(400, result['error'])

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Recognition failed: {str(e)}")

@app.delete("/persons/{person_id}", response_model=RemovePersonResponse)
async def remove_person(person_id: int):
    """
    Remove a person from the database.

    - **person_id**: ID of the person to remove
    """
    try:
        result = vector_store_service.remove_person(person_id)

        if not result['success']:
            raise HTTPException(404, result['error'])

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Removal failed: {str(e)}")

@app.get("/persons/{person_id}", response_model=PersonResponse)
async def get_person(person_id: int):
    """
    Get information about a registered person.

    - **person_id**: ID of the person
    """
    try:
        result = vector_store_service.get_person_info(person_id)

        if not result['success']:
            raise HTTPException(404, result['error'])

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to get person info: {str(e)}")

@app.get("/persons", response_model=ListPersonsResponse)
async def list_persons(
    limit: int = Query(50, description="Maximum number of results"),
    offset: int = Query(0, description="Number of results to skip")
):
    """
    List all registered persons.

    - **limit**: Maximum number of persons to return
    - **offset**: Number of persons to skip
    """
    try:
        result = vector_store_service.list_registered_persons(limit=limit, offset=offset)
        return result

    except Exception as e:
        raise HTTPException(500, f"Failed to list persons: {str(e)}")

@app.get("/database/stats", response_model=DatabaseStatsResponse)
async def get_database_stats():
    """Get statistics about the face database"""
    try:
        result = vector_store_service.get_database_stats()
        return result

    except Exception as e:
        raise HTTPException(500, f"Failed to get database stats: {str(e)}")

@app.post("/verify_faces_with_db", response_model=VerifyWithDatabaseResponse)
async def verify_faces_with_database(
    file_a: UploadFile = File(..., description="First face image"),
    file_b: UploadFile = File(..., description="Second face image"),
    threshold: float = Query(0.25, description="Similarity threshold")
):
    """
    Verify if two faces belong to the same person and check against database.

    - **file_a**: First face image file (JPEG/PNG)
    - **file_b**: Second face image file (JPEG/PNG)
    - **threshold**: Similarity threshold for database matching
    """
    try:
        # Validate file types
        allowed_types = ["image/jpeg", "image/png", "image/jpg"]
        if file_a.content_type not in allowed_types:
            raise HTTPException(400, f"File A: Unsupported file type {file_a.content_type}")
        if file_b.content_type not in allowed_types:
            raise HTTPException(400, f"File B: Unsupported file type {file_b.content_type}")

        # Read file contents
        image_a_bytes = await file_a.read()
        image_b_bytes = await file_b.read()

        # Verify with database
        result = vector_store_service.verify_faces_with_database(
            image_a_bytes, image_b_bytes, threshold=threshold
        )

        if not result['success']:
            raise HTTPException(400, result['error'])

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Verification failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Face Verification & Recognition API",
        "model": "GhostFaceNet + AdaFace",
        "vector_database": "FAISS",
        "endpoints": {
            "POST /verify_faces": "Verify two face images",
            "POST /register_face": "Register new face in database",
            "POST /recognize_face": "Recognize face from database",
            "GET /persons": "List registered persons",
            "GET /persons/{id}": "Get person information",
            "DELETE /persons/{id}": "Remove person from database",
            "GET /database/stats": "Get database statistics",
            "POST /verify_faces_with_db": "Verify faces with database check",
            "GET /health": "Health check"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )