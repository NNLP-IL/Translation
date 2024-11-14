from fastapi import APIRouter, HTTPException
from ..objects.content import InputContent
from app.objects.detection import LanguageDetection
from app.handler.detection.detection_handler import DetectionHandler

router = APIRouter(
    tags=['Detection']
)

detection_handler = DetectionHandler()

@router.post(path='', response_model=dict)
def detect_language(content: InputContent):
    response = LanguageDetection(language=detection_handler.detect_language(words=content.content), words=content.content)
    if response:
        return response.model_dump()
    raise HTTPException(status_code=404, detail="Not found language")
