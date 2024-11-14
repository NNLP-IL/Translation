from fastapi import APIRouter, HTTPException
import os
from typing import Optional
from ..objects.content import InputContent
from app.shared.shared import get_transliterate_service
from ..objects.language import Language

router = APIRouter(
    prefix="/transliterate",
    tags=["Transliteration"]
)

service = get_transliterate_service()

@router.post(path="")
async def get_transliteration_from_body(content: InputContent, source: Optional[Language] = None, target: Optional[Language] = None):
    try:
        res: str = service.transliterate(content.content, src=source, tgt=target)
        if res: 
            return res
        raise HTTPException(status_code=404, detail="Not found transliterate")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))