from fastapi import APIRouter
from typing import List
from ..objects.entities import Sentence
from ..shared.shared import entity_extract_services
from ..objects.forms import EntityExtractContent, EntityExtractLanguage

router = APIRouter(
    prefix="/entity_extract",
    tags=["Named Entity Recognition"]
)

@router.post(path="")
async def extract_entities(content: EntityExtractContent) -> List[Sentence]: 
    if content.language.name == EntityExtractLanguage.HE.name:
        res: List[Sentence] = entity_extract_services.hebrew_extract.get_entities(text=content.text, split_text=content.split_sentences) 
    else:
        res: List[Sentence] = entity_extract_services.arabic_extract.get_entities(text=content.text, split_text=content.split_sentences)    
    return res

@router.get(path="/tagging_map")
async def extract_entities() -> dict: 
    return entity_extract_services.arabic_extract.tag_map