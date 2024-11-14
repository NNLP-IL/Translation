from fastapi import APIRouter, Request, HTTPException
from typing import AsyncGenerator, List
import torch
from fastapi.responses import StreamingResponse
from ..objects.forms import TranslateContent
from ..objects.entities import Sentence
from ..consts import DEVICE
from ..handler.objects.models import TranslateModels



router = APIRouter(
    prefix="/translator",
    tags=["Translator"]
)

# Set device to CUDA if available, otherwise use CPU
device = 'cuda' if DEVICE == 'cuda' and torch.cuda.is_available() else 'cpu'
translate_models = TranslateModels(device=device)


@router.post(path="/translate_sentences")
async def translate_sentences(translate_content: TranslateContent, request: Request) -> List[Sentence]: 
    async def event_generator() -> AsyncGenerator[str, None]: 
        async for translated_sentence in translate_models.translator(translate_direction=translate_content.translate_direction).translate(
                    src_sentences=translate_content.sentences, entity_align=translate_content.entity_alignment):
            if await request.is_disconnected():
                break
            yield f"data: {translated_sentence.model_dump_json()}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

