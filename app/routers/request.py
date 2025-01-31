from fastapi import APIRouter, Depends

from app.dto.request import PredictionRequest
from app.ai_processor import AIQueryProcessor

api_router = APIRouter()

@api_router.post("/request")
async def request(
                data: PredictionRequest,
                service: AIQueryProcessor = Depends(AIQueryProcessor)
):
    return await service.process_request(data)