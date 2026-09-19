from fastapi import APIRouter
from pydantic import BaseModel
from backend.diet_rag_service.rag_engine import GroqDietRAGEngine
from backend.shared.storage import save_json_artifact

router = APIRouter(prefix="/api/diet", tags=["Diet RAG"])
rag_engine = GroqDietRAGEngine()

class DietPlanRequest(BaseModel):
    patient_id: str = "PAT-DEMO123"
    deficiency_type: str = "iron"
    diet_preference: str = "vegetarian"

class ChatQueryRequest(BaseModel):
    query: str
    deficiency_type: str = "iron"

@router.post("/generate-plan")
def generate_diet_plan(req: DietPlanRequest):
    plan = rag_engine.generate_diet_plan(req.deficiency_type, req.diet_preference)
    save_json_artifact("diet_plans", req.patient_id, plan)
    return {"status": "success", "data": plan}

@router.post("/chat")
def rag_chat(req: ChatQueryRequest):
    answer = rag_engine.answer_query(req.query, req.deficiency_type)
    return {"status": "success", "answer": answer}
