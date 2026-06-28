"""
Cassava Disease Detector — FastAPI endpoints
"""

import os
import io
import json
import tempfile
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import CassavaPipeline

FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), "data", "feedback_log.jsonl")

# ── App lifecycle ─────────────────────────────────────────────────────────────

pipeline: CassavaPipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("Loading models...")
    pipeline = CassavaPipeline()
    pipeline._load_image_model()
    pipeline._load_llm()
    print("Warming advice cache (pre-generating advice for all 5 diseases)...")
    pipeline.warm_advice_cache()
    print("All models and cache ready.")
    yield
    print("Shutting down.")

app = FastAPI(
    title="Cassava Disease Detector API",
    description="Detect cassava leaf diseases from images and get farmer advice powered by TinyLlama + DuckDuckGo.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Response schemas ──────────────────────────────────────────────────────────

class Prediction(BaseModel):
    name: str
    confidence: float

class WebResult(BaseModel):
    title: str
    snippet: str
    url: str

class ClassifyResponse(BaseModel):
    disease: Optional[str] = None
    confidence: float = 0.0
    is_cassava: bool = True
    top5: list[Prediction] = []
    error: Optional[str] = None

class AnalyzeResponse(BaseModel):
    disease: Optional[str] = None
    confidence: float = 0.0
    is_cassava: bool = True
    top5: list[Prediction] = []
    advice: Optional[dict[str, str]] = None
    web_results: list[WebResult] = []
    error: Optional[str] = None

class AdviseResponse(BaseModel):
    disease: str
    answer: str

# ── Helpers ───────────────────────────────────────────────────────────────────

def save_upload(file: UploadFile) -> str:
    suffix = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file.file.read())
    tmp.close()
    return tmp.name

def parse_advice(raw: str) -> dict[str, str]:
    """Convert 'Section: text\n\nSection: text' into a dict."""
    sections = {}
    for block in raw.strip().split("\n\n"):
        if ": " in block:
            key, _, val = block.partition(": ")
            sections[key.strip()] = val.strip()
        else:
            sections.setdefault("Info", block.strip())
    return sections

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Check API is alive and models are loaded."""
    return {"status": "ok", "models_loaded": pipeline is not None}


@app.post("/classify", response_model=ClassifyResponse, summary="Classify a cassava leaf image")
async def classify(file: UploadFile = File(..., description="Cassava leaf image (jpg/png)")):
    """
    Upload a cassava leaf image and get the disease classification with confidence scores.
    No LLM or web search — fast response.
    """
    path = save_upload(file)
    try:
        if not pipeline.is_leaf_image(path):
            return ClassifyResponse(
                disease=None, confidence=0.0, is_cassava=False, top5=[],
                error="This image does not appear to be a cassava leaf. Please upload a clear photo of a cassava leaf.",
            )
        result = pipeline.classify(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(path)

    return ClassifyResponse(
        disease=result["display"],
        confidence=result["confidence"],
        is_cassava=True,
        top5=[Prediction(name=n, confidence=p) for n, p in result["top5"]],
    )


@app.post("/analyze", response_model=AnalyzeResponse, summary="Full pipeline: classify + advice + web search")
async def analyze(
    file: UploadFile = File(..., description="Cassava leaf image (jpg/png)"),
    search: bool = Query(True, description="Include DuckDuckGo web results"),
):
    """
    Full pipeline: classify the image, generate structured farmer advice from the fine-tuned LLM,
    and optionally enrich with DuckDuckGo web search results.
    """
    path = save_upload(file)
    try:
        result = pipeline.run(path, search=search)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(path)

    if not result.get("is_cassava"):
        return AnalyzeResponse(
            disease=None, confidence=0.0, is_cassava=False, top5=[],
            advice=None, web_results=[],
            error=result.get("error", "This image does not appear to be a cassava leaf."),
        )

    return AnalyzeResponse(
        disease=result["disease"],
        confidence=result["confidence"],
        is_cassava=True,
        top5=[Prediction(name=n, confidence=p) for n, p in result["top5"]],
        advice=parse_advice(result["advice"]) if result.get("advice") else None,
        web_results=[WebResult(**r) for r in result.get("web_results", [])],
    )


@app.post("/advise", response_model=AdviseResponse, summary="Ask a custom question about a disease")
async def advise(
    file: UploadFile = File(..., description="Cassava leaf image (jpg/png)"),
    question: str = Query(..., description="Custom question to ask the LLM about the detected disease"),
):
    """
    Classify the leaf image then answer a custom farmer question using the fine-tuned LLM.
    """
    path = save_upload(file)
    try:
        classification = pipeline.classify(path)
        answer = pipeline.advise(classification["label"], question=question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(path)

    return AdviseResponse(disease=classification["display"], answer=answer)


@app.get("/diseases", summary="List all detectable diseases")
def list_diseases():
    """Return all disease classes the model can detect."""
    from pipeline import DISEASE_LABELS
    return {"diseases": [{"key": k, "name": v} for k, v in DISEASE_LABELS.items()]}


# ── Feedback endpoints ────────────────────────────────────────────────────────

class FeedbackPayload(BaseModel):
    predicted_disease: str
    predicted_confidence: float
    vote: str                        # 'correct' | 'wrong'
    correct_disease: Optional[str] = None
    comment: Optional[str] = None
    image_filename: Optional[str] = None


@app.post("/feedback", summary="Submit user feedback on a diagnosis")
def submit_feedback(payload: FeedbackPayload):
    """
    Store farmer feedback. Used to build a retraining dataset.
    vote='correct' confirms the prediction; vote='wrong' provides a correction.
    """
    record = payload.model_dump()
    record["timestamp"] = datetime.utcnow().isoformat()
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return {"status": "saved", "record": record}


@app.get("/feedback/stats", summary="Feedback statistics for the dashboard")
def feedback_stats():
    """Return aggregated feedback stats: total, accuracy rate, correction breakdown."""
    if not os.path.exists(FEEDBACK_FILE):
        return {"total": 0, "correct": 0, "wrong": 0, "accuracy_rate": None, "corrections": {}}
    records = []
    with open(FEEDBACK_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    total   = len(records)
    correct = sum(1 for r in records if r.get("vote") == "correct")
    wrong   = total - correct
    corrections = {}
    for r in records:
        if r.get("vote") == "wrong" and r.get("correct_disease"):
            key = f"{r['predicted_disease']} → {r['correct_disease']}"
            corrections[key] = corrections.get(key, 0) + 1
    return {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "accuracy_rate": round(correct / total, 3) if total else None,
        "corrections": corrections,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
