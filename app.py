# ============================================================
# app_fast.py — Optimized FastAPI RAG Server for AIOps Llama3
# ============================================================

import logging
import time
import os
from typing import List, Dict

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# ============================================================
# 초기 환경 설정
# ============================================================
load_dotenv()

# Hugging Face 캐시 경로 지정 (속도 향상)
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"
os.makedirs("./hf_cache", exist_ok=True)

logger = logging.getLogger("uvicorn.error")

# ============================================================
# 환경 변수 / DB 설정
# ============================================================
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    logger.warning("⚠️ HF_TOKEN environment variable not found — gated model access may fail.")

MONGO_URI = "mongodb://root:NYdrCjppRgNRdatI@121.138.215.117:27017/?authSource=admin"
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client["metrics_db"]
mysql_col = db["mysql_metrics"]
node_col = db["node_metrics"]

# ============================================================
# 모델 설정
# ============================================================
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
ADAPTER_MODEL_NAME = "DKCode9/AIOps-peft-Llama3-8B-v1"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
SCENARIO_FILE_PATH = "aiops_scenarios.txt"

# 전역 객체
tokenizer = None
model = None
embedding_model = None
scenario_texts = []
scenario_embeddings = None

# ============================================================
# Alpaca 프롬프트 템플릿
# ============================================================
ALPACA_PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

STATIC_INSTRUCTION = (
    "You are an AI SRE specializing in Kubernetes infrastructure. "
    "Analyze why this situation occurred, describe automated actions, "
    "and identify the corresponding scenario and remediation."
)

# ============================================================
# 모델 로드 함수
# ============================================================
def load_models():
    global tokenizer, model, embedding_model, scenario_texts, scenario_embeddings

    logger.info("🚀 Loading all models and RAG data...")
    start_time = time.time()

    # 4bit 양자화 설정
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True
    )

    # Base Llama 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        token=HF_TOKEN,
        trust_remote_code=True,
        quantization_config=quant_config
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=HF_TOKEN)
    logger.info(f"✅ Base model loaded: {BASE_MODEL_NAME}")

    # LoRA 어댑터 로드 (병합 안 함)
    model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_NAME, token=HF_TOKEN)
    logger.info(f"✅ LoRA adapter loaded: {ADAPTER_MODEL_NAME}")

    # 임베딩 모델 로드
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    logger.info(f"✅ Embedding model loaded: {EMBEDDING_MODEL_NAME}")

    # RAG 파일 로드
    if not os.path.exists(SCENARIO_FILE_PATH):
        logger.error(f"❌ Scenario file not found: {SCENARIO_FILE_PATH}")
    else:
        with open(SCENARIO_FILE_PATH, "r", encoding="utf-8") as f:
            scenario_texts[:] = [line.strip() for line in f if line.strip()]
        scenario_embeddings = embedding_model.encode(scenario_texts, convert_to_tensor=False)
        logger.info(f"✅ Loaded and encoded {len(scenario_texts)} scenarios.")

    logger.info(f"✅ Total load time: {time.time() - start_time:.2f}s")

# ============================================================
# RAG 검색
# ============================================================
def retrieve_context(query: str, top_k: int = 3):
    if not scenario_embeddings or not scenario_texts:
        return "No context available.", 0

    query_vec = embedding_model.encode([query], convert_to_tensor=False)
    similarities = cosine_similarity(query_vec, scenario_embeddings)
    top_indices = np.argsort(similarities[0])[-top_k:][::-1]
    retrieved_docs = [scenario_texts[i] for i in top_indices]
    return "\n".join(retrieved_docs), len(scenario_texts)

# ============================================================
# FastAPI 초기화
# ============================================================
app = FastAPI()

@app.on_event("startup")
def on_startup():
    load_models()

# ============================================================
# 요청 모델
# ============================================================
class GenerationRequest(BaseModel):
    prompt: str

# ============================================================
# 엔드포인트
# ============================================================
@app.post("/generate")
def generate_text(request: GenerationRequest):
    if not all([model, tokenizer, embedding_model]):
        return JSONResponse({"error": "Models not ready"}, status_code=500)

    try:
        start = time.time()
        context, _ = retrieve_context(request.prompt)
        prompt = ALPACA_PROMPT_TEMPLATE.format(
            STATIC_INSTRUCTION,
            f"Problem:\n{request.prompt}\n\nContext:\n{context}",
            ""
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 스트리밍 제거 → 즉시 결과 생성
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            use_cache=True
        )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated in {time.time() - start:.2f}s")

        return {"response": text.strip()}

    except Exception as e:
        logger.error(f"Error during generation: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# ============================================================
# Health check
# ============================================================
@app.get("/health")
def health_check():
    ok = all([model, tokenizer, embedding_model, scenario_embeddings is not None])
    return {"status": "ok" if ok else "error"}

# ============================================================
# 비동기 MongoDB 예시
# ============================================================
class MetricDocumentResponse(BaseModel):
    id: str
    metricName: str
    labels: Dict[str, str]
    value: float
    timestamp: int


@app.get("/metrics/node/filesystem_free", response_model=List[MetricDocumentResponse])
async def get_node_metrics(
    mountpoint: str = Query(...),
    limit: int = Query(10)
):
    query = {"metricName": "node_filesystem_free_bytes", "labels.mountpoint": mountpoint}
    docs = await node_col.find(query).sort("timestamp", -1).to_list(limit)
    return [
        MetricDocumentResponse(
            id=str(doc["_id"]),
            metricName=doc["metricName"],
            labels=doc["labels"],
            value=doc["value"],
            timestamp=doc["timestamp"]
        )
        for doc in docs
    ]

@app.get("/metrics/mysql/commands_total", response_model=List[MetricDocumentResponse])
async def get_mysql_metrics(
    command: str = Query(...),
    limit: int = Query(10)
):
    query = {"metricName": "mysql_global_status_commands_total", "labels.command": command}
    docs = await mysql_col.find(query).sort("timestamp", -1).to_list(limit)
    return [
        MetricDocumentResponse(
            id=str(doc["_id"]),
            metricName=doc["metricName"],
            labels=doc["labels"],
            value=doc["value"],
            timestamp=doc["timestamp"]
        )
        for doc in docs
    ]
