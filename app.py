import os, json, time, logging
from typing import Optional, List, Dict
from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# =====================[ 설정 및 초기화 ]=====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("uvicorn.error")

REPO_ID = "DKCode9/AIOps-Llama3-8B-gguf"
MODEL_FILENAME = "Meta-Llama-3.1-8B.Q4_K_M.gguf"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
MONGO_URI = "mongodb://root:NYdrCjppRgNRdatI@121.138.215.117:27017/?authSource=admin"
SCENARIO_FILE_PATH = "aiops_scenarios.txt"
HF_TOKEN = os.getenv("HF_TOKEN")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client["metrics-db"]
mysql_col, node_col = db["mysql_metrics"], db["node_metrics"]

llm = embedding_model = retriever = None

# =====================[ 프롬프트 템플릿 ]=====================
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
    "and identify the corresponding scenario and remediation. "
    "Your response MUST be a valid JSON object with the following keys: "
    "\"analyze\", \"scenario_id\", and \"solution\". "
    "For 'scenario_id', output ONLY the numeric ID (e.g., '1.1'). "
    "Do not include any extra text outside the JSON format."
)

# =====================[ 데이터 모델 ]=====================
class ScenarioMetrics(BaseModel):
    cpu_idle: Optional[float] = None
    heap_usage_ratio: Optional[float] = None
    disk_usage_ratio: Optional[float] = None
    volume_usage_ratio: Optional[float] = None
    readonly: Optional[bool] = None
    slow_queries: Optional[float] = None
    net_recv_drop: Optional[float] = None
    net_trans_drop: Optional[float] = None
    db_conn_ratio: Optional[float] = None

# =====================[ 유틸 함수 ]=====================
def calc_ratio(num: float, den: float) -> float:
    try: return num / den if den else 0.0
    except: return 0.0

def get_latest_metric(col, name, label=None):
    q = {"metricName": name}
    if label:
        for k, v in label.items(): q[f"labels.{k}"] = v
    doc = col.find_one(q, sort=[("timestamp", -1)])
    return float(doc["value"]) if doc and "value" in doc else 0.0

def run_llm(prompt: str) -> Dict:
    """LLM 호출 및 JSON 파싱"""
    alpaca_prompt = ALPACA_PROMPT_TEMPLATE.format(STATIC_INSTRUCTION, prompt, "")
    output = llm(alpaca_prompt, max_tokens=256, temperature=0.7, top_p=0.9)
    text = output["choices"][0]["text"].strip()
    try:
        parsed = json.loads(text)
        return {
            "scenario_id": parsed.get("scenario_id", ""),
            "analyze": parsed.get("analyze", ""),
            "solution": parsed.get("solution", "")
        }
    except:
        return {"scenario_id": "N/A", "analyze": text, "solution": "N/A"}

# =====================[ 앱 초기화 ]=====================
def startup_event():
    global llm, embedding_model, retriever
    logger.info("🚀 모델 및 RAG 로드 중...")
    t0 = time.time()

    MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME, token=HF_TOKEN)
    llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=os.cpu_count(), n_batch=64,
                use_mmap=True, use_mlock=False)
    logger.info(f"LLM 로드 완료 ({time.time()-t0:.2f}s)")

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    if os.path.exists(SCENARIO_FILE_PATH):
        raw_text = open(SCENARIO_FILE_PATH, encoding="utf-8").read()
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        docs = splitter.create_documents([raw_text])
        retriever = FAISS.from_documents(docs, embedding_model).as_retriever(search_kwargs={"k": 1})
        logger.info(f"RAG 인덱스 완료 ({len(docs)} chunks)")
    else:
        retriever = None
        logger.warning("⚠️ aiops_scenarios.txt 파일 없음")

app = FastAPI(on_startup=[startup_event])

# =====================[ 기본 엔드포인트 ]=====================
@app.get("/health")
def health_check(): return {"status": "ok"}

@app.get("/scenario/metrics", response_model=ScenarioMetrics)
def get_scenario_metrics():
    try:
        cpu_idle = get_latest_metric(node_col, "node_cpu_seconds_total", {"mode": "idle"})
        heap_used = get_latest_metric(mysql_col, "go_memstats_heap_alloc_bytes")
        heap_max = get_latest_metric(mysql_col, "go_memstats_heap_sys_bytes")
        heap_usage_ratio = calc_ratio(heap_used, heap_max)
        disk_avail = get_latest_metric(node_col, "node_filesystem_files_free", {"mountpoint": "/"})
        disk_total = get_latest_metric(node_col, "node_filesystem_size_bytes", {"mountpoint": "/"})
        disk_usage_ratio = 1 - calc_ratio(disk_avail, disk_total)
        volume_used = get_latest_metric(node_col, "node_filesystem_free_bytes")
        volume_cap = get_latest_metric(node_col, "node_filesystem_size_bytes")
        volume_usage_ratio = 1 - calc_ratio(volume_used, volume_cap)
        readonly = get_latest_metric(node_col, "node_filesystem_readonly")
        slow_q = get_latest_metric(mysql_col, "mysql_global_status_slow_queries")
        net_recv = get_latest_metric(node_col, "node_netstat_TcpExt_TCPRcvQDrop")
        net_trans = get_latest_metric(node_col, "node_netstat_TcpExt_TCPOFOQueue")
        db_conn = get_latest_metric(mysql_col, "mysql_global_status_threads_connected")
        db_max = get_latest_metric(mysql_col, "mysql_global_status_max_used_connections")
        db_ratio = calc_ratio(db_conn, db_max)
        return ScenarioMetrics(cpu_idle=cpu_idle, heap_usage_ratio=heap_usage_ratio,
                               disk_usage_ratio=disk_usage_ratio, volume_usage_ratio=volume_usage_ratio,
                               readonly=bool(readonly), slow_queries=slow_q,
                               net_recv_drop=net_recv, net_trans_drop=net_trans, db_conn_ratio=db_ratio)
    except Exception as e:
        logger.error(f"[ERROR] metrics 조회 실패: {e}", exc_info=True)
        return ScenarioMetrics()

# ==========================================
#LLM 텍스트 생성
# ==========================================
class GenerationRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(req: GenerationRequest):
    if llm is None: return {"error": "Model not loaded"}
    t0 = time.time()
    query = req.prompt
    ctx = ""
    if retriever:
        docs = retriever.get_relevant_documents(query)
        if docs: ctx = "\n\n".join(d.page_content for d in docs)
    result = run_llm(f"{query}\n\n[Related Info]\n{ctx}")
    result["elapsed_time_s"] = round(time.time()-t0, 2)
    return result

# ==========================================
#[ 시나리오 기반 진단 (DB) ]
# ==========================================
@app.post("/generate_scenario")
def generate_scenario():
    metrics = get_scenario_metrics()
    return _generate_scenario_core(metrics)

# ==========================================
#시나리오 기반 진단 (입력값)
# ==========================================
@app.post("/generate_scenario_test")
def generate_scenario_test(metrics: ScenarioMetrics):
    return _generate_scenario_core(metrics)

# =====================[ 공통 시나리오 처리 함수 ]=====================
def _generate_scenario_core(metrics: ScenarioMetrics):
    if llm is None:
        return {"error": "Model not loaded"}
    t0 = time.time()

    #조건 매핑
    conditions = [
        ("1.1", metrics.cpu_idle and metrics.cpu_idle > 0.9,
         f'CPU usage exceeds 90% ("node_cpu_seconds_total"={metrics.cpu_idle:.2f}).'),
        ("1.3", metrics.heap_usage_ratio and metrics.heap_usage_ratio > 0.9,
         f'JVM heap usage is near limit ("heap_usage_ratio"={metrics.heap_usage_ratio:.2f}).'),
        ("2.1", metrics.disk_usage_ratio and metrics.disk_usage_ratio > 0.85,
         f'Root filesystem nearly full ("disk_usage_ratio"={metrics.disk_usage_ratio:.2f}).'),
        ("2.2", metrics.volume_usage_ratio and metrics.volume_usage_ratio > 0.9,
         f'PV for MySQL almost full ("volume_usage_ratio"={metrics.volume_usage_ratio:.2f}).'),
        ("2.4", metrics.readonly, 'Filesystem remounted read-only.'),
        ("2.5", metrics.slow_queries and metrics.slow_queries > 0,
         f'MySQL slow queries detected ("slow_queries"={metrics.slow_queries}).'),
        ("3.2", (metrics.net_recv_drop and metrics.net_recv_drop > 10) or
                (metrics.net_trans_drop and metrics.net_trans_drop > 10),
         f'Network packet drops detected (recv={metrics.net_recv_drop}, trans={metrics.net_trans_drop}).'),
        ("3.5", metrics.db_conn_ratio and metrics.db_conn_ratio > 0.85,
         f'DB connection pool saturation ("db_conn_ratio"={metrics.db_conn_ratio:.2f}).')
    ]

    #가장 심각한 시나리오 한개 선택
    for sid, cond, desc in conditions:
        if cond:
            prompt = f"{desc}\nWhat is the analysis, solution, and scenario ID for this issue?"
            result = run_llm(prompt)
            result["scenario_id"] = sid
            elapsed = round(time.time() - t0, 2)
            logger.info(f"시나리오 {sid} 분석 완료 ({elapsed}s)")
            return {"detected_scenario": sid, "result": result, "elapsed_time_s": elapsed}

    return {"message": "No anomalies detected."}
