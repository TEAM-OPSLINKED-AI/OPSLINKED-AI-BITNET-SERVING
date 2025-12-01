import os, json, time, logging
from typing import Optional, Dict
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
mysql_col, node_col, actuator_col = db["mysql_metrics"], db["node_metrics"], db["actuator_metrics"]

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
    cpu_usage_5m: Optional[float] = None
    heap_usage_ratio: Optional[float] = None
    disk_usage_ratio: Optional[float] = None
    volume_usage_ratio: Optional[float] = None
    readonly: Optional[bool] = None
    slow_queries: Optional[float] = None
    net_recv_drop: Optional[float] = None
    net_trans_drop: Optional[float] = None
    db_conn_ratio: Optional[float] = None
    http_5xx_ratio: Optional[float] = None
    http_client_rps_5m: Optional[float] = None
    http_client_avg_5m: Optional[float] = None
    http_server_rps_5m: Optional[float] = None
    http_server_rps_1h: Optional[float] = None

# =====================[ 유틸 함수 ]=====================
def calc_ratio(num: float, den: float) -> float:
    try:
        return num / den if den else 0.0
    except:
        return 0.0

def get_latest_metric(col, name, label=None):
    q = {"metricName": name}
    if label:
        for k, v in label.items():
            q[f"labels.{k}"] = v
    doc = col.find_one(q, sort=[("timestamp", -1)])
    return float(doc["value"]) if doc and "value" in doc else 0.0

def get_docs_5m(col, name, label=None, limit: int = 5):
    """
    최근 limit개 문서를 timestamp 역순으로 가져온 뒤
    오래된 순으로 정렬해서 반환.
    (1분마다 하나씩 수집되므로 5개 ≒ 최근 5분)
    """
    q = {"metricName": name}
    if label:
        for k, v in label.items():
            q[f"labels.{k}"] = v
    docs = list(col.find(q).sort("timestamp", -1).limit(limit))
    # 오래된 순으로 다시 정렬
    docs.reverse()
    return docs

def calc_rate_over_docs(docs):
    """
    누적 counter 타입에 대한 평균 rate 계산용.
    지금 CPU idle은 이미 '비율'로 들어오므로 1.1에서는 사용하지 않음.
    """
    if len(docs) < 2:
        return 0.0
    first, last = docs[0], docs[-1]
    dv = last["value"] - first["value"]
    dt = (last["timestamp"] - first["timestamp"]) / 1000
    if dt <= 0:
        return 0.0
    return dv / dt

def run_llm(prompt: str) -> Dict:
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
    logger.info("모델 및 RAG 로드 중...")
    t0 = time.time()

    MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME, token=HF_TOKEN)
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=os.cpu_count(),
        n_batch=64,
        use_mmap=True,
        use_mlock=False
    )
    logger.info(f"LLM 로드 완료 ({time.time()-t0:.2f}s)")

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    if os.path.exists(SCENARIO_FILE_PATH):
        raw_text = open(SCENARIO_FILE_PATH, encoding="utf-8").read()
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        docs = splitter.create_documents([raw_text])
        retriever_local = FAISS.from_documents(docs, embedding_model).as_retriever(search_kwargs={"k": 1})
        globals()["retriever"] = retriever_local
        logger.info(f"RAG 인덱스 완료 ({len(docs)} chunks)")
    else:
        globals()["retriever"] = None
        logger.warning("aiops_scenarios.txt 파일 없음")

app = FastAPI(on_startup=[startup_event])

# =====================[ 기본 엔드포인트 ]=====================
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/scenario/metrics", response_model=ScenarioMetrics)
def get_scenario_metrics():
    try:
        logger.info("===== [SCENARIO METRICS START] =====")

        # 1.1 CPU Idle 및 Usage 계산 (최근 5개 idle 비율 기준)
        idle_docs = get_docs_5m(
            node_col,
            "node_cpu_seconds_total",
            {"mode": "idle"},
            limit=5,
        )

        logger.info(f"[1.1 RAW] idle_docs_count={len(idle_docs)}")
        logger.info(f"[1.1 RAW] idle_docs={idle_docs}")

        idle_values = [float(d["value"]) for d in idle_docs if "value" in d]
        if idle_values:
            avg_idle = sum(idle_values) / len(idle_values)
        else:
            avg_idle = 0.0

        cpu_usage_5m = 1 - avg_idle

        logger.info(f"[METRIC] 1.1 idle_values={idle_values}")
        logger.info(f"[METRIC] 1.1 avg_idle_5m={avg_idle}")
        logger.info(f"[METRIC] 1.1 cpu_usage_5m={cpu_usage_5m}")

        # 1.3 JVM Heap 사용률
        heap_used = get_latest_metric(actuator_col, "jvm_memory_used_bytes")
        heap_max = get_latest_metric(actuator_col, "jvm_memory_max_bytes")

        logger.info(f"[1.3 RAW] heap_used={heap_used}, heap_max={heap_max}")

        heap_usage_ratio = calc_ratio(heap_used, heap_max)

        logger.info(f"[METRIC] 1.3 heap_usage_ratio={heap_usage_ratio}")

        # 1.5 HTTP 5xx 비율 (2xx vs 5xx)
        success_2xx = get_latest_metric(
            actuator_col,
            "http_server_requests_seconds_count",
            {"status": "200"}
        )
        error_5xx = get_latest_metric(
            actuator_col,
            "http_server_requests_seconds_count",
            {"status": "500"}
        )

        logger.info(f"[1.5 RAW] 2xx={success_2xx}, 5xx={error_5xx}")

        total_2xx_5xx = success_2xx + error_5xx
        http_5xx_ratio = (error_5xx / total_2xx_5xx) if total_2xx_5xx > 0 else 0.0

        logger.info(f"[METRIC] 1.5 http_5xx_ratio={http_5xx_ratio}")

        # 2.1 Root 디스크 사용률
        disk_avail = get_latest_metric(
            node_col,
            "node_filesystem_files_free",
            {"mountpoint": "/"}
        )
        disk_total = get_latest_metric(
            node_col,
            "node_filesystem_size_bytes",
            {"mountpoint": "/"}
        )

        logger.info(f"[2.1 RAW] disk_avail={disk_avail}, disk_total={disk_total}")

        disk_usage_ratio = 1 - calc_ratio(disk_avail, disk_total)

        logger.info(f"[METRIC] 2.1 disk_usage_ratio={disk_usage_ratio}")

        # 2.4 Filesystem readonly 여부
        readonly = get_latest_metric(node_col, "node_filesystem_readonly")
        logger.info(f"[METRIC] 2.4 filesystem_readonly={readonly}")

        # 2.5 MySQL Slow Queries
        slow_q = get_latest_metric(mysql_col, "mysql_global_status_slow_queries")
        logger.info(f"[METRIC] 2.5 slow_queries={slow_q}")

        # 3.2 네트워크 드롭률
        net_recv = get_latest_metric(node_col, "node_network_receive_drop_total")
        net_trans = get_latest_metric(node_col, "node_network_transmit_drop_total")
        logger.info(f"[METRIC] 3.2 net_recv_drop={net_recv}, net_trans_drop={net_trans}")

        # 3.3 Client 측 RPS Spike
        http_client_rps_5m = get_latest_metric(actuator_col, "http_client_requests_seconds_rps_5m")
        http_client_avg_5m = get_latest_metric(actuator_col, "http_client_requests_seconds_avg_5m")
        logger.info(f"[METRIC] 3.3 client_rps_5m={http_client_rps_5m}, avg_5m={http_client_avg_5m}")

        # 3.4 Server RPS Spike
        http_server_rps_5m = get_latest_metric(actuator_col, "http_server_requests_seconds_rps_5m")
        http_server_rps_1h = get_latest_metric(actuator_col, "http_server_requests_seconds_rps_1h")
        logger.info(f"[METRIC] 3.4 server_rps_5m={http_server_rps_5m}, rps_1h={http_server_rps_1h}")

        # 3.5 DB Connection Ratio
        db_conn = get_latest_metric(mysql_col, "mysql_global_status_threads_connected")
        db_max = get_latest_metric(mysql_col, "mysql_global_status_max_used_connections")
        logger.info(f"[3.5 RAW] db_conn={db_conn}, db_max={db_max}")

        db_conn_ratio = calc_ratio(db_conn, db_max)
        logger.info(f"[METRIC] 3.5 db_conn_ratio={db_conn_ratio}")

        logger.info("===== [SCENARIO METRICS END] =====")

        return ScenarioMetrics(
            cpu_usage_5m=cpu_usage_5m,
            heap_usage_ratio=heap_usage_ratio,
            disk_usage_ratio=disk_usage_ratio,
            volume_usage_ratio=0,
            readonly=bool(readonly),
            slow_queries=slow_q,
            net_recv_drop=net_recv,
            net_trans_drop=net_trans,
            db_conn_ratio=db_conn_ratio,
            http_5xx_ratio=http_5xx_ratio,
            http_client_rps_5m=http_client_rps_5m,
            http_client_avg_5m=http_client_avg_5m,
            http_server_rps_5m=http_server_rps_5m,
            http_server_rps_1h=http_server_rps_1h
        )

    except Exception as e:
        logger.error(f"[ERROR] metrics 조회 실패: {e}", exc_info=True)
        return ScenarioMetrics()

# ==========================================
# LLM 텍스트 생성
# ==========================================
class GenerationRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(req: GenerationRequest):
    if llm is None:
        return {"error": "Model not loaded"}
    t0 = time.time()
    query = req.prompt

    ctx = ""
    if retriever:
        docs = retriever.get_relevant_documents(query)
        if docs:
            ctx = "\n\n".join([d.page_content for d in docs])

    result = run_llm(f"{query}\n\n[Related Info]\n{ctx}")
    result["elapsed_time_s"] = round(time.time() - t0, 2)
    return result

# ==========================================
# 시나리오 기반 진단 (DB)
# ==========================================
@app.post("/generate_scenario")
def generate_scenario():
    metrics = get_scenario_metrics()
    return generate_scenario_core(metrics)

# ==========================================
# 시나리오 기반 진단 (입력값)
# ==========================================
@app.post("/generate_scenario_test")
def generate_scenario_test(metrics: ScenarioMetrics):
    return generate_scenario_core(metrics)

# =====================[ 공통 시나리오 판단 + LLM 호출 ]=====================
def generate_scenario_core(metrics: ScenarioMetrics):
    if llm is None:
        return {"error": "Model not loaded"}

    t0 = time.time()

    conditions = [
        ("1.1", metrics.cpu_usage_5m and metrics.cpu_usage_5m > 0.9,
         f'CPU usage exceeds 90% (5m avg usage={metrics.cpu_usage_5m:.2f}).'),

        ("1.3", metrics.heap_usage_ratio and metrics.heap_usage_ratio > 0.9,
         f'JVM heap usage is near limit ({metrics.heap_usage_ratio:.2f}).'),

        ("1.5", metrics.http_5xx_ratio and metrics.http_5xx_ratio > 0.1,
         f'HTTP 5xx error rate > 10% ({metrics.http_5xx_ratio:.2f}).'),

        ("2.1", metrics.disk_usage_ratio and metrics.disk_usage_ratio > 0.85,
         f'Root filesystem nearly full ({metrics.disk_usage_ratio:.2f}).'),

        ("2.4", metrics.readonly,
         'Filesystem remounted read-only.'),

        ("2.5", metrics.slow_queries and metrics.slow_queries > 0,
         f'MySQL slow queries detected ({metrics.slow_queries}).'),

        ("3.2", (metrics.net_recv_drop and metrics.net_recv_drop > 10) or
                (metrics.net_trans_drop and metrics.net_trans_drop > 10),
         f'Network packet drops detected (recv={metrics.net_recv_drop}, trans={metrics.net_trans_drop}).'),

        ("3.3", metrics.http_client_rps_5m and metrics.http_client_avg_5m and
                (metrics.http_client_rps_5m > metrics.http_client_avg_5m * 1.5),
         f'Client request spike (rps_5m={metrics.http_client_rps_5m}, avg={metrics.http_client_avg_5m}).'),

        ("3.4", metrics.http_server_rps_5m and metrics.http_server_rps_1h and
                (metrics.http_server_rps_5m > metrics.http_server_rps_1h * 2.0),
         f'Server RPS spike (5m={metrics.http_server_rps_5m}, 1h={metrics.http_server_rps_1h}).'),

        ("3.5", metrics.db_conn_ratio and metrics.db_conn_ratio > 0.85,
         f'DB connection pool saturation ({metrics.db_conn_ratio:.2f}).')
    ]

    for sid, cond, desc in conditions:
        if cond:
            result = run_llm(desc)
            result["scenario_id"] = sid
            elapsed = round(time.time() - t0, 2)
            logger.info(f"시나리오 {sid} 분석 완료 ({elapsed}s)")
            return {"detected_scenario": sid, "result": result, "elapsed_time_s": elapsed}

    return {"message": "No anomalies detected."}
