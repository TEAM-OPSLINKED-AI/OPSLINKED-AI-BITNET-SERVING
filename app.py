from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import logging

logger = logging.getLogger("uvicorn.error")

# -------------------------------------------------------------
# 가장 가벼운 BitNet 모델 중 하나를 선택
# -------------------------------------------------------------
MODEL_NAME = "ighoshsubho/Bitnet-SmolLM-135M"
# RAG의 검색을 위한 경량 임베딩 모델
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# -------------------------------------------------------------
# 실시간 크롤링 대상 웹사이트 목록
# AIOps 답변의 근거가 될 신뢰도 높은 기술 문서 및 블로그
# -------------------------------------------------------------
CRAWL_TARGET_URLS = [
    # Kubernetes 공식 문서 (핵심 개념 및 디버깅)
    "https://kubernetes.io/docs/concepts/overview/",
    "https://kubernetes.io/docs/concepts/workloads/pods/",
    "https://kubernetes.io/docs/concepts/services-networking/service/",
    "https://kubernetes.io/docs/tasks/debug/debug-application/debug-pods/",
    "https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/",
    # Prometheus Node Exporter (Linux 노드 메트릭)
    "https://prometheus.io/docs/guides/node-exporter/",
    # Spring Boot Actuator (애플리케이션 메트릭)
    "https://docs.spring.io/spring-boot/docs/current/reference/html/actuator.html"
]

# -------------------------------------------------------------
# 전역 변수 초기화
# -------------------------------------------------------------
tokenizer = None
model = None
embedding_model = None

# -------------------------------------------------------------
# 웹 크롤링 및 텍스트 추출 관련 함수
# -------------------------------------------------------------
def scrape_page_content(url: str) -> str:
    """단일 웹 페이지에서 주요 텍스트 콘텐츠를 추출합니다."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # HTTP 오류 시 예외 발생

        soup = BeautifulSoup(response.content, 'lxml')

        # 본문 콘텐츠를 담고 있을 가능성이 높은 태그들을 우선적으로 탐색
        main_content_selectors = ['main', 'article', '.content', '#content', '.post-content', '.td-content']
        content_area = None
        for selector in main_content_selectors:
            content_area = soup.select_one(selector)
            if content_area:
                break
        
        if not content_area:
            content_area = soup.body # 최후의 수단으로 body 전체 사용

        # 불필요한 태그 제거 (nav, footer, script, style 등)
        for tag in content_area.select('nav, footer, script, style, .sidebar, .header, .footer, .menu'):
            tag.decompose()

        # 공백이 많은 텍스트를 정리하여 반환
        return ' '.join(content_area.get_text(separator=' ', strip=True).split())
    except requests.RequestException as e:
        logger.error(f"Error fetching or parsing {url}: {e}")
        return ""

def split_text_into_chunks(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """긴 텍스트를 단어 단위로 의미 있는 청크로 분할합니다."""
    words = text.split()
    if not words:
        return []
    
    chunks = []
    current_pos = 0
    while current_pos < len(words):
        end_pos = current_pos + chunk_size
        chunk = words[current_pos:end_pos]
        chunks.append(" ".join(chunk))
        current_pos += chunk_size - overlap
        if current_pos >= len(words):
            break
        
    return chunks

# -------------------------------------------------------------
# 애플리케이션 시작 시 모델 로드
# -------------------------------------------------------------
def startup_event():
    global tokenizer, model, embedding_model

    logger.info("Application starting... Loading models.")
    load_start = time.time()

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            dtype=torch.float32, 
            device_map="auto"
        )
        logger.info(f"Successfully loaded model: '{MODEL_NAME}'")
        load_mid = time.time()
        logger.info(f"Model loading time: {load_mid - load_start:.2f} seconds")

        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
        logger.info(f"Successfully loaded embedding model: '{EMBEDDING_MODEL_NAME}'")

        load_end = time.time()
        total_load_time = load_end - load_start 
        logger.info(f"Total model loading time: {total_load_time:.2f} seconds")

    except Exception as e:
        logger.error(f"A critical error occurred while loading models: {e}")

# -------------------------------------------------------------
# FastAPI 애플리케이션 설정
# -------------------------------------------------------------
app = FastAPI(on_startup=[startup_event])

class GenerationRequest(BaseModel):
    prompt: str

# -------------------------------------------------------------
# 실시간 RAG 검색 함수
# -------------------------------------------------------------
def retrieve_context_from_web(query: str, top_k: int = 3) -> tuple[str, int]:
    if embedding_model is None:
        return "Embedding model not available.", 0
    
    logger.info(f"Starting real-time web crawl for query: '{query}'")
    
    all_chunks = []
    for url in CRAWL_TARGET_URLS:
        logger.info(f"Crawling: {url}")
        content = scrape_page_content(url)
        if content:
            chunks = split_text_into_chunks(content)
            all_chunks.extend(chunks)
        time.sleep(0.2)
    
    if not all_chunks:
        return "Failed to retrieve any content from the web.", 0

    total_chunks = len(all_chunks)
    logger.info(f"Crawling finished. Found {total_chunks} text chunks.")

    chunk_vectors = embedding_model.encode(all_chunks, show_progress_bar=False)
    query_vector = embedding_model.encode([query], show_progress_bar=False)
    
    similarities = cosine_similarity(query_vector, chunk_vectors)
    top_k_indices = np.argsort(similarities[0])[-top_k:][::-1]
    
    relevant_docs = [all_chunks[i] for i in top_k_indices]
    return " ".join(relevant_docs), total_chunks

# -------------------------------------------------------------
# API 엔드포인트
# -------------------------------------------------------------
@app.post("/generate")
def generate_text(request: GenerationRequest):
    if not all([model, tokenizer, embedding_model]):
        return {"error": "Models are not ready."}

    try:
        start_time = time.time()
        retrieved_context, chunk_count = retrieve_context_from_web(request.prompt)
        retrieval_time = time.time() - start_time
        logger.info(f"Context retrieval took {retrieval_time:.2f} seconds.")

        augmented_prompt = (
            f"Based on the following context, answer the question concisely.\n\n"
            f"Context: {retrieved_context}\n\n"
            f"Question: {request.prompt}\n\n"
            f"Answer:"
        )

        inputs = tokenizer(augmented_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(inputs["input_ids"], max_new_tokens=300, temperature=0.7)
        result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        answer_part = result_text.split("Answer:")[-1].strip()
        
        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.2f} seconds.")
        
        return {
            "response": answer_part, 
            "retrieved_context_summary": retrieved_context[:300] + "...", 
            "debug_info": {
                "total_chunks_found": chunk_count,
                "retrieval_time_seconds": round(retrieval_time, 2),
                "total_time_seconds": round(total_time, 2)
            }
        }
        
    except Exception as e:
        return {"error": f"An error occurred during inference: {e}"}

@app.get("/health")
def health_check():
    status = "ok" if all([model, tokenizer, embedding_model]) else "error"
    return {"status": status}
