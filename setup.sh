python -m venv venv

source ./venv/Scripts/activate

pip install -r requirements.txt

#set HF_TOKEN=hf_your_token_here

# 로컬 환경 FastAPI 실행
uvicorn app:app --host 0.0.0.0 --port 8000

curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "What is a Kubernetes Pod? Answer short"}'

# Docker
docker login

docker build -t judemin/bitnet-fastapi:0.3 .

docker push judemin/bitnet-fastapi:0.3