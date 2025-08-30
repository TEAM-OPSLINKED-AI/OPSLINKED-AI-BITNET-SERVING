FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
# --no-cache-dir 옵션으로 불필요한 캐시를 남기지 않아 이미지 크기를 최적화
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# 컨테이너가 8000번 포트를 외부에 노출하도록 명시
EXPOSE 8000

# 0.0.0.0으로 바인딩하여 컨테이너 외부에서 접근 가능
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]