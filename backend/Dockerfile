FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# requirements-dev.txt는 requirements.txt를 포함하므로 한 번에 설치하면 pytest·httpx도 들어간다.
# 캡스톤 범위에서는 운영/개발 이미지를 분리하지 않고 컨테이너 안에서 pytest를 돌릴 수 있게 둔다.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY app ./app
COPY scripts ./scripts
COPY tests ./tests
COPY frontend ./frontend
COPY pytest.ini ./

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
