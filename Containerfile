# Backlot board server (read-only view of projects/). The agent pipeline runs on the host.
FROM docker.io/library/python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN pip install --no-cache-dir "fastapi>=0.110" "uvicorn>=0.29" "watchfiles>=0.21" "pyyaml>=6.0" "pydantic>=2.0" "jsonschema>=4.20" "python-dotenv>=1.0" "filelock>=3.0" "Pillow>=10.0" "numpy>=1.24"
COPY . .
RUN mkdir -p /app/projects
ENV BACKLOT_HOST=0.0.0.0 BACKLOT_PORT=4750 BACKLOT_ALLOWED_HOSTS=127.0.0.1,localhost
EXPOSE 4750
CMD ["python", "-m", "backlot", "serve"]
