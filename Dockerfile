FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Smoke-test the provided AI contract and the student's SE layer during build.
RUN pytest -q

# Default command uses offline mode so the container runs without API keys.
CMD ["python", "-m", "researcher", "ask", "What is photosynthesis and what are its main stages?", "--offline", "--no-cache"]
