FROM python:3.10-slim

# Install dependencies needed by playwright browsers
RUN apt-get update && apt-get install -y \
    libgtk-4-1 libgraphene-1.0-0 libgstreamer-gl1.0-0 libgstreamer-plugins-base1.0-0 \
    libavif15 libenchant-2-2 libsecret-1-0 libmanette-0.2-0 libgles2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN playwright install --with-deps

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
