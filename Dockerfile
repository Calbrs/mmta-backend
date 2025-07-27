# Use official Python slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for Playwright browsers
RUN apt-get update && apt-get install -y \
    libgtk-4-1 \
    libgraphene-1.0-0 \
    libgstreamer-gl1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libavif15 \
    libenchant-2-2 \
    libsecret-1-0 \
    libmanette-0.2-0 \
    libgles2-mesa \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers with dependencies
RUN pip install --no-cache-dir playwright && \
    playwright install --with-deps

# Copy rest of the app code
COPY . .

# Expose port (match uvicorn port)
EXPOSE 10000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
