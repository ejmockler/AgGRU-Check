# Use a lightweight version of Python
FROM python:3.10.0-slim-buster

# Set a working directory
WORKDIR /app

# Install system dependencies required for PyTorch and torchtext
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy all backend files directly to /app
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs in
EXPOSE 8080
EXPOSE 80

# Specify the command to run with logging configuration
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080", "--timeout", "300", "--reload", "--log-config", "log.ini", "main:app"]
