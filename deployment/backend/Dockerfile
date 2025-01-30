FROM python:3.10-slim-bullseye

# Set a working directory
WORKDIR /app

# Separate layer for dependencies
COPY requirements.txt .
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend files directly to /app
COPY . .

# Expose the port the app runs in
EXPOSE 8080
EXPOSE 80

# Increased timeout to 1800 seconds (30 minutes)
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080", "--timeout", "1800", "--graceful-timeout", "300", "--keep-alive", "5", "--reload", "--log-config", "log.ini", "main:app"]
