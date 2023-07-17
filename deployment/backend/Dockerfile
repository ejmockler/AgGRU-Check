# Use a lightweight version of Python
FROM python:3.10.0-slim-buster

# Set a working directory
WORKDIR /app

# Install dependencies:
# We copy just the requirements.txt first to leverage Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Then, copy the rest of the files
COPY . .

# Expose the port the app runs in
EXPOSE 8080

# Specify the command to run
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080", "--timeout", "300", "--log-config", "log.ini", "--reload", "main:app"]
