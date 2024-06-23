# Use a lightweight version of Python
FROM python:3.10.0-slim-buster

# Set a working directory
WORKDIR /app

# Install dependencies:
# We copy just the requirements.txt first to leverage Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Install cpu-only version of PyTorch, agGRU-Check is light & fast enough to run without GPU
# RUN pip install --no-cache-dir torch==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Then, copy the rest of the files
COPY . .

# Expose the port the app runs in
EXPOSE 8080
EXPOSE 80

# Specify the command to run
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080", "--timeout", "300", "--log-config", "log.ini", "--reload", "main:app"]
