FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt

# Install project
COPY . .
RUN pip install -e .

# Expose ports
EXPOSE 8501 3000

# Default: start Dashboard
CMD ["streamlit", "run", "src/observability/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
