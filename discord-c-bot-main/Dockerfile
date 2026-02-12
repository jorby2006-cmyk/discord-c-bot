FROM python:3.11-slim

WORKDIR /app

# Install g++ for judging C++ submissions
RUN apt-get update && apt-get install -y --no-install-recommends         g++         && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "botted.py"]
