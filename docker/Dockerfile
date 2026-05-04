FROM python:3.10-slim
WORKDIR /app
RUN pip install --upgrade pip
COPY docker/requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt
COPY src/ ./src/
COPY data/ ./data/
EXPOSE 5001
CMD ["python", "src/serve.py"]