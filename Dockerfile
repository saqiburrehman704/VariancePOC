FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py /app/handler.py

CMD ["bash", "-lc", "uvicorn handler:app --host 0.0.0.0 --port ${PORT:-80}"]
