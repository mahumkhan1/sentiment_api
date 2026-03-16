# stage 1: building docker image
FROM python:3.12-slim as BUILDER

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \ 
    && pip install --no-cache-dir --upgrade pip 

# stage 2: running the app
FROM python:3.12-slim

WORKDIR /app

# copy installed dependencies from builder stage
COPY --from=BUILDER /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# copy app code and saved model
COPY main.py .
COPY model/ ./model/

# non-root user for security
RUN useradd -m appuser
USER appuser

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]
