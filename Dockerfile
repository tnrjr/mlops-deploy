FROM python:3.11-slim


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1




WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY src/ ./src
COPY models/ /usr/model/


ENV MODEL_PATH=/usr/model/model.pkl \
    PYTHONPATH=/app/src


EXPOSE 8000

CMD ["uvicorn","main:app","--host","0.0.0.0","--port","${PORT}"]