FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY chatting_bot.py ./chatting_bot.py

EXPOSE 8501

CMD ["streamlit", "run", "chatting_bot.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
