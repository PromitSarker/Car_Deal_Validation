# Use an official Python image (smaller & faster than ubuntu)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /App

# copy & install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source
COPY . .

# expose port
EXPOSE 8008

# run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port","8008"]