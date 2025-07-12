FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Set a default PORT if $PORT is not set by Railway
ENV PORT=5000

EXPOSE $PORT

CMD ["gunicorn", "--bind", "0.0.0.0:${PORT}", "app:app"]
