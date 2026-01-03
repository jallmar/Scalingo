FROM python:3.10-slim  # Slim base to reduce size (instead of full python image)
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt  # No cache to avoid extra files
CMD ["python", "app.py"]