FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd --create-home paper && mkdir -p data && chown -R paper:paper /app
USER paper
CMD ["python","-m","uvicorn","paper.api:create_app","--factory","--host","0.0.0.0","--port","18100"]
