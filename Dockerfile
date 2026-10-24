FROM python:3.11-slim
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter
WORKDIR /var/task
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY app ./app
COPY config.yaml .
ENV PORT=8080 AWS_LWA_PORT=8080 AWS_LWA_READINESS_CHECK_PATH=/health HOME=/tmp LITELLM_LOCAL_MODEL_COST_MAP=True
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
