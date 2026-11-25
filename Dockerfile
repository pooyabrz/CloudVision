FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /opt/cloudvision
COPY requirements.txt .
RUN python -m pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
 && python -m pip install --no-cache-dir -r requirements.txt \
 && useradd --create-home --uid 10001 cloudvision
COPY --chown=cloudvision:cloudvision . .
USER cloudvision
EXPOSE 8000 8501
CMD ["python","-m","uvicorn","api.main:app","--host","0.0.0.0","--port","8000"]
