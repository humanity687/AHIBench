FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
COPY worker.py /opt/ahi_shell_worker.py
WORKDIR /workspace
# 容器以 sleep infinity 常驻；worker 由 docker run -i 显式拉起
CMD ["sleep", "infinity"]
