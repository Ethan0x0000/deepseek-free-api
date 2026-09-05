# syntax=docker/dockerfile:1
#
# deepseek-free-api — Multi-LLM Web Reverse Proxy & Agent Gateway
# Build: docker build -t deepseek-free-api:local .
# Run:   docker compose up -d
#
# 注意:不在镜像内使用 apt-get(Debian 官方源在国内服务器访问极慢)。
# Node.js 通过多阶段构建从 DaoCloud 的官方 Node 镜像直接复制,速度快且稳定。

# ---- 阶段 1:提取 node 二进制(PoW WASM 求解器需要 Node 16+) ----
FROM docker.m.daocloud.io/library/node:20-bookworm-slim AS node-stage

# ---- 阶段 2:运行时镜像 ----
FROM docker.m.daocloud.io/library/python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 复制 node 二进制(与 glibc 动态链接,和 python:3.11-slim-bookworm 同属 bookworm,兼容)
COPY --from=node-stage /usr/local/bin/node /usr/local/bin/node

# Python 依赖(Aliyun PyPI 镜像,国内快)
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

# 应用代码(含 app/wasm/pow_worker.cjs + sha3_wasm_bg.wasm)
COPY app/ app/
COPY cli.py .

# 构建期自检:node 可用 + 应用可导入
RUN node --version && python -c "import app.main"

EXPOSE 8317

# 健康检查用 Python 标准库(镜像内不装 curl)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8317/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8317"]