<div align="center">

# DeepSeek & Qwen Free API Proxy

**高性能异步反代网关 & 交互式控制台，支持 DeepSeek (V4 Pro / Flash / Vision / R1) 与 通义千问 (Qwen 3.7 Plus / 3.8)**

无需付费 API 密钥，直接通过官方 Web 网页会话提供标准 API，完整支持 **Tool Use (Function Calling)**、**Vision 视觉多模态**、**100万 Token 上下文压缩** 与 AI 编程 Agent 深度适配（**OpenCode**, **Cline**, **Roo Code**, **Cursor**, **Claude Code**）。

---

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg?style=flat&logo=python)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?style=flat&logo=docker)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 🌟 核心特性

- 🔓 **免费免 API Key 访问**：复用官方网页端会话，直接调用 DeepSeek 与 通义千问底层模型能力。
- 👁️ **原生 Vision 视觉多模态支持**：
  - 完整实现 Web 端文件上传、OCR 预处理、`fork_file_task` 视觉模型分支调度与 HIF（High-Integrity Framework）防篡改签名机制。
  - 原生兼容 OpenAI 图像格式（`image_url`，支持 Base64 Data URI 与在线图片）及 Anthropic 图片块。
  - 内置图片 SHA-256 缓存机制，多轮对话中避免重复上传相同图片。
- 🛠️ **全功能自主工程 Agent 适配 (多做少说，行动优先)**：
  - 系统提示词深度强化 Agent 执行力，严禁“只给建议不执行”、“口头承诺却不调工具”。
  - 严密监控中英文行动意图声明，内置 `Continuation Recovery`（自动补全恢复机制）。
  - 严格支持 `tool_choice` 参数（`auto`、`required`、强制指定工具）。
  - 支持 DeepSeek 原生 DSML、标准 XML 及 JSON 等多种工具调用格式。
- 📊 **全维度 Token 计量与缓存统计**：
  - **输入 Token** (`prompt_tokens`)：基于多语言精确加权分词计算。
  - **输出 Token** (`completion_tokens`)：直通官方真实生成计数。
  - **上下文缓存读取** (`cached_tokens` / `cache_read_input_tokens`)：自动识别多轮会话重复前缀，精准呈现 Prompt Caching。
  - **思考链统计** (`reasoning_tokens`)：单独计量思维链（Thinking）消耗。
- 🧹 **网页会话自动垃圾回收 (Auto Clean Web Sessions)**：
  - 彻底解决无状态 Agent 调用在 DeepSeek 网页端造成海量垃圾对话刷屏的问题。
  - API 请求完成后由后台异步静默清理网页临时会话，用户网页左侧对话列表始终干净整洁。
- ⚡ **毫秒级 WASM Proof-of-Work (PoW)**：
  - 内置 WebAssembly `DeepSeekHashV1` 求解器，计算挑战耗时通常 < 50ms。
- 🧠 **长上下文与自适应压缩器**：
  - 智能安全截断与摘要压缩，确保复杂长任务不被 Web WAF 拦截。
  - 100% 绝对保护系统设定与 Tools 工具定义不被截断。
- 🔄 **OpenAI & Anthropic 双协议兼容**：
  - `POST /v1/chat/completions`：100% 兼容 OpenAI 格式（含 `reasoning_content` 与 SSE 流式输出）。
  - `POST /v1/messages`：兼容 Anthropic Claude 格式。
- 🖥️ **交互式终端控制台 (`cli.py`)**：
  - 支持浏览器一键免密登录捕获 Token、实时 Proxy 流量监控面板与交互测试。

---

## 📋 支持模型列表

| 模型 ID | 提供商 | 描述 |
| :--- | :---: | :--- |
| `deepseek-v4-pro` | **DeepSeek** | 1.6T MoE (49B 激活参数) — 旗舰复杂编程、架构重构与深度逻辑分析 |
| `deepseek-v4-flash` | **DeepSeek** | 284B MoE — 超高速对话模型，极低延迟 |
| `deepseek-v4-flash-vision-exp` | **DeepSeek** | 视觉多模态模型 — 支持图表分析、截图提问与图像代码解析 |
| `deepseek-reasoner` | **DeepSeek** | DeepSeek-R1 — 完整思维链 (Thinking) 输出的深度推理模型 |
| `deepseek-chat` | **DeepSeek** | DeepSeek V3 — 综合通用任务模型 |
| `deepseek-search` | **DeepSeek** | 内置实时联网搜索增强模式 |
| `qwen3.7-plus` | **通义千问** | 旗舰 Web 模型，支持深度思考 |
| `qwen-3.8-coder` | **通义千问** | 复杂软件工程代码专项模型 |
| `qwen-3.8` | **通义千问** | 第 3 代千问通用旗舰大模型 |

---

## 📦 快速部署与启动

### 方式一：Docker Compose 部署 (推荐)

创建 `docker-compose.yml`：

```yaml
services:
  deepseek-free-api:
    image: deepseek-free-api:local
    build: .
    restart: unless-stopped
    ports:
      - "8317:8317"
    volumes:
      - ./credentials:/root/.deepseek
    environment:
      HOST: 0.0.0.0
      PORT: "8317"
      DEBUG: "false"
      REQUEST_TIMEOUT: "180.0"
      PROXY_MODE: "multi"
      AUTO_CLEAN_WEB_SESSIONS: "true"
      MAX_CONTEXT_TOKENS: "300000"
```

启动服务：
```bash
docker compose up -d --build
```

### 方式二：本地 Python 运行

```bash
# 1. 克隆代码并安装依赖
git clone https://github.com/your-repo/deepseek-free-api.git
cd deepseek-free-api
pip install -r requirements.txt

# 2. 安装浏览器依赖 (仅用于命令行自动登录)
playwright install chromium

# 3. 启动命令行工具或服务
python cli.py
```

---

## 🔑 获取与配置凭证

### 1. 自动提取 (推荐)
运行控制台：
```bash
python cli.py
```
在控制台中输入：
```text
/login deepseek
```
系统将自动调起 Chrome 或 Edge 浏览器窗口，登录后脚本将**自动拦截 JWT Token** 并保存至 `credentials.json`！

### 2. 手动配置
登录 [chat.deepseek.com](https://chat.deepseek.com)，按 F12 打开开发者工具：
- 在 **Application -> Local Storage** 中找到 `userToken`；
- 或在 **Network** 选项卡查看任一 `/api/v0/...` 请求中的 `Authorization: Bearer <token>` 请求头。

通过 API 录入：
```bash
curl -X POST http://127.0.0.1:8317/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"provider": "deepseek", "token": "YOUR_TOKEN"}'
```

---

## 🤖 AI 编程客户端接入配置

### 接入 OpenCode

在 `~/.config/opencode/opencode.json` 中配置：

```json
{
  "provider": {
    "deepseek-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "DeepSeek Free API",
      "options": {
        "baseURL": "http://127.0.0.1:8317/v1",
        "apiKey": "test-key"
      },
      "models": {
        "deepseek-v4-pro": {
          "name": "DeepSeek V4 Pro",
          "limit": { "context": 300000, "output": 128000 }
        },
        "deepseek-v4-flash-vision-exp": {
          "name": "DeepSeek V4 Vision",
          "limit": { "context": 300000, "output": 128000 }
        }
      }
    }
  }
}
```

### 接入 Cline / Roo Code / Cursor

- **API Provider**: `OpenAI Compatible`
- **Base URL**: `http://127.0.0.1:8317/v1`
- **API Key**: `test-key` (填入任意非空字符串)
- **Model ID**: `deepseek-v4-pro` 或 `deepseek-v4-flash-vision-exp`

---

## ⚙️ 环境变量说明

| 变量名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `HOST` | `0.0.0.0` | 监听主机地址 |
| `PORT` | `8317` | 监听端口 |
| `REQUEST_TIMEOUT` | `180.0` | 上游网络请求超时时间 (秒) |
| `PROXY_MODE` | `multi` | 会话隔离模式 (`multi` 每请求独立临时会话 / `single` 单会话) |
| `AUTO_CLEAN_WEB_SESSIONS` | `true` | 请求完成后是否在后台静默删除网页端临时会话 |
| `MAX_CONTEXT_TOKENS` | `300000` | 触发自适应上下文压缩的安全阈值 |
| `MAX_TOOL_OUTPUT_TOKENS` | `25000` | 单个工具执行结果截断上限 |

---

## 📄 开源许可证

本项目基于 [MIT License](LICENSE) 开源发布，仅供技术研究与学习使用。
