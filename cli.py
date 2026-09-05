import asyncio
import os
import sys
from typing import Optional
import httpx
from rich.console import Console
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style

from app.core.config import settings
from app.core.credentials import credentials_manager
from app.providers.registry import provider_registry
from app.schemas.chat import DeepSeekChatRequest
from app.services.session_manager import session_manager

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True, write_through=True)
    except Exception:
        pass

console = Console()

prompt_style = Style.from_dict({
    "prompt": "#5fafff bold",
    "completion-menu.completion": "bg:#202020 #cccccc",
    "completion-menu.completion.current": "bg:#005f87 #ffffff bold",
    "completion-menu.meta.completion": "bg:#202020 #888888",
    "completion-menu.meta.completion.current": "bg:#005f87 #aaaaaa italic",
    "bottom-toolbar": "bg:#1c1c1c #aaaaaa",
})


class MultiProviderCommandCompleter(Completer):
    """支持所有提供商的动态命令与模型自动补全器。"""

    COMMANDS = {
        "/proxy": "进入代理监控模式 (multi 隔离会话或 single 复用会话)",
        "/login": "启动浏览器窗口自动登录并提取凭证 Token",
        "/provider": "切换当前活跃提供商 (deepseek, qwen)",
        "/model": "切换 LLM 大语言模型",
        "/token": "手动设置 Bearer Token (例如: /token qwen <token>)",
        "/think": "思考链显示模式 (show 显示 / hide 隐藏 / off 关闭)",
        "/search": "实时联网搜索增强 (on 开启 / off 关闭)",
        "/new": "开始新对话 (重置上下文)",
        "/sessions": "列出服务端的历史对话列表",
        "/session": "切换到指定历史会话 (例如: /session <id>)",
        "/status": "显示提供商状态、当前模型与会话信息",
        "/clear": "清空终端屏幕",
        "/help": "显示可用命令帮助指南",
        "/exit": "退出终端控制台",
        "/quit": "退出终端控制台",
    }

    SUBCOMMANDS = {
        "/proxy": {
            "multi": "独立会话模式 (每请求独立临时会话 — 适配 Cline/Cursor/OpenCode)",
            "single": "单会话模式 (不创建新会话 — 避免频繁创建)",
            "status": "显示代理运行状态与接入地址",
        },
        "/login": {
            "deepseek": "打开浏览器自动登录 DeepSeek 网页端",
            "qwen": "打开浏览器自动登录 通义千问 网页端",
        },
        "/provider": {
            "deepseek": "DeepSeek (V4 Pro, V4 Flash, V4 Vision, R1, V3)",
            "qwen": "通义千问 (Qwen 3.7 Plus, 3.8, 3.8-Coder, 3-Max)",
        },
        "/think": {
            "show": "开启思考链并展示完整思考过程 (Thinking)",
            "hide": "开启思考链但折叠思考过程 (只展示正文回答)",
            "on": "开启思考链",
            "off": "完全关闭思考链",
        },
        "/search": {
            "on": "开启实时联网搜索",
            "off": "关闭联网搜索",
        },
        "/model": {
            # DeepSeek
            "deepseek-v4-pro": "[DeepSeek] 1.6T MoE (49B 激活) 旗舰编程与深度推理",
            "deepseek-v4-flash": "[DeepSeek] 284B MoE 极速响应对话模型",
            "deepseek-v4-flash-vision-exp": "[DeepSeek] 视觉多模态图片理解模型",
            "deepseek-reasoner": "[DeepSeek] R1 深度慢思考推理模型",
            "deepseek-chat": "[DeepSeek] V3 通用对话模型",
            "deepseek-search": "[DeepSeek] V3 联网搜索增强模型",
            # Qwen
            "qwen3.7-plus": "[Qwen] 通义千问 3.7 Plus (Thinking)",
            "qwen-3.8": "[Qwen] 第 3 代千问通用旗舰模型",
            "qwen-3.8-coder": "[Qwen] 复杂软件工程专项代码模型",
            "qwen-3-max": "[Qwen] 最高智能水平旗舰模型",
            "qwen-3-plus": "[Qwen] 高性价比通用对话模型",
            "qwen-3-flash": "[Qwen] 毫秒级极速响应模型",
        },
        "/token": {
            "deepseek": "设置 DeepSeek Token",
            "qwen": "设置 Qwen Token",
        }
    }

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        parts = text.split()
        if len(parts) == 0:
            return

        if len(parts) == 1 and not text.endswith(" "):
            prefix = parts[0]
            for cmd, desc in self.COMMANDS.items():
                if cmd.startswith(prefix):
                    yield Completion(cmd, start_position=-len(prefix), display_meta=desc)
        else:
            cmd = parts[0]
            if cmd in self.SUBCOMMANDS:
                sub_dict = self.SUBCOMMANDS[cmd]
                sub_prefix = parts[1] if len(parts) > 1 and not text.endswith(" ") else ""
                for sub_cmd, desc in sub_dict.items():
                    if sub_cmd.startswith(sub_prefix):
                        yield Completion(sub_cmd, start_position=-len(sub_prefix), display_meta=desc)


class MultiProviderCLI:
    def __init__(self):
        self.provider_id = "deepseek"
        self.model = "deepseek-v4-pro"
        self.thinking_mode = "show"  # "show" | "hide" | "off"
        self.search_enabled = False
        self.http_client: Optional[httpx.AsyncClient] = None
        self.session: Optional[PromptSession] = None

    async def init(self):
        self.http_client = httpx.AsyncClient(
            timeout=settings.REQUEST_TIMEOUT,
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )
        provider_registry.init_providers(self.http_client)
        self.session = PromptSession(
            history=InMemoryHistory(),
            auto_suggest=AutoSuggestFromHistory(),
            completer=MultiProviderCommandCompleter(),
            style=prompt_style,
        )

    async def close(self):
        if self.http_client:
            await self.http_client.aclose()

    @property
    def is_thinking_enabled(self) -> bool:
        return self.thinking_mode in ["show", "hide"]

    def get_active_session_id(self) -> Optional[str]:
        try:
            prov = provider_registry.get_provider(self.provider_id)
            return prov.get_current_session_id() or session_manager.get_current_session_id()
        except Exception:
            return session_manager.get_current_session_id()

    def get_bottom_toolbar(self):
        prov_name = provider_registry.get_provider(self.provider_id).display_name
        if self.thinking_mode == "show":
            think_str = "🧠 思考过程: 显示"
        elif self.thinking_mode == "hide":
            think_str = "🧠 思考过程: 折叠"
        else:
            think_str = "🧠 思考过程: 关闭"

        search_str = "🌐 联网搜索: 开启" if self.search_enabled else "🌐 联网搜索: 关闭"
        sid = self.get_active_session_id()
        session_short = (sid[:8] + "...") if sid else "新会话"
        return f" [{prov_name}] | [模型: {self.model}] | [{think_str}] | [{search_str}] | [会话: {session_short}] "

    def print_banner(self):
        banner = """
[bold cyan]╔══════════════════════════════════════════════════════════════════╗
║             Multi-LLM Reverse-Engineered Web CLI                 ║
║       DeepSeek V4/R1/Vision    •    Qwen 3.7 Plus / 3.8          ║
╚══════════════════════════════════════════════════════════════════╝[/bold cyan]
        """
        console.print(banner)
        self.print_status()
        console.print("[dim]直接输入消息开始对话，或输入 [bold]/[/bold] 唤起命令交互菜单。[/dim]\n")

    def print_status(self):
        provider = provider_registry.get_provider(self.provider_id)
        is_auth = provider.is_authenticated()
        auth_status = f"[green]✓ 已配置 Token ({provider.display_name})[/green]" if is_auth else f"[bold red]✗ 未配置 Token ({provider.display_name})[/bold red]"
        session_id = self.get_active_session_id() or "[dim]未创建 (发送首条消息时自动生成)[/dim]"

        if self.thinking_mode == "show":
            think_label = "[green]开启 (显示完整思考链)[/green]"
        elif self.thinking_mode == "hide":
            think_label = "[yellow]开启 (折叠思考链)[/yellow]"
        else:
            think_label = "[dim]关闭[/dim]"

        tokens_info = []
        for p in provider_registry.list_providers():
            mark = "[green]✓[/green]" if p["authenticated"] else "[red]✗[/red]"
            active_mark = " [bold cyan](当前活跃)[/bold cyan]" if p["id"] == self.provider_id else ""
            tokens_info.append(f"{mark} {p['name']}{active_mark}")

        status_table = (
            f"  • [bold]可用提供商:[/bold] {' | '.join(tokens_info)}\n"
            f"  • [bold]认证状态:[/bold] {auth_status}\n"
            f"  • [bold]当前模型:[/bold] [yellow]{self.model}[/yellow]\n"
            f"  • [bold]Thinking 思考链:[/bold] {think_label}\n"
            f"  • [bold]联网搜索:[/bold] {'[green]开启[/green]' if self.search_enabled else '[dim]关闭[/dim]'}\n"
            f"  • [bold]会话 ID:[/bold] [cyan]{session_id}[/cyan]"
        )
        console.print(Panel(status_table, title="[bold]服务运行状态[/bold]", border_style="blue"))

    def print_help(self):
        help_text = """
[bold cyan]控制台可用命令 (支持按 Tab 键自动补全):[/bold cyan]
  [bold yellow]/proxy [multi|single][/bold yellow]        - 进入 Proxy 服务模式 (multi: 独立会话, single: 单会话)
  [bold yellow]/login [deepseek|qwen][/bold yellow]    - 打开浏览器窗口扫码登录并自动提取 Token
  [bold yellow]/provider <deepseek|qwen>[/bold yellow] - 切换当前提供商
  [bold yellow]/model <name>[/bold yellow]              - 切换模型 (v4-pro, vision, qwen3.7-plus, coder 等)
  [bold yellow]/token [provider] <token>[/bold yellow]  - 手动保存 Bearer Token
  [bold yellow]/think [show|hide|off][/bold yellow]   - 控制思考链输出模式
  [bold yellow]/search [on|off][/bold yellow]           - 开启或关闭联网搜索增强
  [bold yellow]/new[/bold yellow]                       - 开启新对话 (重置本地与网页上下文)
  [bold yellow]/sessions[/bold yellow]                  - 查看服务端历史对话列表
  [bold yellow]/session <ID>[/bold yellow]              - 切换至指定会话 ID
  [bold yellow]/status[/bold yellow]                    - 查看当前模型、Token 和连接状态
  [bold yellow]/clear[/bold yellow]                     - 清屏
  [bold yellow]/exit[/bold yellow] 或 [bold yellow]/quit[/bold yellow]            - 退出终端
        """
        console.print(Panel(help_text, title="帮助指南", border_style="cyan"))

    async def start_background_server(self):
        """在后台异步任务中启动 FastAPI Proxy 服务。"""
        if hasattr(self, "_server_task") and self._server_task and not self._server_task.done():
            return

        import uvicorn
        from app.main import app as fastapi_app

        config = uvicorn.Config(
            fastapi_app,
            host=settings.HOST,
            port=settings.PORT,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        self._uvicorn_server = server
        self._server_task = asyncio.create_task(server.serve())
        await asyncio.sleep(0.6)

    async def enter_proxy_mode(self, mode: Optional[str] = None):
        """进入为外部 AI Agent (OpenCode, Cline, Cursor, Roo Code) 服务的 Proxy 监控模式。"""
        if mode:
            is_single = mode.lower().strip() in ["single", "1", "s", "true"]
            session_manager.set_single_session_mode(is_single)
        else:
            if sys.stdin and hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
                console.print("\n[bold cyan]请选择 Proxy 会话工作模式:[/bold cyan]")
                console.print("  [bold yellow][1][/bold yellow] [bold]独立会话[/bold] (Multi-Session: 每次请求新建独立临时会话 — [green]推荐用于 Agent[/green])")
                console.print("  [bold green][2][/bold green] [bold]单会话复用[/bold] (Single-Session: 所有请求复用同一个网页会话)")
                try:
                    if self.session:
                        choice = (await self.session.prompt_async([("class:prompt", "请输入您的选择 [1/2] (默认: 1): ")])).strip()
                    else:
                        choice = input("请输入您的选择 [1/2] (默认: 1): ").strip()
                except Exception:
                    choice = "1"
                is_single = (choice == "2")
            else:
                is_single = bool(settings.SINGLE_SESSION_MODE or settings.PROXY_MODE.lower() == "single")
            session_manager.set_single_session_mode(is_single)

        mode_desc = "单会话模式 (Single-Session)" if session_manager.is_single_session_mode() else "独立隔离会话模式 (Multi-Session)"
        console.print(f"[cyan]正在启动端口 {settings.PORT} 上的代理网关 [模式: {mode_desc}]...[/cyan]")
        await self.start_background_server()

        is_single = session_manager.is_single_session_mode()
        mode_badge = (
            "[bold green]单会话复用 (Single-Session)[/bold green]"
            if is_single
            else "[bold yellow]独立临时会话 (Multi-Session: 推荐用于 Agent 编程)[/bold yellow]"
        )

        proxy_banner = f"""
[bold cyan]╔════════════════════════════════════════════════════════════════════════════════════════════════╗
║                   🛡️ PROXY 代理模式: 实时监控与调度 AI Agent 请求                                 ║
║                                                                                                ║
║  • 会话模式:           {mode_badge}
║  • OpenAI 端点:        [bold yellow]http://127.0.0.1:{settings.PORT}/v1/chat/completions[/bold yellow]                               ║
║  • Anthropic 端点:     [bold yellow]http://127.0.0.1:{settings.PORT}/v1/messages[/bold yellow]                                      ║
║  • API Key:            [bold green]任意字符串 (如 'test-key' 或 'deepseek')[/bold green]                                    ║
║                                                                                                ║
║  [bold]客户端接入配置 (OpenCode / Cline / Roo Code / Cursor):[/bold]                                        ║
║    API Provider: [yellow]OpenAI Compatible[/yellow] 或 [yellow]Anthropic[/yellow]                                         ║
║    Base URL:     [yellow]http://127.0.0.1:{settings.PORT}/v1[/yellow]                                                    ║
║    Model ID:     [yellow]deepseek-v4-pro[/yellow] | [yellow]deepseek-v4-flash-vision-exp[/yellow] | [yellow]deepseek-reasoner[/yellow]               ║
╚════════════════════════════════════════════════════════════════════════════════════════════════╝[/bold cyan]
[bold green]● 代理网关已就绪，正在监听客户端调用。[/bold green]
[dim]按 [bold]Ctrl+C[/bold] 可随时退出代理模式并返回终端交互对话。[/dim]
"""
        console.print(proxy_banner)

        from app.services.proxy_logger import proxy_logger
        in_thinking = False

        def on_event(event: dict):
            nonlocal in_thinking
            e_type = event.get("type")

            if e_type == "request_start":
                in_thinking = False
                proto = event.get("protocol", "OpenAI")
                ep = event.get("endpoint", "")
                m = event.get("model", "")
                p = event.get("provider", "")
                toks = event.get("tokens", 0)
                msgs = event.get("messages_count", 0)
                tools = event.get("tools", [])
                t_str = f" | Tools ({len(tools)}): {', '.join(tools[:5])}{'...' if len(tools)>5 else ''}" if tools else " | Tools: 无"
                t_now = event.get("time", "")
                sess_info = "Single-Session" if session_manager.is_single_session_mode() else "Multi-Session"

                console.print(f"\n[bold magenta]┌── 📥 [{t_now}] 收到来自 Agent 的调用请求 ({event.get('user_agent', 'Agent')}) ──────────────────────[/bold magenta]")
                console.print(f"[bold magenta]│[/bold magenta] [bold cyan]协议:[/bold cyan] {proto} ({ep}) | [bold cyan]模型:[/bold cyan] [yellow]{m}[/yellow] -> [green]{p}[/green]")
                console.print(f"[bold magenta]│[/bold magenta] [dim]会话: {sess_info} | 上下文: {msgs} 条消息 (~{toks:,} Token){t_str}[/dim]")
                console.print(f"[bold magenta]└── 流式生成中 ───────────────────────────────────────────────────────────[/bold magenta]")

            elif e_type == "thinking_chunk":
                if not in_thinking:
                    sys.stdout.write("\n\033[90m🧠 思考过程: ")
                    in_thinking = True
                sys.stdout.write(event.get("text", ""))
                sys.stdout.flush()

            elif e_type == "content_chunk":
                if in_thinking:
                    sys.stdout.write("\033[0m\n\n")
                    in_thinking = False
                text_chunk = event.get("text", "")
                if not any(tag in text_chunk for tag in ["<｜DSML｜", "<|DSML|", "<||DSML||", "<tool_call", "<invoke"]):
                    sys.stdout.write(text_chunk)
                    sys.stdout.flush()

            elif e_type == "tool_call":
                fn_name = event.get("tool_name", "")
                args = event.get("arguments", "")
                console.print(f"\n[bold yellow]🛠️  [工具调用][/bold yellow] [bold cyan]{fn_name}[/bold cyan]([dim]{args[:120]}{'...' if len(args)>120 else ''}[/dim])")

            elif e_type == "request_end":
                if in_thinking:
                    sys.stdout.write("\033[0m\n")
                    in_thinking = False
                status_code = event.get("status_code", 200)
                toks_out = event.get("tokens_out", 0)
                status_style = "bold green" if status_code == 200 else "bold red"
                console.print(f"\n[{status_style}]✓ 请求处理完成 [{status_code}][/] | 输出 Token: [cyan]{toks_out}[/cyan] | 网关就绪等待下次请求...\n")

        proxy_logger.subscribe(on_event)

        try:
            while True:
                await asyncio.sleep(0.5)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            proxy_logger.unsubscribe(on_event)
            console.print("\n[yellow]已退出代理模式，返回交互对话。[/yellow]\n")
            self.print_status()

    def set_provider_and_default_model(self, pid: str):
        pid = pid.lower().strip()
        if pid == "qwen":
            self.provider_id = "qwen"
            self.model = "qwen3.7-plus"
            provider_registry.set_default_provider("qwen")
            console.print("[green]✓ 已切换至 Qwen (默认模型: qwen3.7-plus)[/green]")
        elif pid == "deepseek":
            self.provider_id = "deepseek"
            self.model = "deepseek-v4-pro"
            provider_registry.set_default_provider("deepseek")
            console.print("[green]✓ 已切换至 DeepSeek (默认模型: deepseek-v4-pro)[/green]")
        else:
            console.print(f"[red]未知提供商:[/red] {pid}。可用: deepseek, qwen")

    async def handle_chat(self, user_input: str):
        provider = provider_registry.resolve_provider_for_model(self.model)

        if not provider.is_authenticated():
            console.print(
                f"[bold red]错误:[/bold red] 提供商 {provider.display_name} 尚未配置凭证。\n"
                f"请执行浏览器登录: [bold yellow]/login {provider.provider_id}[/bold yellow] 或手动输入 Token: [bold yellow]/token {provider.provider_id} <token>[/bold yellow]"
            )
            return

        req = DeepSeekChatRequest(
            prompt=user_input,
            chat_session_id=provider.get_current_session_id(),
            model=self.model,
            thinking_enabled=self.is_thinking_enabled,
            search_enabled=self.search_enabled,
            stream=True,
        )

        in_thinking = False
        in_content = False
        tokens_count = 0

        try:
            async for chunk in provider.stream_chat(req):
                if chunk.session_id:
                    provider.set_session_id(chunk.session_id)

                if chunk.token_usage:
                    tokens_count = chunk.token_usage

                if chunk.type == "thinking":
                    if self.thinking_mode == "show":
                        if not in_thinking:
                            console.print(f"\n[dim]╭─── 🧠 {provider.display_name} 思考中 ─────────────────────────────────╮[/dim]")
                            sys.stdout.write("\033[90m")
                            in_thinking = True
                        sys.stdout.write(chunk.text)
                        sys.stdout.flush()
                    elif self.thinking_mode == "hide":
                        if not in_thinking:
                            sys.stdout.write(f"\r\033[90m🧠 {provider.display_name} 正在思考...\033[0m")
                            sys.stdout.flush()
                            in_thinking = True

                elif chunk.type == "content":
                    if in_thinking:
                        if self.thinking_mode == "show":
                            sys.stdout.write("\033[0m\n")
                            console.print("[dim]╰───────────────────────────────────────────────────────────────────────╯[/dim]\n")
                        elif self.thinking_mode == "hide":
                            sys.stdout.write("\r\033[K")
                            sys.stdout.flush()
                        in_thinking = False

                    if not in_content:
                        console.print(f"[bold cyan]{provider.display_name}:[/bold cyan]")
                        in_content = True

                    sys.stdout.write(chunk.text)
                    sys.stdout.flush()

            if in_thinking and self.thinking_mode == "show":
                sys.stdout.write("\033[0m\n")
                console.print("[dim]╰───────────────────────────────────────────────────────────────────────╯[/dim]")

            print("\n", flush=True)

            sid = provider.get_current_session_id() or session_manager.get_current_session_id() or "新会话"
            info_str = f"[dim]提供商: {provider.display_name} | 模型: {self.model} | 会话 ID: {sid} | Token 消耗: {tokens_count or 'N/A'}[/dim]\n"
            console.print(info_str)

        except Exception as e:
            console.print(f"\n[bold red]执行请求异常:[/bold red] {e}\n")

    async def run(self, auto_proxy: bool = False, proxy_mode: Optional[str] = None):
        await self.init()
        self.print_banner()

        if auto_proxy:
            await self.enter_proxy_mode(mode=proxy_mode)

        while True:
            try:
                user_input = await self.session.prompt_async(
                    [("class:prompt", "您 > ")],
                    bottom_toolbar=self.get_bottom_toolbar,
                )
                user_input = user_input.strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=1)
                    cmd = parts[0].lower()
                    arg = parts[1].strip() if len(parts) > 1 else ""

                    if cmd in ["/exit", "/quit", "/q"]:
                        console.print("[cyan]再见！[/cyan]")
                        break
                    elif cmd == "/help":
                        self.print_help()
                    elif cmd == "/status":
                        self.print_status()
                    elif cmd == "/clear":
                        os.system("cls" if os.name == "nt" else "clear")
                        self.print_banner()
                    elif cmd in ["/proxy", "/server"]:
                        proxy_arg = arg.lower().strip() if arg else None
                        await self.enter_proxy_mode(mode=proxy_arg)
                    elif cmd == "/login":
                        target_p = arg.lower().strip() if arg else self.provider_id
                        if target_p not in ["deepseek", "qwen"]:
                            target_p = self.provider_id
                        console.print(f"\n[bold cyan]🌐 正在启动 Chrome 浏览器登录 {target_p.upper()}...[/bold cyan]")
                        console.print("[dim]请在打开的浏览器窗口完成登录，Token 将被自动拦截保存。[/dim]\n")
                        from app.services.browser_auth import extract_token_via_browser
                        tok = await extract_token_via_browser(provider=target_p, headless=False, timeout_seconds=120)
                        if tok:
                            console.print(f"\n[bold green]✓ {target_p} Token 捕获成功并已存入 credentials.json！[/bold green]\n")
                        else:
                            console.print(f"\n[bold red]✗ 提取 Token 失败 (超时或窗口被关闭)。[/bold red]\n")
                        self.print_status()
                    elif cmd == "/provider":
                        if not arg:
                            console.print("[yellow]用法:[/yellow] /provider <deepseek | qwen>")
                        else:
                            self.set_provider_and_default_model(arg)
                        self.print_status()
                    elif cmd == "/token":
                        token_parts = arg.split(maxsplit=1)
                        if len(token_parts) == 0 or not token_parts[0]:
                            console.print("[red]用法:[/red] /token <token> 或 /token <provider> <token>")
                        elif len(token_parts) == 1:
                            credentials_manager.save(token_parts[0], provider=self.provider_id)
                            console.print(f"[green]✓ 提供商 {self.provider_id} 的 Token 保存成功！[/green]")
                            self.print_status()
                        else:
                            p_name, p_tok = token_parts[0].lower(), token_parts[1]
                            credentials_manager.save(p_tok, provider=p_name)
                            console.print(f"[green]✓ 提供商 {p_name} 的 Token 保存成功！[/green]")
                            self.print_status()
                    elif cmd == "/new":
                        session_manager.reset_context()
                        try:
                            cur_p = provider_registry.get_provider(self.provider_id)
                            cur_p.reset_session()
                        except Exception:
                            pass
                        console.print("[green]✓ 已开启新对话，上下文已重置。[/green]")
                        self.print_status()
                    elif cmd == "/sessions":
                        try:
                            cur_p = provider_registry.get_provider(self.provider_id)
                            sessions_list = await cur_p.list_sessions()
                            if not sessions_list:
                                console.print(f"[yellow]提供商 {cur_p.display_name} 无可用历史会话。[/yellow]")
                            else:
                                console.print(f"\n[bold cyan]历史对话列表 ({cur_p.display_name}):[/bold cyan]")
                                for idx, s in enumerate(sessions_list[:15], 1):
                                    s_id = s.get("id", "")
                                    s_title = s.get("title", "未命名")
                                    is_curr = " [green](当前)[/green]" if s_id == cur_p.get_current_session_id() else ""
                                    console.print(f"  {idx}. [yellow]{s_id}[/yellow] — [bold]{s_title}[/bold]{is_curr}")
                                console.print("[dim]切换会话请执行:[/dim] [bold yellow]/session <ID>[/bold yellow]\n")
                        except Exception as e:
                            console.print(f"[red]获取历史会话失败:[/red] {e}")
                    elif cmd == "/session":
                        if not arg:
                            console.print("[yellow]用法:[/yellow] /session <会话ID>")
                        else:
                            try:
                                cur_p = provider_registry.get_provider(self.provider_id)
                                cur_p.set_session_id(arg)
                                console.print(f"[green]✓ 已切换至会话 {arg} ({cur_p.display_name})[/green]")
                                self.print_status()
                            except Exception as e:
                                console.print(f"[red]切换会话失败:[/red] {e}")
                    elif cmd == "/think":
                        arg_lower = arg.lower()
                        if arg_lower in ["show", "on"]:
                            self.thinking_mode = "show"
                            console.print("[green]✓ 思考模式: 显示完整思考过程[/green]")
                        elif arg_lower in ["hide", "hidden"]:
                            self.thinking_mode = "hide"
                            console.print("[yellow]✓ 思考模式: 思考过程折叠 (只展示最终正文)[/yellow]")
                        elif arg_lower == "off":
                            self.thinking_mode = "off"
                            console.print("[yellow]✓ 思考模式: 关闭[/yellow]")
                        else:
                            if self.thinking_mode == "off":
                                self.thinking_mode = "show"
                            elif self.thinking_mode == "show":
                                self.thinking_mode = "hide"
                            else:
                                self.thinking_mode = "off"
                        self.print_status()
                    elif cmd == "/search":
                        if arg.lower() == "on":
                            self.search_enabled = True
                            console.print("[green]✓ 联网搜索已开启[/green]")
                        elif arg.lower() == "off":
                            self.search_enabled = False
                            console.print("[yellow]✓ 联网搜索已关闭[/yellow]")
                        else:
                            self.search_enabled = not self.search_enabled
                            console.print(f"联网搜索: {'[green]开启[/green]' if self.search_enabled else '[dim]关闭[/dim]'}")
                        self.print_status()
                    elif cmd == "/model":
                        arg_clean = arg.lower().strip()
                        if not arg_clean:
                            console.print(f"[cyan]当前模型:[/cyan] {self.model}")
                        else:
                            target_provider = provider_registry.resolve_provider_for_model(arg_clean)
                            self.provider_id = target_provider.provider_id
                            self.model = arg_clean
                            console.print(f"[green]✓ 已切换模型: {self.model} (提供商: {target_provider.display_name})[/green]")
                        self.print_status()
                    else:
                        console.print(f"[red]未知命令:[/red] {cmd}。输入 [bold]/help[/bold] 查看帮助。")
                    continue

                await self.handle_chat(user_input)

            except (KeyboardInterrupt, EOFError):
                console.print("\n[cyan]控制台已退出。[/cyan]")
                break

        await self.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="DeepSeek & Qwen 免费网页反代 API 与交互控制台",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--proxy", "-p",
        action="store_true",
        help="直接以 Proxy 代理模式运行 (供 OpenCode, Cline, Cursor, Roo Code 接入)",
    )
    parser.add_argument(
        "--mode", "--proxy-mode",
        dest="proxy_mode",
        type=str,
        default=None,
        choices=["single", "multi"],
        help="代理会话模式: 'multi' (每请求隔离临时会话) 或 'single' (单会话复用)",
    )
    parser.add_argument(
        "--single", "-s",
        action="store_true",
        help="以单会话模式启动代理",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="以独立临时会话模式启动代理 (推荐 Agent 使用)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["deepseek", "qwen"],
        help="指定默认提供商 (deepseek 或 qwen)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="指定默认模型 (如 deepseek-v4-pro, qwen3.7-plus)",
    )
    parser.add_argument(
        "command",
        nargs="*",
        default=[],
        help="快捷执行命令 (例如 'proxy multi' 或 'proxy single')",
    )

    args = parser.parse_args()

    cli = MultiProviderCLI()

    raw_cmds = args.command if isinstance(args.command, list) else [args.command]
    commands = [str(c).lower().strip() for c in raw_cmds if c]

    auto_proxy = args.proxy or any(c in ["proxy", "server", "/proxy", "/server"] for c in commands)

    proxy_mode = None
    if args.single:
        proxy_mode = "single"
    elif args.multi:
        proxy_mode = "multi"
    elif args.proxy_mode:
        proxy_mode = args.proxy_mode.lower()
    else:
        for c in commands:
            if c in ["single", "1", "s"]:
                proxy_mode = "single"
                break
            elif c in ["multi", "2", "m"]:
                proxy_mode = "multi"
                break

    if args.provider:
        cli.set_provider_and_default_model(args.provider)
    if args.model:
        target_provider = provider_registry.resolve_provider_for_model(args.model)
        cli.provider_id = target_provider.provider_id
        cli.model = args.model

    asyncio.run(cli.run(auto_proxy=auto_proxy, proxy_mode=proxy_mode))


if __name__ == "__main__":
    main()
