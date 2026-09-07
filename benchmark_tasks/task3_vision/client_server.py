#!/usr/bin/env python3
"""
client_server.py
================
基于 Python asyncio 的异步网络通讯代码模型，高度还原 architecture_diagram.png
所展示的客户端-服务器拓扑：

     [Laptop]   \\
     [Mobile]  --->  [ Internet ]  --->  [ Server ]
     [Desktop]  /

- 多个异构客户端（笔记本电脑 / 手机 / 台式机）通过 Internet 连接中心服务器。
- 客户端发起请求 (request)，服务器处理并返回响应 (response)。
- 使用 asyncio 实现并发、非阻塞的异步通信，多个客户端可同时在线。
- 使用 asyncio.StreamReader/StreamWriter 实现流式 JSON 行协议。

运行方式:
    python3 client_server.py
"""

import asyncio
import json
import random
import time
import sys

# ---------------------------------------------------------------------------
# 配置常量
# ---------------------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 8848

# 客户端设备清单（对应图中的三个客户端节点）
CLIENTS = [
    {"name": "Laptop",  "type": "laptop",  "latency_ms": (10, 30)},   # 无线，延迟较高
    {"name": "Mobile",  "type": "mobile",  "latency_ms": (15, 40)},   # 移动网络，延迟最高
    {"name": "Desktop", "type": "desktop", "latency_ms": (5, 15)},    # 有线，延迟最低
]


# ---------------------------------------------------------------------------
# 服务器端共享状态
# ---------------------------------------------------------------------------
class ServerState:
    """记录服务器运行时的全局状态，用于 stats 请求。"""
    active_connections = 0
    total_requests = 0
    started_at = time.time()


# ---------------------------------------------------------------------------
# 服务器端 (Server) —— 对应图中的 Server 节点
# ---------------------------------------------------------------------------
class ClientSession:
    """为每个连接的客户端维护会话元数据。"""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer
        self.peer = writer.get_extra_info("peername")
        self.connected_at = time.time()
        self.requests_served = 0


async def handle_request(session: ClientSession, request: dict) -> dict:
    """
    服务器请求分发处理器：根据请求中的 "cmd" 字段调用对应的处理逻辑。
    """
    cmd = request.get("cmd", "unknown")
    payload = request.get("payload", None)

    if cmd == "ping":
        return {"status": "ok", "message": "pong", "ts": time.time()}
    elif cmd == "time":
        return {"status": "ok", "server_time": time.strftime("%Y-%m-%d %H:%M:%S")}
    elif cmd == "echo":
        return {"status": "ok", "echoed": payload}
    elif cmd == "stats":
        return {
            "status": "ok",
            "active_connections": ServerState.active_connections,
            "total_requests": ServerState.total_requests,
        }
    else:
        return {"status": "error", "message": f"unknown command: {cmd}"}


async def _send_response(writer: asyncio.StreamWriter, response: dict):
    """将响应对象序列化为 JSON 一行并发送。"""
    data = (json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8")
    writer.write(data)
    await writer.drain()


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """
    处理单个客户端连接的协程。
    - 读取一行请求（JSON）
    - 处理并返回响应（JSON 一行）
    - 支持多次请求/响应循环，直到客户端断开或发送 "quit"
    """
    session = ClientSession(reader, writer)
    ServerState.active_connections += 1
    peer = session.peer
    print(f"[Server]  客户端接入: {peer}  (当前在线: {ServerState.active_connections})")

    try:
        while True:
            line = await reader.readline()
            if not line:  # 客户端关闭连接
                break

            try:
                request = json.loads(line.decode("utf-8").strip())
            except json.JSONDecodeError:
                response = {"status": "error", "message": "invalid JSON"}
            else:
                if request.get("cmd") == "quit":
                    response = {"status": "ok", "message": "bye"}
                    await _send_response(writer, response)
                    break

                # 模拟服务器处理耗时（非阻塞，让出事件循环）
                await asyncio.sleep(random.uniform(0.01, 0.05))
                response = await handle_request(session, request)
                session.requests_served += 1
                ServerState.total_requests += 1

            await _send_response(writer, response)
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        ServerState.active_connections -= 1
        print(f"[Server]  客户端断开: {peer}  (当前在线: {ServerState.active_connections})")
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def start_server():
    """启动 asyncio TCP 服务器并返回 server 对象。"""
    server = await asyncio.start_server(handle_client, HOST, PORT)
    addr = server.sockets[0].getsockname()
    print(f"[Server]  已启动，监听 {addr[0]}:{addr[1]}")
    return server


# ---------------------------------------------------------------------------
# 客户端 (Clients) —— 对应图中的 Laptop / Mobile / Desktop 节点
# ---------------------------------------------------------------------------
async def run_client(device: dict):
    """
    客户端协程：模拟一台设备发起请求并获得响应。
    - 通过 asyncio.open_connection 建立到服务器的 TCP 连接（经由 Internet）
    - 以 JSON 行协议发送请求，接收响应
    - 执行一次 ping / time / echo / stats 请求序列后关闭连接
    """
    name = device["name"]
    lat_range = device["latency_ms"]
    print(f"[{name:<8}] 尝试连接服务器 {HOST}:{PORT} ...")

    # 模拟经由 Internet 的网络延迟（链路建立阶段）
    await asyncio.sleep(random.uniform(*lat_range) / 1000.0)

    reader, writer = await asyncio.open_connection(HOST, PORT)
    print(f"[{name:<8}] 已连接服务器，开始发起请求序列")

    requests = [
        {"cmd": "ping", "payload": None},
        {"cmd": "time", "payload": None},
        {"cmd": "echo", "payload": f"hello from {name}"},
        {"cmd": "stats", "payload": None},
    ]

    for req in requests:
        # 每个请求经由 Internet 的传输延迟
        await asyncio.sleep(random.uniform(*lat_range) / 1000.0)

        data = (json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8")
        writer.write(data)
        await writer.drain()

        resp_line = await reader.readline()
        resp = json.loads(resp_line.decode("utf-8"))
        print(f"[{name:<8}] 请求 {req['cmd']:<6} -> 响应 {resp}")

    # 发送退出请求
    quit_req = {"cmd": "quit"}
    writer.write((json.dumps(quit_req) + "\n").encode("utf-8"))
    await writer.drain()
    await reader.readline()  # 读取 bye

    writer.close()
    await writer.wait_closed()
    print(f"[{name:<8}] 连接已关闭，会话结束")


async def main():
    """主入口：启动服务器并并发运行多个客户端。"""
    server = await start_server()

    # 并发地运行所有客户端（模拟图中多个设备同时通过 Internet 访问服务器）
    print("[Main]     启动所有客户端并发访问服务器 ...\n" + "-" * 70)
    await asyncio.gather(*(run_client(device) for device in CLIENTS))

    print("-" * 70)
    print(f"[Main]     全部客户端会话完成。服务器累计处理请求: {ServerState.total_requests}")

    server.close()
    await server.wait_closed()
    print("[Main]     服务器已关闭，程序退出。")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Main]     程序被用户中断。")
        sys.exit(0)
