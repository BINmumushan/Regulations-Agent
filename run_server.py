"""启动 FastAPI 服务：python run_server.py [--host ...] [--port ...]"""

from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="启动知识库问答服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("kb_agent.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
