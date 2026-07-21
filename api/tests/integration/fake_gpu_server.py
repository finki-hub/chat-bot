from __future__ import annotations

import argparse
import json
from pathlib import Path
from socketserver import StreamRequestHandler, ThreadingTCPServer
from typing import ClassVar

_VECTOR_DIMENSIONS = 1024


class FakeGpuHandler(StreamRequestHandler):
    mode: ClassVar[str]
    record_path: ClassVar[Path]

    def handle(self) -> None:
        self.rfile.readline()
        headers: dict[str, str] = {}
        while line := self.rfile.readline().decode().strip():
            name, value = line.split(":", 1)
            headers[name.lower()] = value.strip()
        length = int(headers["content-length"])
        payload = json.loads(self.rfile.read(length))
        inputs = payload["input"]
        values = inputs if isinstance(inputs, list) else [inputs]
        with self.record_path.open("a", encoding="utf-8") as record:
            record.write(json.dumps({"count": len(values), "mode": self.mode}) + "\n")

        if self.mode == "error":
            self.wfile.write(
                b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n",
            )
            return

        vectors = [[0.25] * _VECTOR_DIMENSIONS for _value in values]
        if self.mode == "malformed":
            vectors = [[0.25] * 3 for _value in values]
        body = json.dumps({"embeddings": vectors}).encode()
        response_headers = (
            f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n\r\n"
        ).encode()
        self.wfile.write(response_headers)
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--record-path", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("success", "malformed", "error"),
        default="success",
    )
    arguments = parser.parse_args()
    FakeGpuHandler.mode = arguments.mode
    FakeGpuHandler.record_path = arguments.record_path
    server = ThreadingTCPServer(("127.0.0.1", arguments.port), FakeGpuHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
