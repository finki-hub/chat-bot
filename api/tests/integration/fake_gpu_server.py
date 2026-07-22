from __future__ import annotations

import argparse
import json
from pathlib import Path
from socketserver import StreamRequestHandler, ThreadingTCPServer
from time import monotonic
from typing import ClassVar

_VECTOR_DIMENSIONS = 1024


class FakeGpuHandler(StreamRequestHandler):
    modes: ClassVar[tuple[str, ...]]
    request_index: ClassVar[int] = 0
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
        index = FakeGpuHandler.request_index
        FakeGpuHandler.request_index += 1
        mode = self.modes[min(index, len(self.modes) - 1)]
        with self.record_path.open("a", encoding="utf-8") as record:
            record.write(
                json.dumps(
                    {
                        "count": len(values),
                        "mode": mode,
                        "request": index + 1,
                        "monotonic_ms": round(monotonic() * 1000),
                    },
                )
                + "\n",
            )

        if mode == "error":
            self.wfile.write(
                b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n",
            )
            return

        vectors = [[0.25] * _VECTOR_DIMENSIONS for _value in values]
        if mode == "malformed":
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
    parser.add_argument("--modes", default="")
    arguments = parser.parse_args()
    modes = tuple(mode for mode in arguments.modes.split(",") if mode)
    if not modes:
        modes = (arguments.mode,)
    if any(mode not in {"success", "malformed", "error"} for mode in modes):
        parser.error("--modes must contain success, malformed, or error")
    FakeGpuHandler.modes = modes
    FakeGpuHandler.record_path = arguments.record_path
    server = ThreadingTCPServer(("127.0.0.1", arguments.port), FakeGpuHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
