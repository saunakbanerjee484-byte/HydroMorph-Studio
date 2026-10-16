"""Unified PATH-independent launcher for HydroMorph Studio.

Starts the FastAPI compute backend on ``http://127.0.0.1:8000`` as a background
subprocess using ``sys.executable -m uvicorn`` (so it works even when the
Windows PATH does not include the Python Scripts directory), waits for the
``/health`` endpoint to come up, then launches the Streamlit frontend with
``sys.executable -m streamlit run``.

On shutdown (KeyboardInterrupt / process exit) the background FastAPI process is
terminated cleanly so port 8000 is released.
"""

import atexit
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from types import FrameType

ROOT = Path(__file__).resolve().parent
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}/health"
HEALTH_TIMEOUT = 30  # seconds
HEALTH_INTERVAL = 0.5  # seconds

uvicorn_proc: subprocess.Popen[bytes] | None = None

ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(ROOT) + os.pathsep + ENV.get("PYTHONPATH", "")


def _terminate_backend() -> None:
    global uvicorn_proc
    if uvicorn_proc is not None and uvicorn_proc.poll() is None:
        uvicorn_proc.terminate()
        try:
            uvicorn_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            uvicorn_proc.kill()
            uvicorn_proc.wait(timeout=10)
    uvicorn_proc = None


atexit.register(_terminate_backend)


def _wait_for_health() -> None:
    deadline = time.time() + HEALTH_TIMEOUT
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(BACKEND_URL, timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError) as exc:
            last_error = str(exc)
        if uvicorn_proc is not None and uvicorn_proc.poll() is not None:
            raise RuntimeError(
                f"uvicorn backend exited prematurely (code {uvicorn_proc.returncode})"
            )
        time.sleep(HEALTH_INTERVAL)
    raise TimeoutError(f"FastAPI backend did not become healthy at {BACKEND_URL}: {last_error}")


def _sigint_handler(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def main() -> None:
    global uvicorn_proc

    uvicorn_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        BACKEND_HOST,
        "--port",
        str(BACKEND_PORT),
        "--reload",
    ]
    streamlit_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "frontend/app.py",
    ]

    print("[HydroMorph] Starting FastAPI backend ...")
    uvicorn_proc = subprocess.Popen(
        uvicorn_cmd,
        cwd=str(ROOT),
        env=ENV,
    )

    try:
        _wait_for_health()
        print(f"[HydroMorph] FastAPI backend healthy at http://{BACKEND_HOST}:{BACKEND_PORT}")
    except Exception:
        _terminate_backend()
        raise

    print("[HydroMorph] Starting Streamlit frontend ...")
    try:
        subprocess.run(streamlit_cmd, cwd=str(ROOT), check=True, env=ENV)
    finally:
        _terminate_backend()


if __name__ == "__main__":
    if sys.platform == "win32":
        signal.signal(signal.SIGINT, _sigint_handler)
    try:
        main()
    except KeyboardInterrupt:
        print("\n[HydroMorph] Shutting down ...")
        _terminate_backend()
        sys.exit(0)
