#!/usr/bin/env python3
"""Start the VitaScan backend and frontend together.

    python start.py            # both, frontend in dev mode
    python start.py --prod     # both, frontend from a production build
    python start.py --setup    # install dependencies, then start
    python start.py --help

Runs on Windows, Linux and macOS with the same command. See start.sh / start.bat
for the shell entry points.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
IS_WIN = os.name == "nt"

WINDOWS_PYTHON_HINT = (
    "Windows wheels only exist for older CPython builds: recreate the venv with "
    "`py -3.12 -m venv backend\\venv` (Python 3.12 x64), not 3.13/3.14."
)


class Colors:
    def __init__(self) -> None:
        on = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
        self.reset = "\033[0m" if on else ""
        self.bold = "\033[1m" if on else ""
        self.blue = "\033[34m" if on else ""
        self.green = "\033[32m" if on else ""
        self.red = "\033[31m" if on else ""
        self.yellow = "\033[33m" if on else ""
        self.magenta = "\033[35m" if on else ""
        self.dim = "\033[2m" if on else ""


C = Colors()
BACKEND_TAG = f"{C.blue}backend {C.reset}"
FRONTEND_TAG = f"{C.magenta}frontend{C.reset}"


def say(tag: str, message: str) -> None:
    print(f"{tag} | {message}", flush=True)


def step(message: str) -> None:
    print(f"\n{C.bold}==> {message}{C.reset}", flush=True)


def warn(message: str) -> None:
    print(f"{C.yellow}warning{C.reset} | {message}", flush=True)


def fail(message: str, hint: str = "") -> None:
    print(f"\n{C.red}error{C.reset} | {message}", flush=True)
    if hint:
        print(f"       {C.dim}{hint}{C.reset}", flush=True)
    sys.exit(1)


def venv_python() -> Path:
    relative = Path("Scripts/python.exe") if IS_WIN else Path("bin/python")
    return BACKEND / "venv" / relative


def resolve_python() -> Optional[Path]:
    if venv_python().exists():
        return venv_python()
    for name in ("python3", "python", "py"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def port_is_open(host: str, port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.6)
        return probe.connect_ex((host, port)) == 0


def describe_port_owner(port: int) -> str:
    if IS_WIN:
        cmd = ["netstat", "-ano"]
        if not shutil.which(cmd[0]):
            return ""
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
        except Exception:
            return ""
        rows = [ln for ln in out.splitlines() if f":{port}" in ln and "LISTENING" in ln]
        if not rows:
            return ""
        pids = {ln.split()[-1] for ln in rows if ln.split()}
        return f" (pid {', '.join(sorted(pids))})"
    if not shutil.which("lsof"):
        return ""
    try:
        out = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except Exception:
        return ""
    names = {ln.split()[1] for ln in out.splitlines()[1:] if len(ln.split()) > 1}
    return f" ({', '.join(sorted(names))})" if names else ""


def http_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return 200 <= response.status < 500
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def backend_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=3) as response:
            body = response.read(2048).decode("utf-8", "replace")
        return response.status == 200 and '"ready"' in body
    except Exception:
        return False


def copy_env_template(src: Path, dst: Path) -> bool:
    if src.exists() and not dst.exists():
        shutil.copyfile(src, dst)
        return True
    return False


class Service:
    def __init__(self, name: str, tag: str, argv: Sequence[str], cwd: Path) -> None:
        self.name = name
        self.tag = tag
        self.argv = list(argv)
        self.cwd = cwd
        self.proc: Optional[subprocess.Popen] = None

    def spawn(self) -> None:
        kwargs: Dict[str, object] = {
            "cwd": str(self.cwd),
            "env": {**os.environ, "PYTHONUNBUFFERED": "1", "FORCE_COLOR": "1"},
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
        }
        if IS_WIN:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        self.proc = subprocess.Popen(self.argv, **kwargs)  # type: ignore[arg-type]
        threading.Thread(target=self._pump, name=f"{self.name}-out", daemon=True).start()

    def _pump(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.rstrip()
            if line:
                say(self.tag, line)

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        say(self.tag, f"{C.yellow}stopping{C.reset}")
        if IS_WIN:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                capture_output=True,
            )
        else:
            for sig, grace in ((signal.SIGTERM, 5), (signal.SIGKILL, 2)):
                try:
                    os.killpg(os.getpgid(self.proc.pid), sig)
                except (ProcessLookupError, PermissionError):
                    return
                try:
                    self.proc.wait(timeout=grace)
                    return
                except subprocess.TimeoutExpired:
                    continue


def wait_for(label: str, check, timeout: int, tag: str) -> bool:
    say(tag, f"waiting for {label} (first boot loads the ML models, this can take a minute)...")
    start = time.time()
    warned = False
    while time.time() - start < timeout:
        if check():
            say(tag, f"{C.green}ready{C.reset}")
            return True
        if not warned and time.time() - start > 20:
            warned = True
            say(tag, f"{C.dim}still starting...{C.reset}")
        time.sleep(1.0)
    say(tag, f"{C.yellow}not ready yet after {timeout}s - it may still be loading, watch the log above{C.reset}")
    return False


def run_setup(python_exe: Path) -> None:
    step("Installing backend dependencies")
    subprocess.run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"], cwd=str(ROOT), check=True)
    subprocess.run(
        [str(python_exe), "-m", "pip", "install", "-r", str(BACKEND / "requirements.txt")],
        cwd=str(ROOT),
        check=True,
    )
    step("Installing frontend dependencies")
    npm = "npm.cmd" if IS_WIN else "npm"
    subprocess.run([npm, "install"], cwd=str(FRONTEND), check=True)
    print()


def preflight(args: argparse.Namespace) -> Path:
    step("Checking prerequisites")
    if not (BACKEND / "run_pipeline.py").exists() or not (FRONTEND / "package.json").exists():
        fail("Run this script from inside the vitascan_v3 checkout.", f"Expected {ROOT}")

    python_exe = resolve_python()
    if python_exe is None:
        fail("No Python interpreter found.", "Install Python 3.12 and re-run.")
    if not venv_python().exists():
        hint = WINDOWS_PYTHON_HINT if IS_WIN else "python3 -m venv backend/venv"
        fail(
            "backend/venv is missing.",
            f"Create it with: {hint}   (or run: python start.py --setup)",
        )
    say(BACKEND_TAG, f"python {python_exe}")

    if args.backend_only:
        return python_exe

    npm = shutil.which("npm.cmd" if IS_WIN else "npm")
    if npm is None:
        fail("npm was not found on PATH.", "Install Node.js 20 LTS and reopen the terminal.")
    node = shutil.which("node")
    if node:
        version = subprocess.run([node, "-v"], capture_output=True, text=True).stdout.strip()
        say(FRONTEND_TAG, f"node {version}")
    if not (FRONTEND / "node_modules").exists():
        fail("frontend/node_modules is missing.", "Run: cd frontend && npm install   (or: python start.py --setup)")
    if args.prod and not (FRONTEND / ".next" / "BUILD_ID").exists():
        fail(
            "No production build found in frontend/.next.",
            "Build it first with: cd frontend && npm run build   (or: python start.py --prod --build)",
        )

    wanted: List[tuple] = []
    if not args.frontend_only:
        wanted.append((args.backend_port, "backend", "--backend-port"))
    if not args.backend_only:
        wanted.append((args.frontend_port, "frontend", "--frontend-port"))
    for port, name, flag in wanted:
        if port_is_open("127.0.0.1", port):
            fail(
                f"Port {port} ({name}) is already in use{describe_port_owner(port)}.",
                f"Stop that process, or pass {flag} to use another one.",
            )
    print()
    return python_exe


def build_frontend() -> None:
    step("Building the frontend")
    subprocess.run(["npm.cmd" if IS_WIN else "npm", "run", "build"], cwd=str(FRONTEND), check=True)
    print()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Start the VitaScan backend and frontend together.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--setup", action="store_true", help="pip install + npm install before starting")
    parser.add_argument("--prod", action="store_true", help="serve the built frontend (npm start) instead of dev")
    parser.add_argument("--build", action="store_true", help="npm run build before starting (implies --prod)")
    parser.add_argument("--backend-only", action="store_true", help="start only the backend")
    parser.add_argument("--frontend-only", action="store_true", help="start only the frontend")
    parser.add_argument("--backend-port", type=int, default=int(os.environ.get("BACKEND_PORT", 8000)))
    parser.add_argument("--frontend-port", type=int, default=int(os.environ.get("FRONTEND_PORT", 3000)))
    parser.add_argument(
        "--timeout",
        type=int,
        default=240,
        help="seconds to wait for each service to answer before continuing anyway",
    )
    args = parser.parse_args(argv)

    if args.setup:
        run_setup(resolve_python() or Path("python"))
    if args.build:
        args.prod = True

    python_exe = preflight(args)

    step("Environment")
    if not args.frontend_only and copy_env_template(BACKEND / ".env.example", BACKEND / ".env"):
        warn("created backend/.env from the template - add your Groq keys to enable the AI features")
    if not args.backend_only and copy_env_template(FRONTEND / ".env.example", FRONTEND / ".env.local"):
        warn("created frontend/.env.local from the template")
    if not args.backend_only:
        say(FRONTEND_TAG, "frontend/.env.local present" if (FRONTEND / ".env.local").exists() else "no frontend/.env.local - the API URL may be wrong")
    print()

    if args.build:
        build_frontend()

    services: List[Service] = []
    if not args.frontend_only:
        services.append(
            Service(
                "backend",
                BACKEND_TAG,
                [str(python_exe), "-m", "uvicorn", "backend.run_pipeline:app",
                 "--host", "0.0.0.0", "--port", str(args.backend_port)],
                ROOT,
            )
        )
    if not args.backend_only:
        mode = "start" if args.prod else "dev"
        services.append(
            Service(
                "frontend",
                FRONTEND_TAG,
                ["npm.cmd" if IS_WIN else "npm", "run", mode, "--", "-p", str(args.frontend_port)],
                FRONTEND,
            )
        )

    step("Starting")
    for service in services:
        say(service.tag, f"$ {' '.join(service.argv)}")
        service.spawn()

    stopping = threading.Event()

    def shutdown(*_: object) -> None:
        if stopping.is_set():
            return
        stopping.set()
        print()
        for service in services:
            service.stop()

    signal.signal(signal.SIGINT, lambda *a: shutdown())
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *a: shutdown())

    if not args.frontend_only:
        wait_for(f"backend on :{args.backend_port}", lambda: backend_ready(args.backend_port),
                 args.timeout, BACKEND_TAG)
    if not args.backend_only:
        wait_for(f"frontend on :{args.frontend_port}",
                 lambda: http_ready(f"http://127.0.0.1:{args.frontend_port}/"),
                 args.timeout, FRONTEND_TAG)

    print()
    print(f"{C.bold}{C.green}VitaScan is running{C.reset}")
    if not args.backend_only:
        print(f"  {C.bold}App       {C.reset} http://localhost:{args.frontend_port}")
        print(f"  {C.dim}Upload a report at /upload to see the results page{C.reset}")
    if not args.frontend_only:
        print(f"  {C.bold}API       {C.reset} http://localhost:{args.backend_port}/status")
        print(f"  {C.dim}API docs at /docs{C.reset}")
    print(f"  {C.dim}Press Ctrl+C to stop everything{C.reset}")
    print(flush=True)

    try:
        while not stopping.is_set():
            for service in services:
                code = service.proc.poll() if service.proc else None
                if code is not None:
                    say(service.tag, f"{C.red}exited with code {code}{C.reset}")
                    shutdown()
                    return code or 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
