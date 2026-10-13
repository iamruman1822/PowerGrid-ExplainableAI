"""
PowerGrid-ExplainableAI - Unified Full-Stack Launcher
Launches both the Flask backend and React/Vite frontend with a single command.

Usage:
    python run.py
"""

import os
import sys
import shutil
import subprocess
import signal
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")

processes = []

def print_banner():
    banner = r"""
======================================================================
     ____                           ____      _     _ 
    |  _ \ _____      _____ _ __   / ___|_ __(_) __| |
    | |_) / _ \ \ /\ / / _ \ '__| | |  _| '__| |/ _` |
    |  __/ (_) \ V  V /  __/ |    | |_| | |  | | (_| |
    |_|   \___/ \_/\_/ \___|_|     \____|_|  |_|\__,_|
                  Explainable Microgrid AI
======================================================================
    Backend API:     http://localhost:5000/api
    Frontend UI:     http://localhost:5173
    Default Login:   demo@xplainable.ai / demo123
======================================================================
    Press Ctrl+C at any time to shut down all services cleanly.
======================================================================
"""
    print(banner)

def find_python_executable():
    """Find virtualenv python if available, otherwise use current python."""
    if sys.platform == "win32":
        venv_py = os.path.join(BACKEND_DIR, "venv", "Scripts", "python.exe")
    else:
        venv_py = os.path.join(BACKEND_DIR, "venv", "bin", "python")
        
    if os.path.exists(venv_py):
        return venv_py
    return sys.executable

def find_npm_executable():
    """Find npm executable across platforms."""
    npm_cmd = shutil.which("npm.cmd") if sys.platform == "win32" else shutil.which("npm")
    if not npm_cmd:
        npm_cmd = "npm"
    return npm_cmd

def cleanup(sig=None, frame=None):
    """Gracefully terminate child processes."""
    print("\n[Launcher] Shutting down PowerGrid services...")
    for proc in processes:
        if proc.poll() is None:
            try:
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                else:
                    proc.terminate()
            except Exception:
                pass
    print("[Launcher] All services stopped. Goodbye!")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    print_banner()

    # 1. Resolve runtimes
    py_exec = find_python_executable()
    npm_exec = find_npm_executable()
    
    print(f"[Launcher] Python runtime:  {py_exec}")
    print(f"[Launcher] Node/NPM runner: {npm_exec}")

    # 2. Check frontend dependencies
    node_modules = os.path.join(FRONTEND_DIR, "node_modules")
    if not os.path.exists(node_modules):
        print("\n[Launcher] Installing frontend packages (first-time run)...")
        subprocess.run([npm_exec, "install"], cwd=FRONTEND_DIR)

    # 3. Start Backend
    print("\n[Launcher] Starting Flask Backend on port 5000...")
    backend_env = os.environ.copy()
    backend_proc = subprocess.Popen(
        [py_exec, "app.py"],
        cwd=BACKEND_DIR,
        env=backend_env
    )
    processes.append(backend_proc)

    # Brief delay for backend to initialize database & tables
    time.sleep(2)

    # 4. Start Frontend
    print("[Launcher] Starting React + Vite Frontend on port 5173...")
    frontend_proc = subprocess.Popen(
        [npm_exec, "run", "dev"],
        cwd=FRONTEND_DIR
    )
    processes.append(frontend_proc)

    print("\n[Launcher] -> Both services are live!")
    print("[Launcher] -> Open http://localhost:5173 in your browser.\n")

    # Keep script alive and monitor processes
    try:
        while True:
            time.sleep(1)
            # If either process died unexpectedly, alert and exit
            if backend_proc.poll() is not None:
                print(f"[Launcher] Backend exited with code {backend_proc.poll()}")
                break
            if frontend_proc.poll() is not None:
                print(f"[Launcher] Frontend exited with code {frontend_proc.poll()}")
                break
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()

if __name__ == "__main__":
    main()
