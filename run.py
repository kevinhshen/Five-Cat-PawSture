"""
Friendly launcher for PawSture.

Starts the posture monitor from the project root.

Run web: python3 run.py --web
"""
import argparse
import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAIN_SCRIPT = ROOT / "Main Thing" / "main.py"
MPL_CACHE = ROOT / ".cache" / "matplotlib"


def parse_launcher_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch PawSture.",
        epilog=(
            "Any other options are passed to the posture monitor, such as "
            "--camera 1 or --port /dev/cu.usbmodem1101."
        ),
    )
    args, monitor_args = parser.parse_known_args()
    args.monitor_args = monitor_args
    return args


def main() -> int:
    os.chdir(ROOT)
    args = parse_launcher_args()
    print("[startup] Starting PawSture...", flush=True)
    MPL_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
    sys.argv = [str(MAIN_SCRIPT), *args.monitor_args]
    try:
        runpy.run_path(str(MAIN_SCRIPT), run_name="__main__")
    except RuntimeError as exc:
        print(f"[startup] {exc}")
        if "Could not open camera" in str(exc):
            print("[startup] Check that camera permission is allowed and no other app is using it.")
            print("[startup] You can try a different camera with: python3 run.py --camera 1")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
