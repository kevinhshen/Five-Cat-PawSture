"""
Friendly launcher for PawSture.

Checks whether required Python packages are installed, offers to install any
missing ones, then starts the posture monitor.
"""
import argparse
import importlib.util
import os
import runpy
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAIN_SCRIPT = ROOT / "Main Thing" / "main.py"
REQUIREMENTS = ROOT / "requirements.txt"
MPL_CACHE = ROOT / ".cache" / "matplotlib"

REQUIRED_IMPORTS = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "serial": "pyserial",
    "ultralytics": "ultralytics",
}


def missing_packages() -> list[str]:
    missing = []
    for module_name, package_name in REQUIRED_IMPORTS.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)
    return missing


def install_requirements() -> None:
    print("[setup] Installing required packages. This may take a few minutes...")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(REQUIREMENTS),
    ])


def ask_to_install(packages: list[str]) -> bool:
    print("[setup] Missing required packages:")
    for package in packages:
        print(f"  - {package}")

    if not sys.stdin.isatty():
        print("\n[setup] Run this command to install them:")
        print(f"  {sys.executable} -m pip install -r \"{REQUIREMENTS}\"")
        return False

    answer = input("\nInstall them now? [Y/n] ").strip().lower()
    return answer in ("", "y", "yes")


def parse_launcher_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check setup requirements, then launch PawSture.",
        epilog=(
            "Any other options are passed to the posture monitor, such as "
            "--camera 1 or --port /dev/cu.usbmodem1101."
        ),
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install missing packages without asking first.",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Only report missing packages; do not install them.",
    )
    args, monitor_args = parser.parse_known_args()
    args.monitor_args = monitor_args
    return args


def main() -> int:
    args = parse_launcher_args()
    packages = missing_packages()

    if packages:
        should_install = args.install
        if args.no_install:
            should_install = False
        elif not should_install:
            should_install = ask_to_install(packages)

        if not should_install:
            return 1

        try:
            install_requirements()
        except subprocess.CalledProcessError as exc:
            print(f"[setup] Install failed with exit code {exc.returncode}.")
            print(f"[setup] Try running: {sys.executable} -m pip install -r \"{REQUIREMENTS}\"")
            return exc.returncode

    print("[setup] Requirements ready. Starting PawSture...", flush=True)
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
