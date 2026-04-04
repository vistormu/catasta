from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        capture_output=capture_output,
    )
    if process.returncode != 0:
        raise RuntimeError(f"command failed ({process.returncode}): {' '.join(cmd)}")
    return process


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    dist = root / "dist"
    wheels = sorted(dist.glob("catasta-*.whl"), key=lambda path: path.stat().st_mtime)
    if not wheels:
        raise RuntimeError("no wheel found in dist/. run `python -m build` first.")

    wheel = wheels[-1]
    with tempfile.TemporaryDirectory() as tmp_dir:
        venv_dir = Path(tmp_dir) / "venv"

        run([sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)])

        if sys.platform.startswith("win"):
            python_bin = venv_dir / "Scripts" / "python.exe"
        else:
            python_bin = venv_dir / "bin" / "python"

        run([str(python_bin), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(python_bin), "-m", "pip", "install", "--no-deps", str(wheel)])

        import_cmd = [
            str(python_bin),
            "-c",
            (
                "import catasta; "
                "from catasta import Dataset, Scaffold, Archway, Foundation; "
                "from catasta.models import FeedforwardRegressor, GPRegressor, GPBinaryClassifier"
            ),
        ]
        import_process = subprocess.run(import_cmd, check=False, text=True, capture_output=True)

        if import_process.returncode != 0 and "ModuleNotFoundError" in (import_process.stderr or ""):
            fallback_env = dict()
            fallback_env.update(os.environ)
            dependency_paths = [path for path in sys.path if path.endswith("site-packages")]
            if dependency_paths:
                fallback_env["PYTHONPATH"] = os.pathsep.join(dependency_paths)
                import_process = subprocess.run(import_cmd, check=False, text=True, capture_output=True, env=fallback_env)

        if import_process.returncode != 0:
            raise RuntimeError(
                "install smoke import check failed\n"
                f"stdout:\n{import_process.stdout}\n"
                f"stderr:\n{import_process.stderr}\n"
            )

    print("install smoke test passed")


if __name__ == "__main__":
    main()
