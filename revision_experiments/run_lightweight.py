from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def run(script: str, *args: str) -> None:
    command = [sys.executable, str(SCRIPT_DIR / script), *args]
    print("\n$ " + " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    run("eval_baselines.py")
    run("make_comparison_table.py")
    run("complexity_analysis.py", "--skip-timing")
    print("\nLightweight revision experiments completed.")


if __name__ == "__main__":
    main()
