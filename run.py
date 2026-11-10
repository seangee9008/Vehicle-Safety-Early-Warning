"""One entry point for common project tasks. See AGENTS.md (Commands).

    python run.py check | data | dbt | test | lint | waymo | briefs | status
    python run.py contracts | pipeline | demo | all     (not built yet)

Standard library only, so it works on Mac and Windows before anything else is installed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
SAMPLE_WATCHLIST = Path("src/agent/fixtures/watchlist_sample.csv")


def _run(cmd: list[str], cwd: Path = ROOT) -> int:
    where = "" if cwd == ROOT else f"(in {cwd.relative_to(ROOT)}) "
    shown = " ".join("python" if part == PY else part for part in cmd)
    print(f"\n$ {where}{shown}", flush=True)
    return subprocess.call(cmd, cwd=cwd)


def _not_merged(name: str) -> int:
    print(f"{name}: not merged yet: see STATUS.md")
    return 0


def _not_built(name: str) -> int:
    print(f"{name}: not built yet: see STATUS.md")
    return 0


def _venv_tool(name: str) -> str | None:
    exe = name + (".exe" if os.name == "nt" else "")
    local = Path(PY).parent / exe
    return str(local) if local.exists() else shutil.which(name)


def cmd_test(_: argparse.Namespace) -> int:
    return _run([PY, "-m", "pytest", "-q"])


def cmd_lint(_: argparse.Namespace) -> int:
    return _run([PY, "-m", "ruff", "check", "."])


def cmd_check(args: argparse.Namespace) -> int:
    results = {
        "check_setup": _run([PY, "scripts/check_setup.py"]),
        "pytest": cmd_test(args),
        "ruff": cmd_lint(args),
    }
    print("\nSummary: " + ", ".join(
        f"{name} {'ok' if code == 0 else 'FAILED'}" for name, code in results.items()
    ))
    return 0 if all(code == 0 for code in results.values()) else 1


def cmd_data(args: argparse.Namespace) -> int:
    code = _run([PY, "scripts/download_nhtsa.py"])  # skips files already downloaded
    if code:
        return code
    db = ROOT / "data" / "nhtsa.duckdb"
    if db.exists() and not args.reload:
        print(f"\nskip load: {db.relative_to(ROOT)} already exists (use --reload to rebuild it)")
        return 0
    return _run([PY, "scripts/load_nhtsa.py"])


def cmd_dbt(_: argparse.Namespace) -> int:
    dbt_dir = ROOT / "dbt"
    profiles = dbt_dir / "profiles.yml"
    if not profiles.exists():
        shutil.copyfile(dbt_dir / "profiles.yml.example", profiles)
        print("copied dbt/profiles.yml.example -> dbt/profiles.yml")
    dbt = _venv_tool("dbt")
    if dbt is None:
        print("dbt not found: activate .venv and pip install -r requirements.txt", file=sys.stderr)
        return 1
    return _run([dbt, "build", "--profiles-dir", "."], cwd=dbt_dir)


def cmd_waymo(_: argparse.Namespace) -> int:
    if not (ROOT / "src" / "trajectory" / "baseline.py").exists():
        return _not_merged("waymo")
    return _run([PY, "-m", "src.trajectory.baseline"])


def cmd_briefs(_: argparse.Namespace) -> int:
    if not (ROOT / "src" / "agent" / "run.py").exists():
        return _not_merged("briefs")
    # Cache on: existing briefs are revalidated, never regenerated (no --force).
    return _run([PY, "-m", "src.agent.run", "--watchlist", str(SAMPLE_WATCHLIST), "--top", "5"])


def cmd_status(_: argparse.Namespace) -> int:
    lines = (ROOT / "STATUS.md").read_text(encoding="utf-8").splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("| Task")), None)
    if start is None:
        print("No task table found in STATUS.md", file=sys.stderr)
        return 1
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        print(line)
    return 0


COMMANDS = {
    "check": cmd_check,
    "data": cmd_data,
    "dbt": cmd_dbt,
    "test": cmd_test,
    "lint": cmd_lint,
    "waymo": cmd_waymo,
    "briefs": cmd_briefs,
    "status": cmd_status,
    **{name: (lambda _, n=name: _not_built(n)) for name in ("contracts", "pipeline", "demo", "all")},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Project tasks. See AGENTS.md.")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--reload", action="store_true", help="data: rebuild data/nhtsa.duckdb")
    args = parser.parse_args()
    raise SystemExit(COMMANDS[args.command](args))


if __name__ == "__main__":
    main()
