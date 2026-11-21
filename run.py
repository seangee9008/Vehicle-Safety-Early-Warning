"""One entry point for common project tasks. See AGENTS.md (Commands).

    python run.py check | data | dbt | test | lint | waymo | briefs | status | contracts | publish | snapshot
    python run.py pipeline | all    (rebuild everything in order; --dry-run lists the steps)
    python run.py demo              Streamlit demo (VSE_DATA=local to skip BigQuery)

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
BRIEF_REQUEST_CAP = 60
COMPLAINT_YEARS = ["2010-2014", "2015-2019", "2020-2024", "2025-2026"]
# Lint skips Angelina's cleaning scripts and notebook until their fixes land (same list as CI).
LINT_EXCLUDE = "notebooks,scripts/nhtsa_data_cleaning.py,scripts/nhtsa_data_table_cleaning.py"


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
    return _run([PY, "-m", "ruff", "check", ".", "--extend-exclude", LINT_EXCLUDE])


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
    # Justine's split needs complaints from 2010 on; files already downloaded are skipped.
    code = _run([PY, "scripts/download_nhtsa.py", "--complaints", *COMPLAINT_YEARS])
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
    # Watchlist via src/sources.py: Justine's main.watchlist (latest month, top 10) when built,
    # else the sample CSV. Cache on: existing briefs are revalidated, never forced.
    # The request cap keeps one run inside the Gemini free tier; unfinished briefs wait.
    return _run([PY, "-m", "src.agent.run", "--top", "10", "--max-requests", str(BRIEF_REQUEST_CAP)])


def cmd_contracts(_: argparse.Namespace) -> int:
    if not (ROOT / "src" / "contracts.py").exists():
        return _not_merged("contracts")
    return _run([PY, "-m", "src.contracts", "--check"])


def cmd_publish(args: argparse.Namespace) -> int:
    # Final small tables -> BigQuery sandbox. Dry run unless --publish is passed.
    return _run([PY, "scripts/publish_bigquery.py", *args.extra])


def _recall_step(module: str) -> list[str]:
    return [PY, "-m", f"src.recall_model.{module}", "--source", "stg"]


# Rebuild from the loaded raw tables (python run.py data) to the published results, using each
# owner's documented command (src/recall_model/README.md, src/trajectory/README.md).
PIPELINE = [
    ("clean: stg.complaints + stg.recalls (Angelina)", [PY, "scripts/nhtsa_data_table_cleaning.py"]),
    ("labels (Justine)", _recall_step("labels")),
    ("features: main.monthly_features (Justine)", _recall_step("features")),
    ("model + backtest (Justine)", _recall_step("model")),
    ("watchlist: main.watchlist (Justine)", _recall_step("watchlist")),
    ("results tables (Justine)", _recall_step("results")),
    ("contracts", [PY, "-m", "src.contracts", "--check"]),
    ("label audit", [PY, "-m", "src.recall_audit.label_audit"]),
    ("agent briefs (cache on, request cap)",
     [PY, "-m", "src.agent.run", "--top", "10", "--max-requests", str(BRIEF_REQUEST_CAP)]),
    ("Waymo: model vs baseline", [PY, "-m", "src.trajectory.evaluate", "--no-plot"]),
    ("Waymo: demo examples", [PY, "-m", "src.trajectory.demo_examples"]),
    ("BigQuery publish plan (dry run)", [PY, "scripts/publish_bigquery.py"]),
]


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Every step in order; stops at the first failure. Never uploads (publish is a dry run)."""
    for i, (name, cmd) in enumerate(PIPELINE, 1):
        print(f"\n[{i}/{len(PIPELINE)}] {name}", flush=True)
        if args.dry_run:
            print("  " + " ".join("python" if part == PY else part for part in cmd))
            continue
        if _run(cmd) != 0:
            print(f"\npipeline stopped at step {i} ({name}); fix it and rerun.")
            return 1
    if not args.dry_run:
        print("\nDone. Justine's model step rewrites her committed reports/recall_*_stg.csv with this "
              "run's numbers; `git checkout -- reports` restores hers. To upload: "
              "python run.py publish --publish")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    if not args.dry_run and cmd_check(args) != 0:
        print("\nall: setup/tests/lint failed; pipeline not run.")
        return 1
    return cmd_pipeline(args)


def _repo_venv_python() -> Path | None:
    venv = ROOT / ".venv"
    exe = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return exe if exe.exists() else None


def cmd_demo(args: argparse.Namespace) -> int:
    # Reads BigQuery via src/bq_reader.py, falls back to local files. Extra args go to streamlit.
    # Always this repo's .venv, which has duckdb, google-cloud-bigquery and plotly installed.
    venv_python = _repo_venv_python()
    if venv_python is None:
        print("No .venv in this repo (see SETUP.md); starting the demo with "
              f"{PY}, which may be missing duckdb / google-cloud-bigquery.")
        python = PY
    else:
        python = str(venv_python)
        if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
            print(f"run.py was started with {PY}, not this repo's .venv; "
                  f"starting the demo with {venv_python} instead.")
    return _run([python, "-m", "streamlit", "run", "app/streamlit_app.py", *args.extra])


def cmd_snapshot(args: argparse.Namespace) -> int:
    # Save every table the demo reads from BigQuery into app/snapshot/ (--final: use it for good).
    return _run([PY, "-m", "src.snapshot", *args.extra])


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
    if (ROOT / "src" / "sources.py").exists():
        print("\nTables (real = teammate's table passes its contract; else placeholder):")
        return _run([PY, "-m", "src.sources"])
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
    "contracts": cmd_contracts,
    "publish": cmd_publish,
    "pipeline": cmd_pipeline,
    "all": cmd_all,
    "demo": cmd_demo,
    "snapshot": cmd_snapshot,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Project tasks. See AGENTS.md.")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--reload", action="store_true", help="data: rebuild data/nhtsa.duckdb")
    parser.add_argument("--dry-run", action="store_true", help="pipeline/all: list the steps only")
    args, extra = parser.parse_known_args()
    if extra and args.command not in ("publish", "demo", "snapshot"):
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    args.extra = extra
    raise SystemExit(COMMANDS[args.command](args))


if __name__ == "__main__":
    main()
