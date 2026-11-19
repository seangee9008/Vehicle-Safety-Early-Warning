"""Check tables against the hand-off contracts in contracts/contracts.yml.

    python -m src.contracts --check     # exit 1 if any real table breaks its contract

A table passes when it exists, has rows, and has every contract column with a compatible
type. Extra columns are fine. A missing real table is reported, not a failure:
src/sources.py falls back to a placeholder for it.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS_PATH = ROOT / "contracts" / "contracts.yml"

_INTEGER = {
    "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
    "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT",
}
_FLOAT = {"FLOAT", "DOUBLE", "REAL"}
TYPE_FAMILIES: dict[str, set[str]] = {
    "text": {"VARCHAR"},
    "integer": _INTEGER,
    "number": _INTEGER | _FLOAT | {"DECIMAL"},
    "date": {"DATE", "TIMESTAMP", "TIMESTAMP_NS", "TIMESTAMP_MS", "TIMESTAMP_S",
             "TIMESTAMP WITH TIME ZONE"},
    "boolean": {"BOOLEAN"},
}


def load_contracts(path: Path = CONTRACTS_PATH) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def _base_type(duckdb_type: str) -> str:
    return re.sub(r"\(.*\)$", "", str(duckdb_type).upper()).strip()


@dataclass
class ContractResult:
    name: str
    relation: str
    exists: bool
    rows: int = 0
    missing: list[str] = field(default_factory=list)
    wrong_type: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.exists and self.rows > 0 and not self.missing and not self.wrong_type

    def describe(self) -> str:
        if not self.exists:
            return f"{self.name}: {self.relation} not found"
        problems = []
        if self.missing:
            problems.append(f"missing {', '.join(self.missing)}")
        if self.wrong_type:
            problems.append(
                "wrong type " + ", ".join(f"{c} ({t})" for c, t in self.wrong_type.items())
            )
        if self.rows == 0:
            problems.append("no rows")
        status = "ok" if self.ok else "FAILS: " + "; ".join(problems)
        return f"{self.name}: {self.relation} {self.rows:,} rows, {status}"


def relation_exists(con: duckdb.DuckDBPyConnection, relation: str) -> bool:
    schema, _, table = relation.rpartition(".")
    return bool(
        con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_schema = ? AND table_name = ?",
            [schema or "main", table],
        ).fetchall()
    )


def check_contract(
    con: duckdb.DuckDBPyConnection,
    name: str,
    relation_sql: str | None = None,
    contracts: dict[str, dict[str, Any]] | None = None,
) -> ContractResult:
    """Check one contract against its real table, or against any SELECT (``relation_sql``)."""
    contract = (contracts or load_contracts())[name]
    relation = contract["table"]
    if relation_sql is None:
        if not relation_exists(con, relation):
            return ContractResult(name, relation, exists=False)
        relation_sql = f"SELECT * FROM {relation}"
    else:
        relation = "placeholder"
    described = con.sql(f"DESCRIBE {relation_sql}").fetchall()
    have = {row[0]: _base_type(row[1]) for row in described}
    result = ContractResult(name, relation, exists=True)
    for column, family in contract["columns"].items():
        if column not in have:
            result.missing.append(column)
        elif have[column] not in TYPE_FAMILIES[family]:
            result.wrong_type[column] = f"{have[column]}, expected {family}"
    result.rows = con.sql(f"SELECT count(*) FROM ({relation_sql})").fetchall()[0][0]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="check every real table")
    parser.parse_args()
    from src import sources

    contracts = load_contracts()
    failed = False
    with sources.connect() as con:
        for name in contracts:
            result = check_contract(con, name, contracts=contracts)
            if result.exists and not result.ok:
                failed = True
            note = "" if result.exists else " (sources.py uses the placeholder)"
            print(result.describe() + note)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    sys.exit(main())
