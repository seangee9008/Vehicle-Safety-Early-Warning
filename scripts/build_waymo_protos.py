"""Compile Waymo's Scenario .proto files into Python modules (run once after pip install).

This lets us read Waymo Motion data with plain Python + protobuf, on Mac, Windows or Linux,
without TensorFlow or the Linux-only `waymo-open-dataset` package.

The .proto files in third_party/waymo_protos/ are copied from
https://github.com/waymo-research/waymo-open-dataset (Apache 2.0, see the LICENSE there).

Usage:
    python scripts/build_waymo_protos.py
"""
import sys
from pathlib import Path

from grpc_tools import protoc

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "third_party" / "waymo_protos"
OUT = ROOT / "src" / "waymo_pb"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protos = sorted(str(p.relative_to(SRC)) for p in SRC.rglob("*.proto"))
    include = Path(protoc.__file__).parent / "_proto"          # google/protobuf/*.proto
    rc = protoc.main(["protoc", f"-I{SRC}", f"-I{include}", f"--python_out={OUT}", *protos])
    if rc != 0:
        sys.exit(f"protoc failed with exit code {rc}")
    for d in [OUT, *[p for p in OUT.rglob("*") if p.is_dir()]]:
        (d / "__init__.py").touch()
    print(f"Compiled {len(protos)} protos into {OUT}")


if __name__ == "__main__":
    main()
