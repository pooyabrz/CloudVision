import argparse
from pathlib import Path

from src.data import audit
from src.schema import ROOT, write_json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "data/processed")
    p.add_argument("--out", type=Path, default=ROOT / "experiments/data_audit.json")
    a = p.parse_args()
    r = audit(a.data)
    write_json(a.out, r)
    for name, s in r["splits"].items():
        print(name, s["scenes"], "scenes,", s["days"], "UTC days")
