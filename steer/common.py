"""Paths, config loading, and JSONL helpers shared by every stage."""
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("STEER_OUT", ROOT / "out"))


def load_yaml(name):
    with open(ROOT / "configs" / name) as f:
        return yaml.safe_load(f)


def traits():
    return load_yaml("traits.yaml")["traits"]


def model_id(short, kind="steered"):
    """Resolve a short model name from configs/models.yaml; pass HF ids through."""
    table = load_yaml("models.yaml")[kind]
    return table[short]["hf"] if short in table else short


def grid(name):
    cfg = load_yaml("grid.yaml")
    g = dict(cfg["grids"][name])
    g.setdefault("max_new_tokens", cfg["max_new_tokens"])
    g.setdefault("temperature", cfg["temperature"])
    return g


def trait_file(trait, name):
    """name is 'extract_A', 'extract_B', 'extract_C', or 'eval'."""
    with open(ROOT / "data" / "traits" / trait / f"{name}.json") as f:
        return json.load(f)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # a job killed mid-write leaves one truncated last line; drop it
                continue
    return rows


class JsonlWriter:
    """Append-only writer; flushes each batch so a killed job loses little."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # if the previous run died mid-line, start on a fresh line
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                needs_newline = f.read(1) != b"\n"
        else:
            needs_newline = False
        self.f = open(self.path, "a")
        if needs_newline:
            self.f.write("\n")

    def write(self, rows):
        for r in rows:
            self.f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        self.f.close()
