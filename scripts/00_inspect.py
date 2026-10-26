#!/usr/bin/env python3
"""Inspect raw CICIDS-style files and write a manifest."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from ids.data.inspect_raw import inspect_directory, inspect_path
from ids.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir")
    source.add_argument("--input")
    parser.add_argument("--output", default="data/manifests/raw_inspection.json")
    args = parser.parse_args()
    payload = inspect_directory(args.input_dir) if args.input_dir else {"files": [inspect_path(args.input)]}
    write_json(payload, args.output)
    print(f"Inspected {len(payload['files'])} file(s); wrote {args.output}")


if __name__ == "__main__":
    main()
