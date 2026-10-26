#!/usr/bin/env python3
"""Write the validation-frozen transparent fusion policy."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from ids.utils.io import read_yaml, write_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fusion.yaml")
    parser.add_argument("--output", default="artifacts/thresholds/fusion.yaml")
    args = parser.parse_args()
    write_yaml(read_yaml(args.config), args.output)
    print(f"Copied validation fusion policy to {args.output}")


if __name__ == "__main__":
    main()
