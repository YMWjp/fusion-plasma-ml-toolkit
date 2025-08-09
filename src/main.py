from __future__ import annotations

import argparse

from src.application.native_pipeline import run_native_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run plasma data collection via src stack')
    parser.add_argument('--mode', '-m', choices=['manual', 'automatic'], help='Detection mode (manual or automatic)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_native_pipeline(detection_mode=args.mode)

if __name__ == '__main__':
    main()
