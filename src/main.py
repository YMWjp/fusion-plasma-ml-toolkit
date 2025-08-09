from __future__ import annotations

import argparse
from src.config.settings import (
    get_parameters, get_basic_info_for_header, load_config
)
from src.utils.utils import write_csv_header, get_dataset_path
from src.application.collect_plasma_data import run_plasma_collection
from src.application.native_pipeline import run_native_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run plasma data collection via src stack')
    parser.add_argument('--mode', '-m', choices=['manual', 'automatic'], help='Detection mode')
    parser.add_argument('--method', choices=['derivative', 'threshold', 'peak'], help='Automatic detection method')
    parser.add_argument('--engine', choices=['legacy', 'native'], default='native', help='Processing engine')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    headers = get_basic_info_for_header() + get_parameters()
    config = load_config()
    output_dataset = config['files']['output_dataset']
    dataset_file_path = get_dataset_path(output_dataset)

    # CSV ヘッダを初期化
    write_csv_header(dataset_file_path, headers, overwrite=True)

    if args.engine == 'legacy':
        run_plasma_collection(
            output_filename=output_dataset,
            detection_mode=args.mode,
            method=args.method,
        )
    else:
        run_native_pipeline(detection_mode=args.mode, method=args.method)


if __name__ == '__main__':
    main()