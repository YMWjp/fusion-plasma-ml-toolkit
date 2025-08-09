# main.py
from src.config.settings import (
    get_shot_numbers, get_parameters, get_basic_info_for_header, load_config
)
from src.utils.utils import (
    write_csv_header, get_dataset_path, log_error_shot
)

def main() -> None:
    shot_numbers = get_shot_numbers()
    headers = get_basic_info_for_header() + get_parameters()
    config = load_config()
    output_dataset = config['files']['output_dataset']
    dataset_file_path = get_dataset_path(output_dataset)
    write_csv_header(dataset_file_path, headers, overwrite=True)
        


if __name__ == '__main__':
    main()