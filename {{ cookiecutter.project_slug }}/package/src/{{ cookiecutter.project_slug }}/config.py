# pylint:disable=missing-docstring
from pathlib import Path


def get_config():
    # project_dir = Path(__file__).parent.parent.parent
    project_dir = Path(__file__).parents[3]
    data_dir = project_dir / 'data'
    config = {
        'PROJECT_DIR': project_dir,
        'DATA_DIR': data_dir,
    }
    return config
