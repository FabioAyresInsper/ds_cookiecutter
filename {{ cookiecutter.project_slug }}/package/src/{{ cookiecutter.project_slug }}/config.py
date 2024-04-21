# pylint:disable=missing-docstring
from pathlib import Path


def get_config():
    project_dir = Path(__file__).parent.parent.parent
    config = {
        'PROJECT_DIR': project_dir,
    }
    return config
