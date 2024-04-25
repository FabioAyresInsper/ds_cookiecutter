# pylint: disable=missing-docstring
import subprocess
from pathlib import Path


def main():

    project_name = '{{cookiecutter.project_slug}}'

    source_dir = Path(__file__).parents[2]
    target_dir = f'/home/user/{project_name}'
    subprocess.run(
        [
            'docker',
            'run',
            '-it',
            '--mount',
            f'type=bind,source={source_dir},target={target_dir}',
            '--name',
            'projeto',
            'projeto',
        ],
        check=False,
    )


if __name__ == "__main__":
    main()
