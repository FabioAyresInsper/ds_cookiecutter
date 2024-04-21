# pylint:disable=missing-docstring
import subprocess
from pathlib import Path


def main():
    project_dir = Path.cwd()
    data_dir = project_dir / 'data'
    data_dir.mkdir(exist_ok=True)

    subprocess.run(['git', 'init'], check=True)
    subprocess.run(['git', 'add', '.'], check=True)
    subprocess.run(['git', 'commit', '-m', 'Initial commit'], check=True)


if __name__ == '__main__':
    main()
