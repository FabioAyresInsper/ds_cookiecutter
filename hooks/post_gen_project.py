# pylint:disable=missing-docstring
from pathlib import Path


def main():
    project_dir = Path.cwd()
    data_dir = project_dir / 'data'

    data_dir.mkdir(exist_ok=True)

    with open('.env', 'w', encoding='utf8') as f:
        f.write(f'PROJECT_DIR={project_dir}\n')
        f.write(f'DATA_DIR={data_dir}\n')


if __name__ == '__main__':
    main()
