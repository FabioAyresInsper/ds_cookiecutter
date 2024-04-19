# pylint:disable=missing-docstring
from pathlib import Path


def main():
    with open('.env', 'w', encoding='utf8') as f:
        f.write(f'PROJECT_DIR={Path.cwd()}\n')


if __name__ == '__main__':
    main()
