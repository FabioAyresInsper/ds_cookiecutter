# pylint: disable=missing-docstring,import-error
from config import get_config


def test_config():
    assert True


def main():
    config = get_config()
    print(config)


if __name__ == '__main__':
    main()
