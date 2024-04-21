# pylint: disable=missing-docstring,import-error
import blablabla.config as cfg


def test_config():
    assert True


def main():
    config = cfg.get_config()
    print(config)


if __name__ == '__main__':
    main()
