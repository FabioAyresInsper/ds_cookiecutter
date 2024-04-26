# pylint: disable=missing-docstring
import subprocess


def main():
    subprocess.run(
        [
            'docker',
            'rmi',
            '{{ cookiecutter.project_slug }}',
        ],
        check=False,
    )


if __name__ == '__main__':
    main()
