import os
import shutil
from pathlib import Path

PROJECT_DIRECTORY = os.path.realpath(os.path.curdir)

if __name__ == '__main__':
    # Copy
    if not '{{cookiecutter.binaries_path}}':
        raise ValueError('Binaries path variable is empty')

    shutil.copytree(
        '{{cookiecutter.binaries_path}}',
        Path(PROJECT_DIRECTORY) / '{{cookiecutter.project_name}}' / '{{cookiecutter.entrypoint.dir}}' / 'binaries'
    )
