import tempfile
from os.path import join, dirname
from pathlib import Path

import pydantic
from cookiecutter.main import cookiecutter
from odahuflow.gppi.python.creator import create_python_library


class ModelLibraryEntrypoint(pydantic.BaseModel):
    file: str = "entrypoint"
    dir: str


class ModelLibraryPackage(pydantic.BaseModel):
    name: str = "model"
    version: str = "1.0.0"


class ModelLibraryInfo(pydantic.BaseModel):
    project_name: str = "model"
    model_entrypoint_name: str
    entrypoint: ModelLibraryEntrypoint
    package: ModelLibraryPackage
    binaries_path: Path

    # TODO: lol, rewrite this))
    # need to convert this to the dict only one level. dict() method convert recursively
    def items(self):
        for k, _ in self.dict().items():
            yield k, self.__dict__[k]


def generate_model_library(cookiecutter_context: ModelLibraryInfo, output: Path, template=None):
    """
    Generate a typical model library that contains one exported model and set of binaries

    :param cookiecutter_context:
    :param output:
    :param template:
    """
    # TODO: replace with importlib_resources
    template = template or join(dirname(__file__), 'library_template')

    with tempfile.TemporaryDirectory() as temp_library_dir:
        cookiecutter(
            template,
            no_input=True,
            extra_context=cookiecutter_context.dict(),
            output_dir=temp_library_dir,
        )

        create_python_library(Path(temp_library_dir) / cookiecutter_context.project_name, output)
