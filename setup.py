import io
import os
import re

from setuptools import find_packages
from setuptools import setup
import versioneer


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    name="obsarray",
    url="https://github.com/comet-toolkit/obsarray",
    license="GPLv3",
    author="CoMet Toolkit Team",
    author_email="team@comet-toolkit.org",
    description="Measurement data handling in Python",
    long_description=read("README.md"),
    packages=find_packages(exclude=("tests",)),
    install_requires=["numpy", "xarray", "comet_maths"],
    extras_require={
        "dev": [
            "pre-commit",
            "tox",
            "sphinx",
            "sphinx_design",
            "ipython",
            "sphinx_autosummary_accessors",
        ]
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
