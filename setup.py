import os
import re
import subprocess

from setuptools import Command, find_packages, setup


def read(fname):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), fname), "rt") as fh:
        return fh.read()

about: dict = {}
root_path = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(root_path, "src/textacy_corpora", "about.py")) as f:
    exec(f.read(), about) # pylint: disable=exec-used


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options: list = []
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')

setup(
    name=about['__title__'],
    maintainer=about['__maintainer__'],
    version=about['__version__'],
    maintainer_email=about['__maintainer_email__'],
    description=about['__description__'],
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    install_requires=[
        'spacy',
        'textacy',
        'cytoolz',
        'numpy'
    ],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires=">=3.6",
    keywords="textacy, spacy, nlp, text processing, linguistics",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Natural Language :: English",
        "Topic :: Text Processing :: Linguistic",
    ],
    cmdclass={
        'clean': CleanCommand,
    })
