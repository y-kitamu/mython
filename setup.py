import os
from setuptools import setup, find_packages


def load_requirements(f):
    return list(
        filter(None,
               [l.split("#", 1)[0].strip() for l in open(os.path.join(os.getcwd(), f)).readlines()]))


init = os.path.join(os.path.dirname(__file__), "mython", "__init__.py")

version_line = list(filter(lambda l: l.startswith("VERSION"), open(init)))[0]
VERSION = ".".join(["{}".format(x) for x in eval(version_line.split("=")[-1])])

setup(
    name="mython",
    version=VERSION,
    url="https://github.com/y-kitamu/mython",
    install_requires=load_requirements("requirements.txt"),
    packages=find_packages(where="mython"),
)
