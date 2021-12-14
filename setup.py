#!/usr/bin/env python3
from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='pycoordinates',
    version="0.0.1",
    author='pycoordinates contributors',
    author_email='gpulkin@gmail.com',
    packages=find_packages(),
    setup_requires="pytest-runner",
    tests_require="pytest",
    data_files=["requirements.txt"],
    description='Create and manipulate coordinates in vector bases',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=requirements,
)
