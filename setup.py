#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'torch',
]

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Fernando Perez-Garcia",
    author_email='fernando.perezgarcia.17@ucl.ac.uk',
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="PyTorch implementation of 2D and 3D U-Net",
    entry_points={
        'console_scripts': [
            'unet=unet.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='unet',
    name='unet',
    packages=find_packages(include=['unet', 'unet.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/fepegar/unet',
    version='0.7.7',
    zip_safe=False,
)
