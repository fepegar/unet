from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='unet',
    version='0.1.1',
    author='Fernando Perez-Garcia',
    author_email='fernando.perezgarcia.17@ucl.ac.uk',
    description='PyTorch implementation of UNet',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/fepegar/unet',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
    ],
)
