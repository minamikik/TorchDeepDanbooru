from setuptools import setup, find_packages

setup(
    name='torch-deep-danbooru',
    version='1',
    description='',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'Pillow',
    ],
)