from setuptools import setup, find_packages

setup(
    name='torch-deep-danbooru',
    version='0.0.2',
    description='',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'Pillow',
    ],
)