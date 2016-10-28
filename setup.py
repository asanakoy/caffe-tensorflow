from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='kaffe',
    description='Caffe-TensorFlow model converter.',
    license='MIT',
    keywords='machine-learning caffe tensorflow',
    packages=find_packages(),
    scripts=['convert.py'], install_requires=['numpy']
)