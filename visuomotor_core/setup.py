from setuptools import find_packages, setup

setup(
    name='visuomotor_core',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision',
        'zarr',
        'numpy',
        'pyyaml',
    ],
    author='Brayan Durán Toconás',
    author_email='brayandurantoconas@gmail.com',
    description='Off-ROS core module for generative visuomotor policies (Diffusion, Flow Matching).',
    license='Apache-2.0',
)
