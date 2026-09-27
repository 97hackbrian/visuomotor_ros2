from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'visuomotor_data_collection'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Brayan Durán Toconás',
    maintainer_email='brayandurantoconas@gmail.com',
    description='Data ingestion and processing for generative visuomotor policies.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sim_collector = visuomotor_data_collection.sim_collector_node:main',
            'pickup_approximate = visuomotor_data_collection.pickup_approximate:main',
            'bag_converter = visuomotor_data_collection.bag_converter_node:main',
        ],
    },
)
