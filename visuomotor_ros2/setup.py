from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'visuomotor_ros2'

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
    description='Inference runtime for generative visuomotor policies.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'policy_node = visuomotor_ros2.policy_node:main',
        ],
    },
)
