from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'hb_task2a'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*'))
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TODO: name',
    maintainer_email='shreyasdas54@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'feedback = hb_task2a.feedback:main',
            'controller = hb_task2a.controller:main',
            'service = hb_task2a.service_node:main'
         ],
    },
)
