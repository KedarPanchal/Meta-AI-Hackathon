from setuptools import find_packages, setup

package_name = 'fly_drone'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['../../launch/launch-fly.yaml']),
    ],
    package_data={'fly_drone': ['py.typed']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='KedarPanchal',
    maintainer_email='kedar.t.panchal@outlook.com',
    description='ROS 2 node, detector and vehicle bridges that fly a multirotor with fly_brain.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'drone_controller = fly_drone.drone_controller:main',
            'brain_pilot = fly_drone.node:main',
        ],
    },
)
