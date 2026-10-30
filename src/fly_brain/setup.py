from setuptools import find_packages, setup

package_name = 'fly_brain'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    package_data={'fly_brain': ['py.typed', 'circuit.json']},
    include_package_data=True,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='KedarPanchal',
    maintainer_email='kedar.t.panchal@outlook.com',
    description=('Connectome-constrained fly brain and its sensory/motor decoders '
                 '(no ROS dependency).'),
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
