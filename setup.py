from setuptools import setup

package_name = 'ros_bridge_gym'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TS, Yin',
    maintainer_email='tianshi.yin@gmail.com',
    description='OpenAI Gym with ROS2-to-ROS1 bridge',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lab_ddpg = ros_bridge_gym.lab_ddpg:main',
            'gym_ddpg = ros_bridge_gym.gym_ddpg:main'
        ],
    },
)
