from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt') as req:
        return req.read().strip().split('\n')

setup(
    name='HackathonProject',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements(),
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'runhackathon=hackathon.server:main',
        ],
    }
)