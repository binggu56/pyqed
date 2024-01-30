#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=6.0', 'numpy', 'scipy' ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Bing Gu",
    author_email='binggu56@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description=" ultrafast spectroscopy, cavity quantum electrodynamics",
    entry_points={
        'console_scripts': [
            'pyqed=pyqed.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pyqed',
    name='pyqed',
    packages=find_packages(),
#    packages=['pyqed.qchem'],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
#    package_dir={'': 'pyqed'},
    url='https://github.com/binggu56/pyqed',
    version='0.1.3',
    zip_safe=False,
)
