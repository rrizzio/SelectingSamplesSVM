# -*- coding: utf-8 -*-
"""
@author: rrizzio@rrizzio.com
"""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SelectingSamplesSVM-RRIZZIO", 
    version="0.0.1",
    author="RRIZZIO",
    author_email="rrizzio@rrizzio.com",
    description="A simple example of data selection for SVM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rrizzio/SelectingSamplesSVM",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)


