#!/usr/bin/env python

from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="epidemic-suppression-model-lib-python",
        version="0.0.1",
        description="Repo to support custom epidemics suppression calculations, according to the model of this paper: <ref_paper>",
        author="Andrea Maiorana, Marco Meneghelli",
        author_email="anm@bendingspoons.com, mm@bendingspoons.com",
        url="https://github.com/MarcoMene/epidemics-suppression",
        packages=find_packages(exclude="demo"),
        install_requires=["numpy", "matplotlib"],
        zip_safe=False,
    )
