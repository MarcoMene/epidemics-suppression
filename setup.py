#!/usr/bin/env python

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="epidemic-suppression-model-lib-python",
        version="0.0.1",
        # description="Library and repository for marketing-related tools and analyses.",
        # author="Andrea Maiorana",
        # author_email="anm@bendingspoons.com",
        # url="https://github.com/BendingSpoons/marketing-analytics-lib-python",
        packages=find_packages(exclude="demo"),
        install_requires=["numpy", "scipy", "matplotlib", "jupyter",],
        zip_safe=False,
    )
