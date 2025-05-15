from setuptools import setup, find_packages

setup(name='CliffordBuilder', packages=find_packages(include=["stabiliser_suite"]), version="0.0.1",
      description="Circuit sampler from Clifford group",
      author="Anton Perepelenko", author_email="anton.perepelenko@achaad.eu",
      maintainer_email="anton.perepelenko@achaad.eu",
      install_requires=[
          "qiskit>=1.3",
          "numpy>=2.2.0",
          "matplotlib>=3.10.0",
          "pylatexenc>=2.10",
          "termtables>=0.2.4",
          "numba>=0.61.0",
          "tqdm>=4.67.0"
      ],
      setup_requires=["pytest-runner"],
      tests_require=[
          "pytest",
          "tox>=4.23.2",
          "coverage>=7.6.9",
          "sympy>=1.13.3"],
      test_suite="tests", )