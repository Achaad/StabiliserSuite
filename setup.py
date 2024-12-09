from setuptools import setup, find_packages

setup(name='CliffordBuilder', packages=find_packages(include=["clifford_builder"]), version="0.0.1",
      description="Circuit sampler from Clifford group",
      author="Anton Perepelenko", author_email="anton.perepelenko@achaad.eu",
      maintainer_email="anton.perepelenko@achaad.eu",
      install_requires=[],
      setup_requires=["pytest-runner"],
      tests_require=["pytest"],
      test_suite="tests",)
