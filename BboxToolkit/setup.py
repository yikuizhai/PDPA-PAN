from setuptools import setup, find_packages

setup(name='BboxToolkit',
      version='1.1',
      description='a tiny toolkit for special bounding boxes',
      author='XBY',
      packages=find_packages(),
      install_requires=[
          'matplotlib>=3.4',
          'opencv-python',
          'terminaltables',
          'shapely',
          'tqdm',
          'scipy',
          'numpy'])
