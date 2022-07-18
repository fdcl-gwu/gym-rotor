from setuptools import setup

setup(name='gym-rotor',
      version='0.0.1',
      url='https://github.com/fdcl-gwu/gym-rotor',
      install_requires=[
            'gym',
            'numpy',
            'scipy',
            'torch',
            'vpython',
            'matplotlib',
      ] # And any other dependencies
)  