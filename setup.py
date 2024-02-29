from setuptools import setup

setup(name='gym-rotor',
      version='0.0.2',
      url='https://github.com/fdcl-gwu/gym-rotor',
      install_requires=[
            'gymnasium',
            'numpy',
            'scipy',
            'torch',
            'vpython',
            'matplotlib',
      ] # And any other dependencies
)  