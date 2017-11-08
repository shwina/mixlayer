from setuptools import setup

setup(name='mixlayer',
    version='0.1',
    description='A finite-difference code for analyzing mixing layer flow',
    url='github.com/shwina/mixlayer',
    author='Ashwin Srinath',
    py_modules='mixlayer.mixlayer',
    entry_points={
        'console_scripts': [
            'mixlayer = mixlayer.mixlayer:main',
            ],
        },
    zip_safe=False)
