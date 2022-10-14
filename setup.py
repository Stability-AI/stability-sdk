# fmt: off

from setuptools import setup, find_packages

with open('README.md','r') as f:
    README = f.read()

setup(
    name='stability-sdk',
    version='0.2.6',
    author='Wes Brown',
    author_email='wesbrown18@gmail.com',
    maintainer='David Marx',
    maintainer_email='david@stability.ai',
    url='https://beta.dreamstudio.ai/',
    download_url='https://github.com/Stability-AI/stability-sdk/',

    description='Python SDK for interacting with stability.ai APIs',
    long_description=README,
    long_description_content_type="text/markdown",

    install_requires=[
        'Pillow',
        'grpcio==1.48.1',
        'grpcio-tools==1.48.1',
        'python-dotenv',
        'protobuf==3.19.5'
    ],
    packages=find_packages(
        where='src',
        include=['stability_sdk*'],
    ),
    package_dir = {"": "src"},

    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'Topic :: Artistic Software',
        'Topic :: Education',
        'Topic :: Games/Entertainment',
        'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
        'Topic :: Internet :: WWW/HTTP :: Dynamic Content :: CGI Tools/Libraries',
        'Topic :: Multimedia',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Multimedia :: Graphics :: 3D Rendering',
        'Topic :: Multimedia :: Graphics :: Editors',
        'Topic :: Multimedia :: Graphics :: Editors :: Raster-Based',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Human Machine Interfaces',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    keywords=[],
    license='MIT',
)
