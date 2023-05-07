# fmt: off

from setuptools import setup, find_namespace_packages

with open('README.md','r') as f:
    README = f.read()

setup(
    name='stability-sdk',
    version='0.8.0',
    author='Stability AI',
    author_email='support@stability.ai',
    maintainer='Stability AI',
    maintainer_email='support@stability.ai',
    requires=['python (>=3.8.0)'],
    url='https://beta.dreamstudio.ai/',
    download_url='https://github.com/Stability-AI/stability-sdk/',

    description='Python SDK for interacting with Stability AI APIs',
    long_description=README,
    long_description_content_type="text/markdown",

    install_requires=[
        'Pillow',
        'grpcio>=1.49.0',
        'grpcio-tools>=1.49.0',
        'pydantic>=1.9.2,<2.0',
        'python-dotenv',
        'protobuf>=3.19,<5.0',
        'sagemaker>=2.112.2',
        'tensorizer>=1.1.0',        
    ],
    extras_require={
        'dev': [
            'pytest',
            'grpcio-testing'
    ]},
    packages=find_namespace_packages(
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
