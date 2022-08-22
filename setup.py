from setuptools import setup, find_packages

setup(
    name='stability-sdk',
    version='0.0.1',
    ########################
    install_requires=[
        'Pillow',
        'grpcio',
        'grpcio-tools',
        'python-dotenv',
    ],
    ########################
    packages=find_packages(
        where='src',
        include=['stability_sdk*'],
        #include=['pkg*'],
        #exclude=['additional*'],
    ),
    package_dir = {"": "src"}
    # directory containing all the packages (e.g.  src/mypkg, src/mypkg/subpkg1, ...)
)