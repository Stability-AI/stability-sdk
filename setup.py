from setuptools import setup, find_packages

setup(
    name='stability-sdk',
    version='0.0.1',
    author='stability.ai',
    author_email='api@stability.ai',
    url='https://beta.dreamstudio.ai/',
    install_requires=[
        'Pillow',
        'grpcio',
        'grpcio-tools',
        'python-dotenv',
    ],
    packages=find_packages(
        where='src',
        include=['stability_sdk*'],
    ),
    package_dir = {"": "src"}
)