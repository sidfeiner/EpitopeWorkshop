from setuptools import setup, find_packages

version_info = (0, 0, 2)
version = '.'.join(map(str, version_info))

setup(
    name='epitope-workshop',
    version=version,
    url='https://github.com/sidfeiner/EpitopeWorkshop',
    license='For internal usage only',
    description='Epitope Workshop codebase',
    packages=find_packages(include='EpitopeWorkshop'),
)
