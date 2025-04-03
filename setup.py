from setuptools import setup, find_packages

setup(
    name='pymcintegrate',
    version='0.1.0',
    description='An adaptive Monte Carlo integration and uncertainty quantification package.',
    author='Solai Ramu',
    author_email='solai.ramusaravanan@mail.utoronto.ca',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
