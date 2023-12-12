from setuptools import setup, find_packages
from typing import List


hyphen_e = '-e .'
def get_requirements(file_path: str) -> List[str]:
    requirements = [] # creating an empty list
    with open(file_path) as f:# openning the file
        requirements = f.readlines() # reading the file and spliting the lines
        requirements = [req.replace('\n', '') for req in requirements] # replacing the new line with empty string

        if hyphen_e in requirements:
            requirements.remove(hyphen_e) # removing the hyphen_e from the requirements
    return requirements

setup(
    name='house_price_prediction', # name of the package
    author_email='hjskhan47@gmail.com', # author email
    author='Hamza Jamal', # author name
    version='0.0.1', # version of the package
    description='House Price Prediction', # short description
    packages=find_packages(), # list of all packages
    install_requires=get_requirements('requirements.txt') # installing the requirements
)

    
    