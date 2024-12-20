from setuptools import find_packages, setup

from typing import List

#HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
    
    return requirements

setup(
    name='mlprojects',
    version='0.0.1',
    description='Exercices for ML projects',
    author='Laurent Sturm',
    author_email='l.sturm@free.fr',
    url='https://github.com/Monvieux/mlproject_2/',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
     )