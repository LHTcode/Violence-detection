from setuptools import setup

def get_packages():
    pkg_list = ['data', 'model', 'tools', 'trainer', '']
    return pkg_list

setup(
    name="violence-detection",
    author="HanTingLi",
    author_email="LHT13535837097@163.com",
    packages=get_packages(),
    project_urls={
        "Source": "https://github.com/LHTcode/Violence-detection.git"
    }
)