from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name='funcsforprajay',
    version='0.1',
    description='essential funcs written in python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://github.com/{.....}',
    author='Prajay Shah',
    author_email='prajay.shah95@gmail.com',
    license='MIT',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.6',
    zip_safe=False
)
