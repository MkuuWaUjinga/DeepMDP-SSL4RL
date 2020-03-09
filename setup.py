from setuptools import setup

reqs=[
    "garage==2019.10.1",
    "scipy",
    "numpy",
    "matplotlib",
    "tqdm",
    "attr",
    "sacred",
    "hashfs",
    "tinydb-serialization",
    "raylib",
    "seaborn",
    "jupyter",
    "ipython",
    "ipdb",
    "termcolor",
]

setup(
    name='deepmdp',
    version='0.0',
    packages=['deepmdp', 'deepmdp.experiments'],
    url='',
    license='',
    author='',
    author_email='',
    description='',
    install_requires=reqs
)
