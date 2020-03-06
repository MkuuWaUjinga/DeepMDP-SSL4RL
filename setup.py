from setuptools import setup

reqs=[
    "torch",
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
    "gym",
    "garage"
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
