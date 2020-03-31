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
    "visdom"
]

setup(
    name='deepmdp',
    version='0.1.4',
    packages=['deepmdp', 'deepmdp.experiments', 'deepmdp.garage_mod', 'deepmdp.garage_mod.algos',
              'deepmdp.garage_mod.env_wrappers', 'deepmdp.garage_mod.policies', 'deepmdp.garage_mod.q_functions'],
    url='',
    license='',
    author='',
    author_email='',
    description='',
    install_requires=reqs
)
