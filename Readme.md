# Repo Hygene

* use [git lfs](https://github.com/git-lfs/git-lfs/wiki/Installation) when keeping <50mb files within the repo (images, pdfs, small models)
* use [switch](https://drive.switch.ch/) or the EPFL gdrive for larger files
* don't commit junk files
* use latex/notes for math discussions, logbook in this readme to keep track of Todos/Dones

# General guidelines
* use [ray](https://ray.readthedocs.io/en/latest/rllib.html) as a starting block for distribution/working loop, [sacred](https://github.com/IDSIA/sacred) with the Filesystem or Tinydb observer to run experiments
* Jupyter Notebooks are okay, but don't commit any long analysis/hacking as final product. Exctract and move into the module (`mcts-mri`)
* write experiments in `mctis-mri/experiments`, put any scripts to launch them or do postprocessing in `scripts`
* Composition over inheritance. Use [attr.s](https://pypi.org/project/attrs/) to make data classes, this helps [mypy](http://mypy-lang.org/) (strong recommendation to use it with [pycharm-mypy](mcts_mri) to help you find bugs via [type hints](https://docs.python.org/3/library/typing.html)
* Use submodules to organize the code

# Logbook
## IK: 2020-02-25 post-discussion

- Tuesday 1630 regular meeting slot
- Adrian: till next week, set up benchmark environment with arbitrary model free RL algorithm
- Igor: this log add references to distributional RL paper to pdfs, [garage](https://github.com/rlworkgroup/garage)  and [rllib](https://github.com/ray-project/ray/tree/master/rllib)
