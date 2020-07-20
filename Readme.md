# Reproduction Study of DeepMDP-paper

* Reimplementation using torch, garage and gym.
* See [report](https://github.com/MkuuWaUjinga/deepmdp-repro/blob/master/report.pdf) for details.


# Repo Hygene

* use [git lfs](https://github.com/git-lfs/git-lfs/wiki/Installation) when keeping <50mb files within the repo (images, pdfs, small models)
* use [switch](https://drive.switch.ch/) or the EPFL gdrive for larger files
* don't commit junk files
* use latex/notes for math discussions, logbook in this readme to keep track of Todos/Dones

# General guidelines
* use [ray](https://ray.readthedocs.io/en/latest/rllib.html) as a starting block for distribution/working loop, [sacred](https://github.com/IDSIA/sacred) with the Filesystem or Tinydb observer to run experiments
* Jupyter Notebooks are okay, but don't commit any long analysis/hacking as final product. Exctract and move into the module (`deepmdp`)
* write experiments in `deepmdp/experiments`, put any scripts to launch them or do postprocessing in `scripts`
* Composition over inheritance. Use [attr.s](https://pypi.org/project/attrs/) to make data classes, this helps [mypy](http://mypy-lang.org/) (strong recommendation to use it with [pycharm-mypy](mcts_mri) to help you find bugs via [type hints](https://docs.python.org/3/library/typing.html)
* Use submodules to organize the code

# Setup
* Run `pip install -e .` to install deepmdp as a python package. 

# Logbook
## IK: 2020-02-25 post-discussion

- Tuesday 1630 regular meeting slot
- Adrian: till next week, set up benchmark environment with arbitrary model free RL algorithm.
- Igor: this log add references to distributional RL paper to pdfs, [garage](https://github.com/rlworkgroup/garage)  and [rllib](https://github.com/ray-project/ray/tree/master/rllib)

## IK: 2020-03-10 (previous meeting was in bus)

### References:
- [Depth scales](https://arxiv.org/abs/1611.01232)
- [ExtraAdam](https://arxiv.org/pdf/1802.10551.pdf)
- [On the regularization of Wasserstein GANs](https://arxiv.org/abs/1709.08894)
- [Which GAN methods actually converge](https://arxiv.org/pdf/1801.04406.pdf)

### Next steps

-  run VPGD/TRPO/DQN on a simple atari task
-  no mujoco experiments for now
-  then afterwards set up dummy deepmdp framework: add reward loss to abstraction network on top of PG, add dummy transition loss (RSME)
-  TODO igor: check how they do Wasserstein Distance
