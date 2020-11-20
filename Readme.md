# DeepMDP Replication Study
Representation learning for RL promises to alleviate two problems 
agents face in real-world environments:
 - Learning good policies from high dimensional and noisy input.
 - Ensure robustness and generalization outside test environment (often simulated).
 
DeepMDP is one of the first algorithms in this area that brings theoretical guarantees to the
deep learning world. In line with that it also shows good empirical results in comparison to model-free RL. It can be combined with any model-free policy learning strategy. In the paper the author's used C51. We are going to use a DQN for simplicity. 

Since the code was not published we provide an implementation of the algorithm with a DQN as base algorithm.
Along with the code we also ran experiments to check the author's claims. Have a look at the 
[report](https://github.com/MkuuWaUjinga/deepmdp-repro/blob/master/report.pdf) to learn more about our results.

### Setup
* Clone the repo.
* Run `pip install -e .` to install deepmdp as a python package. 

### Running an Experiment
All experiments are logged and executed using sacred. The configs for the experiments are located in `scripts/configs`.
If you want to use the auxiliary DeepMDP losses simply set the flag in the config to `true`. 
The specifed DQN architecture is then split into an encoder part that is shared between the q-head, transition-head and reward head and the q-head itself.
When you are done simply run `python scripts/experiment_dqn_baseline.py --config_path [path]`.

Furthermore, we use Visdom to log experiment data. Make sure to have your Visdom server running on port 9098. For that run `visdom -port 9098`.
Then you can access it any time via [http://localhost:9098]().

![Alt text](visdom-screenshot.png?raw=true "Your Visdom Server")
