import dowel
from deepmdp.experiments.dqn_baseline import ex
from sacred.observers import FileStorageObserver
from garage.experiment import deterministic
from dowel import logger

logger.add_output(dowel.StdOutput())
deterministic.set_seed(1)
ex.observers.append(FileStorageObserver("runs"))
ex.add_config({
    "env_name": "LunarLander-v2",
    "dqn_config": {
        "learning_rate": 0.001,
        "replay_buffer_size": 500000,
        "buffer_batch_size": 64
    }
})
ex.run()