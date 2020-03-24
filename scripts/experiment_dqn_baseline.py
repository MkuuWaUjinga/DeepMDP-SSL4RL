import dowel
from deepmdp.experiments.dqn_baseline import ex
from sacred.observers import FileStorageObserver
from garage.experiment import deterministic
from dowel import logger

logger.add_output(dowel.StdOutput())
deterministic.set_seed(1)
ex.observers.append(FileStorageObserver("runs"))
ex.run()