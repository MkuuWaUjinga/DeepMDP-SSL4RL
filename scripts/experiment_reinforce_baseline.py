# Doesn't work
from deepmdp.experiments.reinforce import ex
from sacred.observers import FileStorageObserver
from garage.experiment import deterministic

deterministic.set_seed(1)
ex.observers.append(FileStorageObserver("runs"))
ex.run()