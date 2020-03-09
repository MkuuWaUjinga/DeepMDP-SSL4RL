# Doesn't work
#from deepmdp.experiments.reinforce import ex
#from sacred.observers import FileStorageObserver
#ex.observers.append(FileStorageObserver("runs"))
#ex.run()

# Works (Still tensor shape error though)
from deepmdp.experiments.reinforce import run_task
from garage.experiment import LocalRunner, SnapshotConfig, run_experiment
run_experiment(run_task, snapshot_mode='last', seed=1)

