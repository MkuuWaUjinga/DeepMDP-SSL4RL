from deepmdp.experiments.reinforce import ex
from sacred.observers import FileStorageObserver
ex.observers.append(FileStorageObserver("runs"))
ex.run()
