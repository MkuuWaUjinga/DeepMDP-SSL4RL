from deepmdp.experiments.example import ex
from sacred.observers import FileStorageObserver
ex.observers.append(FileStorageObserver("runs"))
# to run programmatically
for i in range(5):
    ex.run(config_updates={
        "foo":i
    })
# OR to directly run from the command line
#import sys
#ex.run_commandline(sys.argv)
