import time

import sacred

ex=sacred.experiment.Experiment("Example")
@ex.config
def config():
    foo=1
    bar=2
@ex.main
def run(foo,bar):
    time.sleep(foo)
    return foo*bar
