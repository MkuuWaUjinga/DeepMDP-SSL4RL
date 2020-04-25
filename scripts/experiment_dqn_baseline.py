import dowel
import click
from deepmdp.experiments.dqn_baseline import ex
from sacred.observers import FileStorageObserver
from garage.experiment import deterministic
from dowel import logger

@click.command()
@click.option('--config_path', help='Path for experiment\'s config')
def start_exp(config_path):
    logger.add_output(dowel.StdOutput())
    deterministic.set_seed(21)
    ex.observers.append(FileStorageObserver("runs"))
    ex.add_config(config_path)
    ex.run()

if __name__=="__main__":
    start_exp()