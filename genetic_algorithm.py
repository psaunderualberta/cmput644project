from src.utility.util import load_data
from src.utility.constants import *
import wandb


def main():
    print("Hello, World!")


if __name__ == "__main__":
    config = {
        "SEED": 42,
        "POPULATION_SIZE": 20,
        "TIMEOUT": 30,
        "WANDB": False,
        "WANDB_PROJECT": "cmput644project",
        "WANDB_ENTITY": "psaunder",
    }

    if config["WANDB"]:
        wandb.init(
            config=config,
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
        )

    main()
