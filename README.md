# F23 CMPUT 644 Project - Paul Saunders

## Creating environment
To create an environment capable of running all scripts in this repository, use your virtual environment manager of choice to create an environment with Python 3.10.9. Other versions of python will likely work, but have not been tested. Once this is complete, run `pip install -r ./requirements.txt` to install the appropriate libraries. The total size of the virtual environment is approximately 20-25MB. 

## Producing Dataset
If you only wish to use a small part of the data, then this is already included in the `./shortened_data` directory. To create a compatible dataset from the entire original dataset, one must just download the CICIoT2023 dataset from [this](https://www.unb.ca/cic/datasets/iotdataset-2023.html) link and place it into the directory `CICIoT2023/data/original`. Once this is complete, run `python combine_dataset.py`. This script performs a few manipulations to the data, such as changing column names and converting the original csv files into `parquet` format. This enables faster loading of the data, among other benefits. 

## Genetic Algorithm
The genetic algorithm is found in `genetic_algorithm.py`. To change the parameters of the genetci algorithm, one must change the `config` dictionary near the top of the `main` function. For testing and quick evaluation, I would recommend the following settings: 
```python
    config = {
        # Random Seed, can be anything
        "SEED": 1337,

        # Point to the short data for quick loading
        "POPULATION_SRC": SHORTENED_DATA_FILES,

        # Can be anything > 0, experiments used a setting of 20
        "POPULATION_SIZE": 20,

        # Timeout for the genetic algorithm. 10 minutes, expressed in seconds
        "TIMEOUT": 10 * 60,

        # Cloud-based experiment logging, make sure this is false
        "WANDB": False,  

        # These can be anything if WANDB is set to false
        "WANDB_PROJECT": "cmput644project",
        "WANDB_ENTITY": "psaunder",  

        # How the population should be stored
        # - 'mapelites' -> use mapelites,
        # - 'traditional' -> use elitism 
        "POPULATION_TYPE": "mapelites",

        # Log file to write the results of synthesis as they are produced
        "LOG_FILE": os.path.join("logs", "results", f"{today}-log.txt"),
    }    
```

Once the above config is set appropriately, simply run `python genetic_algorithm.py` from the root of the repository. Note that it will take a little bit of time for output to start appearing, as the `dask` client requires a little bit of time to load.

At the start of the output, you will see a local link appear (i.e. http://127.0.0.1:8787/). This site represents the status of the parallel workers operating on your machine. It is primarily a diagnostic tool to analyse worker status, but the main page is somewhat interesting to look at.

## Training Models

