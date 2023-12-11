import os
import pickle
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    file_location = os.path.dirname(os.path.realpath(__file__))
    table_files = glob(
        os.path.join(
            file_location,
            "artifacts",
            "**",
            "tables.pkl",
        )
    )

    tables = []
    dirnames = []
    for table_file in table_files:
        # Get the name of the folder containing the table
        dirname = Path(table_file).parents[0].name
        dirnames.append(dirname)
        with open(table_file, "rb") as f:
            tables.append(pickle.load(f).tables)

    plot_dir = os.path.join(file_location, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for table_array, dirname in zip(tables, dirnames):
        for table in table_array:
            df = table.get_heatmap_data()
            ax = sns.heatmap(
                df,
                annot=True,
                fmt=".2f",
                cmap="viridis",
            )

            title = "MAP-Elites Fitness | Resolution: {} | Run: {}".format(
                table.resolution, dirname
            )

            ax.set(
                xlabel="Heuristic Depth",
                ylabel="Heuristic Size",
                title=title,
            )

            fname = "map_elites_fitness_resolution_{}_run_{}".format(
                table.resolution, dirname
            )
            plt.savefig(
                os.path.join(plot_dir, "{}.pdf".format(fname.replace(" ", "_").lower()))
            )
            plt.clf()


if __name__ == "__main__":
    main()
