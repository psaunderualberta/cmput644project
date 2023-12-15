import os
import pickle
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
from src.utility.constants import *
from src.heuristic.parsing import parse_heuristic
from dask import delayed, compute 
from dask.distributed import Client, LocalCluster
import scienceplots
plt.style.use(["science", "grid"])

@delayed
def delayed_load_data(file):
    df = pd.read_parquet(file)
    df.columns = list(
        map(
            lambda col: NORMALIZED_COLUMN_NAMES_MAPPING[col]
            if col in NORMALIZED_COLUMN_NAMES_MAPPING
            else col,
            df.columns,
        )
    )
    return df


@delayed(nout=2)
def delayed_execute_heuristic(df, heuristics, use_heuristic=True):
    if use_heuristic:
        new_cols = {str(h): h.execute(df) for h in heuristics}
        new_df = df.assign(**new_cols)
        x_cols = NORMALIZED_COLUMN_NAMES + list(new_cols.keys())
    else:
        new_df = df
        x_cols = NORMALIZED_COLUMN_NAMES
    return new_df[x_cols], new_df[CLASSES_2_Y_COLUMN]


@delayed(nout=2)
def delayed_trim_nans(X, y):
    X = X.replace([np.inf, -np.inf], np.nan)
    idxs = np.where(pd.isnull(X).any(axis=1))[0]

    X = X.drop(idxs)
    y = y.drop(idxs)
    assert X.shape[0] == y.shape[0]
    return X, y


def heatmaps():
    file_location = os.path.dirname(os.path.realpath(__file__))
    table_files = list(map(lambda f: os.path.join(f, "tables.pkl"), MAPELITES_RESULTS))

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


def fitness_vs_coefs_kde_1d():
    file_location = os.path.dirname(os.path.realpath(__file__))
    source = "Elitism"
    if source == "MAP-Elites":
        folders = MAPELITES_RESULTS
    elif source == "Elitism":
        folders = TRADITIONAL_RESULTS
    else:
        raise ValueError(f"Invalid source: {source}")

    table_files = list(map(lambda f: os.path.join(f, "models.pkl"), folders))

    model_data = []
    for table_file in table_files:
        with open(table_file, "rb") as f:
            model_data.append(pickle.load(f))
    
    # Get the fitnesses, coefs, and columns for each model
    fitnesses = list(map(lambda d: d["fitnesses"], model_data))
    coefs = list(map(lambda d: d["models"][-1].coef_[0], model_data))
    columns = list(map(lambda d: d["columns"][0], model_data))

    # Trim the columns to only include the synthesized features
    num_x_cols = len(X_COLUMNS)
    columns = list(map(lambda c: c[num_x_cols:], columns))
    original_coefs = list(map(lambda c: c[:num_x_cols], coefs))
    coefs = list(map(lambda c: c[num_x_cols:], coefs))

    assert all(map(lambda col, coef, f: len(col) == len(coef) == len(f), columns, coefs, fitnesses))

    fitnesses = np.array(fitnesses).flatten()
    original_coefs = np.array(original_coefs).flatten()
    coefs = np.array(coefs).flatten()

    # plot the data
    ax = sns.kdeplot(x=coefs, label="Synthesized")
    ax = sns.kdeplot(x=original_coefs, label="Original")
    ax.set(
        xlabel="Coefficient",
        ylabel="Density",
        title=f"{source} - Coefficient Density",
    )
    ax.legend(loc="upper right")

    plot_dir = os.path.join(file_location, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f"coef_density_{source.lower()}.pdf"))
    plt.clf()


def plot_run():
    file_location = os.path.dirname(os.path.realpath(__file__))
    run_path = os.path.join(file_location, "logs", "results", "mapelites-12hr-ga-1.txt")

    df = pd.read_csv(run_path, sep="|")
    df.columns = list(map(lambda col: col.strip(), df.columns))
    gb = df.groupby("Population Number")
    fitnesses = gb["Fitness"]
    best_fitnesses = pd.Series(fitnesses.max()).cummax().to_numpy()
    mean_fitnesses = fitnesses.mean().to_numpy()
    all_fitnesses = fitnesses.apply(lambda x: x.to_numpy()).reset_index()["Fitness"]

    _, ax = plt.subplots()
    # Plot best and all fitnesses.
    x = np.linspace(0, 12, len(mean_fitnesses))
    ax.plot(x, best_fitnesses, c="r", linewidth=3)
    x_all_fitnesses = np.repeat(x, all_fitnesses.apply(len))
    y_all_fitnesses = np.concatenate(all_fitnesses)
    ax.scatter(
        x_all_fitnesses,
        y_all_fitnesses,
        alpha=0.1,
        s=0.2,
    )

    ax.set(
        xlabel="Hours of Synthesis",
        ylabel="Fitness",
        title="MAP-Elites: Fitness vs. Generation",
    )
    plt.savefig(os.path.join(file_location, "plots", "mapelites_fitness.pdf"))
    plt.clf()


if __name__ == "__main__":
    # heatmaps()
    # fitness_vs_coefs_kde_1d()
    # fitness_vs_coefs_kde_2d()
    # plot_run()
    # mapelites_vs_traditional()
    pass