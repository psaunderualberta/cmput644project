import os
import pickle
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.utility.constants import *
from src.heuristic.parsing import parse_heuristic
from dask import delayed, compute 
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


def fitness_vs_coefs_kde_1d():
    file_location = os.path.dirname(os.path.realpath(__file__))
    source = "Elitism"
    if source == "MAP-Elites":
        folders = MAPELITES_RESULTS
    else:
        folders = TRADITIONAL_RESULTS

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


def feature_explain():
    # \sqrt{\min\left(5.6^{2}, \sqrt{\text{rst count} - 3.0}\right) - 5.1} in our heuristic grammar
    feature_str = "(sqrt (min (sqr 5.6) (sqrt (- rst_count 3.0)) -5.1))"
    feature = parse_heuristic(feature_str)

    # Load the data, execute the heuristic, and trim the nans
    dfs = [delayed_load_data(file) for file in COMBINED_DATA_FILES]
    Xs = [None for _ in dfs]
    ys = [None for _ in dfs]

    # Execute heuristics
    for i, df in enumerate(dfs):
        Xs[i], ys[i] = delayed_execute_heuristic(
            df, [feature], use_heuristic=True
        )

        Xs[i], ys[i] = delayed_trim_nans(Xs[i], ys[i])

    X_raw = [df["rst_count"] for df in dfs]
    X = [df[feature_str] for df in Xs]
    y = [df[CLASSES_2_Y_COLUMN] for df in ys]

    X_raw, X, y = compute(X_raw, X, y)
    X_raw = np.concatenate(X_raw)
    X = np.concatenate(X)
    y = np.where(np.concatenate(y) == BENIGN_CLASS, "Benign", "Malicious")

    # Basic statistics for X_raw and X
    print("X_raw stats:")
    print("min: {}".format(np.min(X_raw)))
    print("max: {}".format(np.max(X_raw)))
    print("mean: {}".format(np.mean(X_raw)))
    print("std: {}".format(np.std(X_raw)))
    print("X stats:")
    print("min: {}".format(np.min(X)))
    print("max: {}".format(np.max(X)))
    print("mean: {}".format(np.mean(X)))
    print("std: {}".format(np.std(X)))

    # Plot X vs. X_raw
    ax = sns.scatterplot(x=X_raw, y=X, hue=y)
    ax.set(
        xlabel="rst_count",
        ylabel="Synthesized Feature",
        title="Synthesized Feature vs. rst_count",
    )

    plot_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "feature_vs_rst_count.pdf"))

    # Plot the distributions of X vs. y and X_raw vs. y
    fig, axs = plt.subplots(2, 1, sharex=True)
    sns.histplot(x=X_raw, hue=y, ax=axs[0])
    sns.histplot(x=X, hue=y, ax=axs[1])
    axs[0].set(
        ylabel="Count",
        title="rst_count vs. Target",
    )
    axs[1].set(
        xlabel="Synthesized Feature",
        ylabel="Count",
        title="Synthesized Feature vs. Target",
    )

    plt.savefig(os.path.join(plot_dir, "feature_vs_rst_count_distributions.pdf"))
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
    fitness_vs_coefs_kde_1d()
    # fitness_vs_coefs_kde_2d()
    # feature_explain()
    # plot_run()
    # mapelites_vs_traditional()