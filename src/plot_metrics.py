import argparse
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


def load_data(file):
    with open(file, "r") as f:
        json_data = f.read()
        data = json.loads(json_data)
        return pd.DataFrame(data)


def save_plot(fig, filepath):
    fig.savefig(filepath)
    plt.close(fig)


def plot_column(df, column, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(df[column])
    ax.set_xlabel("Epoch")
    ax.set_ylabel(column)
    ax.set_title(f"Evolution of {column} throughout training")
    filepath = os.path.join(output_dir, f"{column}.png")
    save_plot(fig, filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to the CSV file")
    parser.add_argument(
        "-o", "--output", default=None, help="Path to the folder to put the plots in."
    )
    args = parser.parse_args()
    output_folder = (
        args.output if args.output else os.path.abspath(os.path.join(args.file, os.pardir))
    )

    df = load_data(args.file)
    for column in df.columns:
        plot_column(df, column, output_folder)


if __name__ == "__main__":
    main()
