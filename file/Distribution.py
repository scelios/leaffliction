#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import sys
import argparse
import matplotlib
matplotlib.use('TkAgg')  # select a GUI backend BEFORE importing pyplot


def create_charts(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")

    subdirs = [d for d in os.listdir(
        directory) if os.path.isdir(os.path.join(directory, d))]
    counts = {}

    for subdir in subdirs:
        subdir_path = os.path.join(directory, subdir)
        counts[subdir] = len([f for f in os.listdir(
            subdir_path) if os.path.isfile(os.path.join(subdir_path, f))])

    labels = list(counts.keys())
    sizes = list(counts.values())

    # Pie Chart
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Subdirectory Types - Pie Chart')

    # Bar Chart
    plt.subplot(1, 2, 2)
    plt.bar(labels, sizes, color='skyblue')
    plt.xlabel('Subdirectory Types')
    plt.ylabel('Number of Files')
    plt.title('Distribution of Subdirectory Types - Bar Chart')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
    # plt.savefig("distribution.png", bbox_inches="tight")


def main():
    try:
        parser = argparse.ArgumentParser(
            description="Create pie charts and bar charts "
            "for subdirectory types in a given directory.")
        parser.add_argument(
            "path", type=str, help="Path to the Images directory")
        args = parser.parse_args()
        create_charts(args.path)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
