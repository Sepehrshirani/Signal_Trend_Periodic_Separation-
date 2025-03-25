#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
from src.io.loaders import load_edf
from src.core.ssa import SSA
from src.visualization.plots import plot_components

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="EDF file path")
    parser.add_argument("--channel", type=int, default=5)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    # Load data
    signal, fs = load_edf(args.input, args.channel)
    time = np.arange(len(signal)) / fs

    # Process
    analyzer = SSA(signal)
    trend, periodic, noise = analyzer.decompose()

    # Save results
    Path(args.output).mkdir(exist_ok=True)
    np.savez(f"{args.output}/components.npz", 
             trend=trend, periodic=periodic, noise=noise)
    
    # Plot
    fig = plot_components(time, signal, trend, periodic, noise)
    fig.savefig(f"{args.output}/decomposition.png", dpi=300)

if __name__ == "__main__":
    main()
