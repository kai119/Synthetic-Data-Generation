# Synthetic-Data-Generation

This code accompanies the "Interpreting Neural Networks by Identifying Critical Activation Paths" research and is intended to show the experiments and analysis used in the research.

## Installing Required Packages

The libraries needed to run this code can be installed using pip. To install the necessary packages using the terminal, navigate to the root of this project and run the command

`pip install -r requirements.txt` 

## How to Run Experiments

Each experiment can by run by running the file that the experiment is in. To do this from the terminal, navigate to the root of this project and run 

`python <experiment file name>.py`

The runnable experiment files correspond to the experiments discussed in Section 8 of the research and have the following names:

- `base_method.py` - Section 8.1
- `iterations_experiment.py` - Section 8.2
- `all_incorrect_data.py` - Section 8.3
- `iterations_generating_new_data.py` - Section 8.4

When run, each of these experiments will create a csv file in the `results` directory, containing information in 100 runs of the experiment performed.

## Results Analysis

The analysis of the csv files created from running the experiments can be found in the `results_analysis` directory, each file is a Jupyter Notebook that imports the results from an experiment and shows various statistics for the results. Instructions on how to run Jupyter Notebooks are below:

1. install jupyter notebook: https://jupyter.org/install
2. navigate to the root of this project via the terminal and run the command `jupyter notebook`
3. a web session will start that will allow you to navigate to the results_analysis folder and select a notebook