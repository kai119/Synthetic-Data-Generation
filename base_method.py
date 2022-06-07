from method import *
from network import Network
from data_transformer import transform_gravity_data
import pandas as pd
import torch
import pyfiglet
import os
from timeit import default_timer as timer


class BaseMethod():

    def __init__(self, synth_data_filename, target_data_filename, h1_size, h2_size):
        self.synth_data_filename = synth_data_filename
        self.target_data_filename = target_data_filename

        self.synth_df = pd.DataFrame()
        self.X_synth = torch.FloatTensor()
        self.y_synth = torch.LongTensor()
        self.target_df = pd.DataFrame()
        self.X_target = torch.FloatTensor()
        self.y_target = torch.LongTensor()

        self.network = Network(5, h1_size, h2_size, 2)
        self.h1_size = h1_size
        self.h2_size = h2_size

    def initialize_data(self):
        """Create the synthetic and target data sets, transform the data, and split the data into X and y variables."""

        self.synth_df = pd.read_excel('data/' + self.synth_data_filename, 'Synthetic Data', skiprows=5)
        self.synth_df.drop(['Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9'], inplace=True, axis=1)
        self.synth_df = transform_gravity_data(self.synth_df, 'Gravity Data Synthetic Transformed.xlsx')
        self.X_synth = torch.FloatTensor(self.synth_df.drop(['OUTCOME'], axis=1).values)
        self.y_synth = torch.LongTensor(self.synth_df['OUTCOME'].values)

        self.target_df = pd.read_excel('data/' + self.target_data_filename, 'Target Data', skiprows=5)
        self.target_df.drop(['Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9'], inplace=True, axis=1)
        self.target_df = transform_gravity_data(self.target_df, 'Gravity Data Transformed.xlsx')
        self.X_target = torch.FloatTensor(self.target_df.drop(['OUTCOME'], axis=1).values)
        self.y_target = torch.LongTensor(self.target_df['OUTCOME'].values)

    def experiment(self):
        """Perform the main experiment."""

        self.network = Network(5, self.h1_size, self.h2_size, 2)
        self.network.train_network(self.X_synth, self.y_synth)
        synth_data_info = get_data_info(self.network, 3, self.X_synth)
        print('---------------------------------Evaluation---------------------------------')
        before_accuracy = self.network.evaluate_model(self.X_target, self.y_target)
        print('----------------------------------------------------------------------------')
        print()
        print('------------------------------Method Execution------------------------------')
        target_data_info = get_data_info_target(self.network, 3, self.X_target)
        target_paths = find_incorrect_paths(self.network, self.X_target, self.y_target, target_data_info)
        unique_importances, all_importances = get_unique_importances(target_paths, target_data_info)

        # Get critical paths for synth data
        c_paths = []
        for i in range(len(synth_data_info)):
            c_paths.append(synth_data_info[i]['critical_path'])
        synth_paths = get_critical_paths_in_order(c_paths)

        items_to_remove = find_synth_data_to_remove(synth_paths, target_paths, all_importances)
        print(f'Removing {len(items_to_remove)} items from synthetic data')
        self.synth_df = remove_synthetic_data(self.synth_df, items_to_remove)
        print(f'Size of synthetic data after removal: {len(self.synth_df)}')
        print('----------------------------------------------------------------------------')
        print()

        self.X_synth = torch.FloatTensor(self.synth_df.drop(['OUTCOME'], axis=1).values)
        self.y_synth = torch.LongTensor(self.synth_df['OUTCOME'].values)
        self.network = Network(5, self.h1_size, self.h2_size, 2)
        self.network.train_network(self.X_synth, self.y_synth)
        print('---------------------------------Evaluation---------------------------------')
        after_accuracy = self.network.evaluate_model(self.X_target, self.y_target)
        print('----------------------------------------------------------------------------')
        print()

        return before_accuracy, after_accuracy

    def run(self, banner_text, results_file):
        """Run the experiment 100 times and export the results to a csv."""

        ascii_banner = pyfiglet.figlet_format(banner_text)
        print(ascii_banner)
        print()
        completed_runs = 0
        columns = ['Before Accuracy', 'After Accuracy', 'Accuracy Change']
        if os.path.isfile('results/' + results_file):
            run_results = pd.read_csv('results/' + results_file)
            run_results = run_results[columns]
            completed_runs = run_results.shape[0]
            print(f'Resuming experiment from run {completed_runs + 1}')
        else:
            run_results = pd.DataFrame(columns=columns)
        for i in range(100):
            self.initialize_data()
            print(f'-------------------------------------------Run No {i+completed_runs+1}-------------------------------------------')
            start_time = timer()
            before_accuracy, after_accuracy = self.experiment()
            end_time = timer()
            results = pd.Series([before_accuracy, after_accuracy, after_accuracy - before_accuracy], index=columns)
            run_results = run_results.append(results, ignore_index=True)
            run_results.to_csv('results/' + results_file)


if __name__ == '__main__':
    executor = BaseMethod('Gravity Data.xlsx', 'Gravity Data.xlsx', 10, 5)

    executor.run("Synthetic Data Generation - Base Method", 'Base Method.csv')
