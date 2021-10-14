from base_method import BaseMethod
from network import Network
from data_generator import DataGenerator
from method import *
from data_transformer import transform_gravity_data
from all_incorrect_data import find_synth_data_to_remove
import pandas as pd


class IterationGenerateNewData(BaseMethod):

    def __init__(self, synth_data_filename, target_data_filename, h1_size, h2_size, data_generations):
        super().__init__(synth_data_filename, target_data_filename, h1_size, h2_size)
        self.data_generations = data_generations

    def perform_iteration(self):
        self.network = Network(5, self.h1_size, self.h2_size, 2)
        self.network.train_network(self.X_synth, self.y_synth)
        synth_data_info = get_data_info(self.network, 3, self.X_synth)
        print('---------------------------------Evaluation---------------------------------')
        self.network.evaluate_model(self.X_target, self.y_target)
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
        self.network.evaluate_model(self.X_target, self.y_target)
        print('----------------------------------------------------------------------------')
        print()

    def experiment(self):
        final_df = pd.DataFrame(columns=['Height (m)', 'Mass (KG)', 'FRAGILE', 'MEDIUM', 'HARD', 'OUTCOME'])
        self.network = Network(5, self.h1_size, self.h2_size, 2)
        self.network.train_network(self.X_synth, self.y_synth)
        before_accuracy = self.network.evaluate_model(self.X_target, self.y_target)
        for i in range(self.data_generations):
            print(f'----------------------------------------Iteration {i+1}-------------------------------------')
            self.perform_iteration()
            final_df = final_df.append(self.synth_df, ignore_index=True)
            self.synth_df = DataGenerator.generate_gravity_data(2000)
            self.synth_df = transform_gravity_data(self.synth_df, 'Gravity Data Synthetic Transformed.xlsx')
            self.X_synth = torch.FloatTensor(self.synth_df.drop(['OUTCOME'], axis=1).values)
            self.y_synth = torch.LongTensor(self.synth_df['OUTCOME'].values)

        self.X_synth = torch.FloatTensor(final_df.drop(['OUTCOME'], axis=1).to_numpy(dtype='float32'))
        self.y_synth = torch.LongTensor(final_df['OUTCOME'].to_numpy(dtype='int64'))
        self.network = Network(5, self.h1_size, self.h2_size, 2)
        self.network.train_network(self.X_synth, self.y_synth)
        print(f'------------------------------------------Results---------------------------------------')
        print(f'final df size: {final_df.size}')
        after_accuracy = self.network.evaluate_model(self.X_target, self.y_target)

        return before_accuracy, after_accuracy


if __name__ == '__main__':
    executor = IterationGenerateNewData('Gravity Data.xlsx', 'Gravity Data.xlsx', 10, 5, 20)
    executor.run("Synthetic Data Generation - Experiment 3 - Generating New Data", 'Experiment 3 - Generating New Data.csv')
