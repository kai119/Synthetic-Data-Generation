from network import Network
from method import *
from base_method import BaseMethod


class IterationsExperiment(BaseMethod):

    def __init__(self, synth_data_filename, target_data_filename, h1_size, h2_size, num_iterations):
        super().__init__(synth_data_filename, target_data_filename, h1_size, h2_size)
        self.num_iterations = num_iterations

    def experiment(self):
        before_accuracy = 0
        after_accuracy = 0
        for iteration in range(self.num_iterations):
            print(f'----------------------------------------Iteration {iteration+1}-------------------------------------')
            print()
            self.network = Network(5, self.h1_size, self.h2_size, 2)
            print('----------------------------------Training----------------------------------')
            self.network.train_network(self.X_synth, self.y_synth)
            print('----------------------------------------------------------------------------')
            print()
            synth_data_info = get_data_info(self.network, 3, self.X_synth)
            print('---------------------------------Evaluation---------------------------------')
            accuracy = self.network.evaluate_model(self.X_target, self.y_target)
            if iteration == 0:
                before_accuracy = accuracy
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
            if iteration == self.num_iterations - 1:
                print('---------------------------------ReTraining---------------------------------')
                self.network = Network(5, self.h1_size, self.h2_size, 2)
                self.network.train_network(self.X_synth, self.y_synth)
                print('----------------------------------------------------------------------------')
                print()
                print('---------------------------------Evaluation---------------------------------')
                after_accuracy = self.network.evaluate_model(self.X_target, self.y_target)
                print('----------------------------------------------------------------------------')
                print()
            else:
                print(
                    f'--------------------------------------------------------------------------------------------------')
                print()
        return before_accuracy, after_accuracy


if __name__ == '__main__':
    executor = IterationsExperiment('Gravity Data.xlsx', 'Gravity Data.xlsx', 10, 5, 20)
    executor.run("Synthetic Data Generation - Experiment 1 - Iterations", 'Experiment 1 - Iterations.csv')
