from network import Network
from method import *
from base_method import BaseMethod


class AllIncorrectData(BaseMethod):
    """Notes on this process

    - Takes a VERY long time to do all 100 test runs
    - Reduces the data down to 1/3 of its original size - could require multiple generations of random data to be
      fully effective"""

    def __init__(self, synth_data_filename, target_data_filename, h1_size, h2_size):
        super().__init__(synth_data_filename, target_data_filename, h1_size, h2_size)

    def experiment(self):
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


def find_similar_paths(clf, target_incorrect_path, importance):
    """Find the how similar each piece of synthetic data's path is to an incorrectly classified target data's
    path.

    Parameters
    ----------
    target_incorrect_path: list
        a critical path of each piece of incorrectly classified target data
    importance: list
        the importances of the neurons in target_incorrect_paths

    Returns
    -------
    list
        the distance of each piece of synthetic data from the specified incorrectly classified target data,
        along with the index number of the synthetic data
    """

    dist, ind = clf.kneighbors(np.array(target_incorrect_path).reshape(1, -1))

    weighted_distances = []
    for i in dist:
        weighted_distances.append(i * importance)
    zipped = list(zip(weighted_distances[0], ind[0]))
    return sorted(zipped, key=lambda t: t[1])


def find_synth_data_to_remove(synth_paths, target_paths, all_importances):
    """Find all pieces of synthetic data that have a total 'distance to closest incorrect path' that is either
    the smallest or second smallest value.

    Multiple pieces of synthetic data will share the same distance values, so there will most likely be more
    than one piece of data that has the smallest distance value (same with second smallest).

    Parameters
    ----------
    synth_paths: list
        critical paths for every piece of synthetic data
    target_paths: list
        critical paths for every piece of incorrectly classified target data
    all_importances: list
        path importances for every piece of incorrectly classified target data

    Returns
    -------
    list
        the index of each piece of synthetic data to remove
    """

    items_to_remove = set()

    first_paths = [i[0] for i in synth_paths]
    first_paths_clf = NearestNeighbors(n_neighbors=len(first_paths))
    first_paths_clf.fit(first_paths)

    second_paths = [i[1] for i in synth_paths]
    second_paths_clf = NearestNeighbors(n_neighbors=len(second_paths))
    second_paths_clf.fit(second_paths)

    third_paths = [i[2] for i in synth_paths]
    third_paths_clf = NearestNeighbors(n_neighbors=len(third_paths))
    third_paths_clf.fit(third_paths)

    for path_num in range(len(target_paths)):
        first_paths_similar_paths = find_similar_paths(first_paths_clf, target_paths[path_num][0][0], all_importances[path_num][0])

        second_paths_similar_paths = find_similar_paths(second_paths_clf, target_paths[path_num][0][1], all_importances[path_num][1])

        third_paths_similar_paths = find_similar_paths(third_paths_clf, target_paths[path_num][0][2], all_importances[path_num][2])

        # Find the total "distance to closest incorrect path" for each piece of synthetic data
        total_distances = []
        for i in range(len(first_paths_similar_paths)):
            total_distance = first_paths_similar_paths[i][0] + \
                             second_paths_similar_paths[i][0] + \
                             third_paths_similar_paths[i][0]
            total_distances.append((total_distance, first_paths_similar_paths[i][1]))
        total_distances = sorted(total_distances, key=lambda t: t[0])

        # Find all synthetic pieces of data with a distance of 0 for piece of incorrectly classified data
        for i in total_distances:
            if i[0] == 0:
                items_to_remove.add(i[1])
    return items_to_remove


if __name__ == '__main__':
    executor = AllIncorrectData('Gravity Data.xlsx', 'Gravity Data.xlsx', 10, 5)
    executor.run("Synthetic Data Generation - Experiment 2 - All Incorrect Data", 'Experiment 2 - All Incorrect Data.csv')
