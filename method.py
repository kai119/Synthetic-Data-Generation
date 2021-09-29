import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def find_max_values_in_np_array(data, n):
    """Find the n highest values in a numpy array.
    If n is more than the size of the data, n will be set to the size of the data.

    Parameters
    ----------
    data: np.array
        to find the highest values for
    n: int
        the number of values in the list to find

    Returns
    -------
    np.array
        containing the n highest values in the given list
    """

    if n > len(data):
        n = len(data)
    partition = np.partition(-data, n)
    return -partition[:n]


def find_max_value_locations_of_np_array(data, n):
    """Find the location of the n highest values in a numpy array.
    If n is more than the size of the data, n will be set to the size of the data.

    Parameters
    ----------
    data: np.array
        to find the highest values for
    n: int
        the number of values in the list to find

    Returns
    -------
    np.array
        containing the location of the n highest values in the given list
    """

    if n > len(data):
        n = len(data)
    partition = np.argpartition(-data, n)
    return partition[:n]


def find_n_highest(data, n):
    """Find the location of the n highest values in a numpy array and return their locations

    Parameters
    ----------
    data: np.array
        to find the highest values for
    n: int
        the number of values in the list to find

    Returns
    -------
    np.array
        an array where all values are zero apart from the indexes of the n highest values in the given array,
        which are represented as their order in the n highest values (for example, array [8, 2, 9, 3, 7] would
        return [2, 0, 1, 0, 3])
    np.array
        an array showing the location of the n highest values in the given array
    """

    critical_path = find_max_value_locations_of_np_array(data, n)

    data_path = np.zeros(len(data)).astype(int)

    for path_num in range(len(critical_path)):
        data_path[critical_path[path_num]] = path_num + 1

    return data_path, critical_path


def get_data_info(network, n_highest, data):
    """Get information on every piece of data that has been supplied to a network.

    Information includes input values ('input'), positions of n highest neurons for each hidden
    layer ('h1', 'h2', ...), output values ('output'), and the locations of the n highest activated
    neurons for each layer ('critical_path')

    Parameters
    ----------
    network: torch.nn.Module
        the network to get information for
    n_highest
        the number of heghest activations to locate
    data
        the data to get the network information for

    Returns
    -------
    list[dict]
        information on the activations of each piece of data supplied to the network
    """

    network(data)
    features = network.get_activations()
    all_data_info = []
    for i in range(len(data)):
        data_info = {'input': data[i]}

        critical_path = []
        for layer_num in range(len(features)):
            if layer_num != len(features) - 1:
                layer_name = 'h{}'.format(layer_num)
                info_for_layer, path = find_n_highest(features[layer_num][i], n_highest)
                data_info[layer_name] = info_for_layer
                critical_path.append(path)
            else:
                data_info['output'] = features[layer_num][i]
        data_info['critical_path'] = critical_path
        all_data_info.append(data_info)
    return all_data_info


def get_data_info_target(network, n_highest, data):
    """Get information on every piece of data that has been supplied to a network.

    Information includes input values ('input'), positions of n highest neurons for each hidden
    layer ('h1', 'h2', ...), output values ('output'), and the locations of the n highest activated
    neurons for each layer ('critical_path')

    Parameters
    ----------
    network: torch.nn.Module
        the network to get information for
    n_highest
        the number of n_highest activations to locate
    data
        the data to get the network information for

    Returns
    -------
    list[dict]
        information on the activations of each piece of data supplied to the network
    """

    network(data)
    features = network.get_activations()
    all_data_info_target = []
    for i in range(len(data)):
        data_info = {'input': data[i]}

        critical_path = []
        for layer_num in range(len(features)):
            if layer_num != len(features) - 1:
                layer_name = 'h{}'.format(layer_num)
                info_for_layer, path = find_n_highest(features[layer_num][i], n_highest)
                data_info[layer_name] = features[layer_num][i]
                critical_path.append(path)
            else:
                data_info['output'] = features[layer_num][i]
        data_info['critical_path'] = critical_path
        all_data_info_target.append(data_info)
    return all_data_info_target


def get_critical_paths_in_order(critical_paths):
    """Get critical paths in correct path form.

    gives a critical path list ([[2, 6, 1], [10, 3, 6]]) in the 1st, 2nd, and 3rd paths form
    ([[2, 10], [6, 3], [1, 6]])

    Parameters
    ----------
    critical_paths: list
        the critical paths of neuron activations for a data set in the network

    Returns
    -------
    list
        a list containing the 1st, 2nd,and 3rd paths forms for each piece of data
    """

    paths = []
    for path in critical_paths:
        first_n = []
        second_n = []
        third_n = []
        for layer in path:
            first_n.append(layer[0])
            second_n.append(layer[1])
            third_n.append(layer[2])
        paths.append([first_n, second_n, third_n])
    return paths


def find_incorrect_paths(network, X_target, y_target, target_data_info):
    """Find the critical paths of all incorrectly classified pieces of target data.

    Parameters
    ----------
    network: torch.nn.Module
        the network to find incorrect predictions for
    X_target: torch.FloatTensor
        the input variables for the target data
    y_target: torch.LongTensor
        the output variable for the target data
    target_data_info: list[dict]
        information on the network's activations for each piece of target data

    Returns
    -------
    list[Tuple[list, int]]
        all critical paths of incorrectly predicted pieces of target data, along with the index number
        of the data
    """

    incorrect_paths = []
    predictions = network(X_target)
    max_value, prediction = torch.max(predictions, 1)
    for i in range(len(predictions)):
        if prediction[i] != y_target[i]:
            incorrect_paths.append((target_data_info[i]['critical_path'], i))

    paths = get_critical_paths_in_order(path[0] for path in incorrect_paths)
    return [(paths[i], incorrect_paths[i][1]) for i in range(len(paths))]


def get_paths_importance(paths, data_num, data_info):
    """Gets how 'important' each critical path is in the network's decision.

    Parameters
    ----------
    paths: Tuple[list, int]
        the ordered critical paths for a piece of data, and the index of the piece of data
    data_num: int
        the index of the piece of data that the path relates to
    data_info: list[dict]
        information on the activations of all pieces of data in a network

    Returns
    -------
    list
        the importences of each critical path for a piece of data
    """

    importance = []
    for path in paths[0]:
        total_for_path = 0
        for i in range(len(path)):
            layer = f'h{i}'
            neuron_of_importance = data_info[paths[1]][layer][path[i]]
            total = 0
            for x in data_info[data_num][layer]:
                total += np.abs(x)

            # gets the amount of contribution that the neurons in a path have for the layers that they are part of
            total_for_path += neuron_of_importance / total
        importance.append(total_for_path)
    # return the total contribution that each path has on the network (how important each path is the the
    # decision that the network makes)
    return importance


def get_unique_importances(paths, data_info):
    """Finds all path importances using the get_paths_importance function, then finds all unique variations of
    those importances

    Parameters
    ----------
    paths: list
        the ordered critical paths
    data_info: list[dict]
        information on the network's activations for each piece of data

    Returns
    -------
    list
        all unique importances
    list
        all importances
    """

    all_importances = []
    for i in range(len(paths)):
        all_importances.append(get_paths_importance(paths[i], i, data_info))

    unique_importances = []
    for importance in all_importances:
        if importance not in unique_importances:
            unique_importances.append(importance)

    return unique_importances, all_importances


def find_similar_paths(synth_paths, target_incorrect_path, importance):
    """Find the how similar each piece of synthetic data's path is to an incorrectly classified target data's
    path.

    Parameters
    ----------
    synth_paths: list
        a critical path of each piece of synthetic data
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

    clf = NearestNeighbors(n_neighbors=len(synth_paths))
    clf.fit(synth_paths)

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

    first_paths = [i[0] for i in synth_paths]
    first_paths_similar_paths = find_similar_paths(first_paths, target_paths[0][0][0], all_importances[0][0])

    second_paths = [i[1] for i in synth_paths]
    second_paths_similar_paths = find_similar_paths(second_paths, target_paths[0][0][1], all_importances[0][1])

    third_paths = [i[2] for i in synth_paths]
    third_paths_similar_paths = find_similar_paths(third_paths, target_paths[0][0][2], all_importances[0][2])

    # Find the total "distance to closest incorrect path" for each piece of synthetic data
    total_distances = []
    for i in range(len(first_paths_similar_paths)):
        total_distance = first_paths_similar_paths[i][0] + \
                         second_paths_similar_paths[i][0] + \
                         third_paths_similar_paths[i][0]
        total_distances.append((total_distance, first_paths_similar_paths[i][1]))
    total_distances = sorted(total_distances, key=lambda t: t[0])

    # Find all synthetic pieces of data with the lowest and second lowest "distance to closest incorrect path"
    same_dist = []
    lowest_dist = total_distances[0][0]
    second_lowest = 0
    for i in total_distances:
        if i[0] > lowest_dist:
            second_lowest = i[0]
            break
    for i in total_distances:
        if i[0] == lowest_dist or i[0] == second_lowest:
            same_dist.append(i[1])
    return same_dist


def remove_synthetic_data(df, items_to_remove):
    """Removes all pieces of data with row indexes defined in items_to_remove

    Parameters
    ----------
    df: pd.Dataframe
        the dataframe to remove items from
    items_to_remove: list
        the row indexes of the items to remove from the dataframe

    Returns
    -------
    pd.Dataframe
        the dataframe with the items at the specified row indexes removed
    """

    return df.drop(items_to_remove).reset_index().drop('index', axis=1)
