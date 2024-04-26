import logging
import pickle

import numpy as np

root_dir = "C:\\Users\\dein_el\\PycharmProjects\\rlor_vrp\\data\\vrp_tw\\vrp_with_tw\\"


def load(filename, root_dir=root_dir):
    return pickle.load(open(root_dir + filename, "rb"))


file_catalog = {
    "test": {
        20: "vrp_with_tw20_test_seed1234.pkl",
        50: "vrp_with_tw50_test_seed1234.pkl",
        100: "vrp_with_tw100_test_seed1234.pkl",
        500: "vrp_with_tw500_test_seed1234.pkl",
    },
    "eval": {
        20: "vrp_with_tw20_validation_seed4321.pkl",
        50: "vrp_with_tw50_validation_seed4321.pkl",
        100: "vrp_with_tw100_validation_seed4321.pkl",
        500: "vrp_with_tw500_validation_seed4321.pkl",
    },
}


def make_instance(args):
    depot, loc, demand, tw, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size, scale = args
    return {
        "loc": np.array(loc) / grid_size,
        "demand": np.array(demand) / capacity,
        "depot": np.array(depot) / grid_size,
        "tw": np.array(tw) #/ scale,

    }


class lazyClass:
    data = {
        "test": {},
        "eval": {},
    }

    def __getitem__(self, index):
        partition, nodes, idx, tw = index
        if not (partition in self.data) or not (nodes in self.data[partition]):
            logging.warning(
                f"Data sepecified by ({partition}, {nodes}) was not initialized. Attepmting to load it for the first time."
            )
            data = load(file_catalog[partition][nodes])
            self.data[partition][nodes] = [make_instance(instance) for instance in data]

        return self.data[partition][nodes][idx]


VRPDatasetTW = lazyClass()
