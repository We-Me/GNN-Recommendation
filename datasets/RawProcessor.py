from copy import deepcopy

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from datasets.RawLoader import RawLoader
from utils.matrix import build_weighted_interaction_matrix, build_interaction_matrix, build_normalized_adj_matrix


class RawProcessor(object):
    def __init__(self,
                 user_key: str,
                 item_key: str,
                 rate_key: str | None,
                 edge_key: str | None,
                 timestamp_key: str | None,
                 data_dir: str,
                 user_item_file: str | None = None,
                 user_item_rate_file: str | None = None,
                 user_item_timestamp_file: str | None = None,
                 user_item_rate_timestamp_file: str | None = None,
                 user_item_edge_timestamp_file: str | None = None,
                 load_method: str = None,
                 test_size: float = 0.2,
                 seed: int = 42
                 ):
        self.user_key = user_key
        self.item_key = item_key
        self.rate_key = rate_key
        self.edge_key = edge_key
        self.timestamp_key = timestamp_key
        self.data_dir = data_dir
        self.user_item_file = user_item_file
        self.user_item_rate_file = user_item_rate_file
        self.user_item_timestamp_file = user_item_timestamp_file
        self.user_item_rate_timestamp_file = user_item_rate_timestamp_file
        self.user_item_edge_timestamp_file = user_item_edge_timestamp_file

        assert (load_method in RawLoader.COMMON_METHODS or
                load_method in RawLoader.TEMPORAL_METHODS)
        self.load_method = load_method

        self.test_size = test_size
        self.seed = seed

        self.data = None
        self.idx2user = None
        self.idx2item = None
        self.idx2edge = None
        self.num_users = None
        self.num_items = None
        self.num_edges = None

        self.train_data = None
        self.test_data = None

        self._load_data()

    def _load_data(self):
        loader = RawLoader(self.data_dir,
                           self.user_item_file,
                           self.user_item_rate_file,
                           self.user_item_timestamp_file,
                           self.user_item_rate_timestamp_file,
                           self.user_item_edge_timestamp_file)

        if self.load_method == "user_item":
            self.data, self.idx2user, self.idx2item = loader.load_user_item(
                user_key=self.user_key, item_key=self.item_key)
        elif self.load_method == "user_item_rate":
            self.data, self.idx2user, self.idx2item = loader.load_user_item_rate(
                user_key=self.user_key, item_key=self.item_key,
                rate_key=self.rate_key)
        elif self.load_method == "user_item_timestamp":
            self.data, self.idx2user, self.idx2item = loader.load_user_item_timestamp(
                user_key=self.user_key, item_key=self.item_key,
                timestamp_key=self.timestamp_key)
        elif self.load_method == "user_item_rate_timestamp":
            self.data, self.idx2user, self.idx2item = loader.load_user_item_rate_timestamp(
                user_key=self.user_key, item_key=self.item_key,
                rate_key=self.rate_key, timestamp_key=self.timestamp_key)
        elif self.load_method == "user_item_edge_timestamp":
            self.data, self.idx2user, self.idx2item, self.idx2edge = loader.load_user_item_edge_timestamp(
                user_key=self.user_key, item_key=self.item_key,
                edge_key=self.edge_key, timestamp_key=self.timestamp_key)
            self.num_edges = len(self.idx2edge)
        else:
            raise ValueError("Invalid load_method. Please choose from 'user_item', 'user_item_timestamp', "
                             "'user_item_rate_timestamp', 'user_item_edge_timestamp'.")

        self.num_users = len(self.idx2user)
        self.num_items = len(self.idx2item)

    def _split_data(self):
        assert self.data is not None

        data = self.data
        train_list, test_list = [], []

        for user_idx in self.idx2user.keys():
            user_data = data[data[:, 0] == user_idx]
            if self.load_method in RawLoader.COMMON_METHODS:
                if len(user_data) > 1:
                    train_u, test_u = train_test_split(user_data,
                                                       test_size=self.test_size,
                                                       random_state=self.seed)
                    train_list.append(train_u)
                    test_list.append(test_u)
                else:
                    train_list.append(user_data)
            else:
                sorted_user_data = user_data[user_data[:, 2].argsort()]
                test_len = int(len(sorted_user_data) * self.test_size)
                if test_len == 0:
                    train_list.append(sorted_user_data)
                else:
                    train_u = sorted_user_data[:-test_len]
                    test_u = sorted_user_data[-test_len:]
                    train_list.append(train_u)
                    test_list.append(test_u)

        train_data = np.vstack(train_list) if train_list else np.empty((0, data.shape[1]))
        test_data = np.vstack(test_list) if test_list else np.empty((0, data.shape[1]))

        if self.load_method in RawLoader.TEMPORAL_METHODS:
            train_data = train_data[train_data[:, 2].argsort()]
            test_data = test_data[test_data[:, 2].argsort()]

        self.train_data = torch.tensor(train_data, dtype=torch.float32)
        self.test_data = torch.tensor(test_data, dtype=torch.float32)

    def get_data(self):
        return deepcopy(self.data)

    def get_train_test(self):
        if self.train_data is None or self.test_data is None:
            self._split_data()

        return deepcopy(self.train_data), deepcopy(self.test_data)

    def get_num_users(self):
        return self.num_users

    def get_num_items(self):
        return self.num_items

    def get_idx2user(self):
        return self.idx2user

    def get_idx2item(self):
        return self.idx2item

    def get_interaction_matrix(self):
        if self.train_data is None or self.test_data is None:
            self._split_data()

        if self.load_method == "user_item_rate" or self.load_method == "user_item_rate_timestamp":
            return build_weighted_interaction_matrix(self.train_data, self.num_users, self.num_items)
        else:
            return build_interaction_matrix(self.train_data, self.num_users, self.num_items)

    def get_norm_adj(self):
        if self.train_data is None or self.test_data is None:
            self._split_data()

        assert self.load_method in RawLoader.COMMON_METHODS
        return build_normalized_adj_matrix(self.train_data, self.num_users, self.num_items)
