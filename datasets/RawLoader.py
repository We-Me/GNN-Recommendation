import os

import pandas as pd


class RawLoader(object):
    COMMON_METHODS = [
        "user_item",
        "user_item_rate"
    ]
    TEMPORAL_METHODS = [
        "user_item_timestamp",
        "user_item_rate_timestamp",
        "user_item_edge_timestamp"
    ]

    def __init__(self,
                 data_dir: str,
                 user_item_file: str | None,
                 user_item_rate_file: str | None,
                 user_item_timestamp_file: str | None,
                 user_item_rate_timestamp_file: str | None,
                 user_item_edge_timestamp_file: str | None
                 ):
        self.data_dir = data_dir
        self.user_item_file = user_item_file
        self.user_item_rate_file = user_item_rate_file
        self.user_item_timestamp_file = user_item_timestamp_file
        self.user_item_rate_timestamp_file = user_item_rate_timestamp_file
        self.user_item_edge_timestamp_file = user_item_edge_timestamp_file

    def load_user_item(self, user_key: str, item_key: str):
        assert self.user_item_file is not None

        file_path = os.path.join(self.data_dir, self.user_item_file)
        raw_data = pd.read_csv(file_path, sep='\t', engine='python', on_bad_lines='warn')
        raw_data = raw_data.dropna(subset=[user_key, item_key])

        # 使用 factorize 高效编码
        raw_data['user_idx'], user_ids = pd.factorize(raw_data[user_key])
        raw_data['item_idx'], item_ids = pd.factorize(raw_data[item_key])

        idx2user = {i: uid for i, uid in enumerate(user_ids)}
        idx2item = {i: iid for i, iid in enumerate(item_ids)}

        data = raw_data[['user_idx', 'item_idx']].to_numpy()

        return data, idx2user, idx2item

    def load_user_item_rate(self, user_key: str, item_key: str, rate_key: str):
        assert self.user_item_rate_file is not None

        file_path = os.path.join(self.data_dir, self.user_item_rate_file)
        raw_data = pd.read_csv(file_path, sep='\t', engine='python', on_bad_lines='warn')
        raw_data = raw_data.dropna(subset=[user_key, item_key, rate_key])

        raw_data['user_idx'], user_ids = pd.factorize(raw_data[user_key])
        raw_data['item_idx'], item_ids = pd.factorize(raw_data[item_key])
        raw_data['rate'] = raw_data[rate_key]

        idx2user = {i: uid for i, uid in enumerate(user_ids)}
        idx2item = {i: iid for i, iid in enumerate(item_ids)}

        data = raw_data[['user_idx', 'item_idx', 'rate']].to_numpy()

        return data, idx2user, idx2item

    def load_user_item_timestamp(self, user_key: str, item_key: str, timestamp_key: str):
        assert self.user_item_timestamp_file is not None

        file_path = os.path.join(self.data_dir, self.user_item_timestamp_file)
        raw_data = pd.read_csv(file_path, sep='\t', engine='python', on_bad_lines='warn')
        raw_data = raw_data.dropna(subset=[user_key, item_key, timestamp_key])

        raw_data['user_idx'], user_ids = pd.factorize(raw_data[user_key])
        num_users = len(user_ids)
        raw_data['item_idx'], item_ids = pd.factorize(raw_data[item_key])
        raw_data['item_idx'] += num_users
        raw_data['timestamp'] = raw_data[timestamp_key].astype(int)

        idx2user = {i: uid for i, uid in enumerate(user_ids)}
        idx2item = {i + num_users: iid for i, iid in enumerate(item_ids)}

        sorted_data = raw_data[['user_idx', 'item_idx', 'timestamp']].sort_values(by='timestamp', ascending=True)
        data = sorted_data.to_numpy()
        return data, idx2user, idx2item

    def load_user_item_rate_timestamp(self, user_key: str, item_key: str, rate_key: str, timestamp_key: str):
        assert self.user_item_rate_timestamp_file is not None

        file_path = os.path.join(self.data_dir, self.user_item_rate_timestamp_file)
        raw_data = pd.read_csv(file_path, sep='\t', engine='python', on_bad_lines='warn')
        raw_data = raw_data.dropna(subset=[user_key, item_key, rate_key, timestamp_key])

        raw_data['user_idx'], user_ids = pd.factorize(raw_data[user_key])
        num_users = len(user_ids)
        raw_data['item_idx'], item_ids = pd.factorize(raw_data[item_key])
        raw_data['item_idx'] += num_users
        raw_data['timestamp'] = raw_data[timestamp_key].astype(int)
        raw_data['rate'] = raw_data[rate_key].astype(float)

        idx2user = {i: uid for i, uid in enumerate(user_ids)}
        idx2item = {i + num_users: iid for i, iid in enumerate(item_ids)}

        sorted_data = raw_data[['user_idx', 'item_idx', 'timestamp', 'rate']].sort_values(by='timestamp', ascending=True)
        data = sorted_data.to_numpy()
        return data, idx2user, idx2item

    def load_user_item_edge_timestamp(self, user_key: str, item_key: str, edge_key: str, timestamp_key: str):
        assert self.user_item_edge_timestamp_file is not None

        file_path = os.path.join(self.data_dir, self.user_item_edge_timestamp_file)
        raw_data = pd.read_csv(file_path, sep='\t', engine='python', on_bad_lines='warn')
        raw_data = raw_data.dropna(subset=[user_key, item_key, edge_key, timestamp_key])

        raw_data['user_idx'], user_ids = pd.factorize(raw_data[user_key])
        num_users = len(user_ids)
        raw_data['item_idx'], item_ids = pd.factorize(raw_data[item_key])
        raw_data['item_idx'] += num_users
        raw_data['timestamp'] = raw_data[timestamp_key]
        raw_data['edge_idx'], edge_ids = pd.factorize(raw_data[edge_key])

        idx2user = {i: uid for i, uid in enumerate(user_ids)}
        idx2item = {i + num_users: iid for i, iid in enumerate(item_ids)}
        idx2edge = {i: eid for i, eid in enumerate(edge_ids)}

        sorted_data = raw_data[['user_idx', 'item_idx', 'timestamp', 'edge_idx']].sort_values(by='timestamp', ascending=True)
        data = sorted_data.to_numpy()
        return data, idx2user, idx2item, idx2edge
