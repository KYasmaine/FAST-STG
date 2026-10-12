# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def search_recent_data(train, label_start_idx, T_p, T_h):
    """
    T_p: prediction time steps
    T_h: historical time steps
    """
    if label_start_idx + T_p > len(train): return None
    start_idx, end_idx = label_start_idx - T_h, label_start_idx - T_p + T_p
    if start_idx < 0 or end_idx < 0: return None
    return (start_idx, end_idx), (label_start_idx, label_start_idx + T_p)


def search_multihop_neighbor(adj, hops=5):
    node_cnt = adj.shape[0]
    hop_arr = np.zeros((adj.shape[0], adj.shape[0]))
    for h_idx in range(node_cnt):  # refer node idx(n)
        tmp_h_node, tmp_neibor_step = [h_idx], [h_idx]  # save spatial corr node  # 0 step(self) first
        hop_arr[h_idx, :] = -1  # if the value exceed maximum hop, it is set to (hops + 1)
        hop_arr[h_idx, h_idx] = 0  # at begin, the hop of self->self is set to 0
        for hop_idx in range(hops):  # how many spatial steps
            tmp_step_node = []  # neighbor nodes in the previous k step
            tmp_step_node_kth = []  # neighbor nodes in the kth step
            for tmp_nei_node in tmp_neibor_step:
                tmp_neibor_step = list((np.argwhere(adj[tmp_nei_node] == 1).flatten()))  # find the one step neighbor first
                tmp_step_node += tmp_neibor_step
                tmp_step_node_kth += set(tmp_step_node) - set(tmp_h_node)  # the nodes that have appeared in the first k-1 step are no longer needed
                tmp_h_node += tmp_neibor_step
            tmp_neibor_step = tmp_step_node_kth.copy()
            all_spatial_node = list(set(tmp_neibor_step))  # the all spatial node in kth step
            hop_arr[h_idx, all_spatial_node] = hop_idx + 1
    return hop_arr[:, :, np.newaxis]

class CleanDataset():
    def __init__(self, config):

        self.data_name = config.data.name
        self.feature_file = config.data.feature_file
        self.val_start_idx = config.data.val_start_idx
        self.air_nan_fill = config.data.get('air_nan_fill', 'zero')
        self.adj = np.load(config.data.spatial)
        self.observed_mask = None
        self.label, self.feature = self.read_data()

        #for stpgcn
        if config.model.get('alpha', None) is not None:
            self.alpha = config.model.alpha
            self.t_size = config.model.t_size
            self.spatial_distance = search_multihop_neighbor(self.adj, hops=self.alpha)
            self.range_mask = self.interaction_range_mask(hops=self.alpha, t_size=self.t_size)

    @staticmethod
    def _fill_missing_air(data: np.ndarray, strategy: str, train_end=None) -> np.ndarray:
        """
        data: (T, V, 1) with possible NaN.
        strategy:
            - zero: NaN -> 0
            - node_mean: NaN -> training-only per-node mean (fallback 0)
            - ffill: causal forward fill, leading missing values -> 0
        """
        x = data[..., 0]
        if strategy == 'zero':
            out = np.nan_to_num(x, nan=0.0)
            return out[..., None]
        if strategy == 'node_mean':
            out = x.copy()
            if train_end is None:
                raise ValueError('node_mean requires the training boundary')
            node_mean = np.nanmean(out[:train_end], axis=0)
            node_mean = np.where(np.isfinite(node_mean), node_mean, 0.0)
            nan_idx = np.isnan(out)
            if nan_idx.any():
                out[nan_idx] = np.take(node_mean, np.where(nan_idx)[1])
            return out[..., None]
        if strategy == 'ffill':
            df = pd.DataFrame(x)
            out = df.ffill().values
            out = np.nan_to_num(out, nan=0.0)
            return out[..., None]
        raise ValueError(f"Unknown AIR nan fill strategy: {strategy}")

    def read_data(self):
        if 'PEMS' in self.data_name:
            data = np.expand_dims(np.load(self.feature_file)[:, :, 0], -1)
        elif 'AIR' in self.data_name:
            data_raw = np.expand_dims(np.load(self.feature_file)[:, :, 0], -1)
            self.observed_mask = np.isfinite(data_raw).astype('float32')
            data = self._fill_missing_air(data_raw, self.air_nan_fill, self.val_start_idx)
        elif 'Metro' in self.data_name:
            data = np.expand_dims(np.load(self.feature_file)[:, :, 0], -1)
            data = np.nan_to_num(data, nan=0)
        else:
            data = np.load(self.feature_file)
        # return data.astype('float32'), self.normalization(data).astype('float32')
        feature = self.normalization(data).astype('float32')
        label = feature.copy()
        if self.observed_mask is not None:
            # Preserve missing targets instead of supervising/evaluating fills.
            label[~self.observed_mask.astype(bool)] = np.nan
        return label, feature





    def normalization(self, feature):
        train = feature[:self.val_start_idx]
        # if 'Metro' in self.data_name:
        #     idx_lst = [i for i in range(train.shape[0]) if i % (24 * 6) >= 7 * 6 - 12]
        #     train = train[idx_lst]

        if self.observed_mask is not None:
            obs = self.observed_mask[:self.val_start_idx].astype(bool)
            train_obs = train[obs]
            if train_obs.size > 0:
                mean = np.mean(train_obs)
                std = np.std(train_obs)
            else:
                raise ValueError('No observed training targets for normalization')
        else:
            mean = np.mean(train)
            std = np.std(train)
        std = std if std > 1e-8 else 1.0

        # since the feature is actual the flow, the mean and std of feature is also the label's mean and std
        self.mean = mean
        self.std = std
        return (feature - mean) / std

    def reverse_normalization(self, x):
        return self.mean + self.std * x

    # for stpgcn
    def interaction_range_mask(self, hops=2, t_size=3):
        hop_arr = self.spatial_distance
        hop_arr[hop_arr != -1] = 1
        hop_arr[hop_arr == -1] = 0
        return np.concatenate([hop_arr.squeeze()] * t_size, axis=-1)  # V,tV



class TrafficDataset(Dataset):
    def __init__(self, clean_data, data_range, config):
        self.T_h = config.model.T_h
        self.T_p = config.model.T_p
        self.V = config.model.V
        self.points_per_hour = config.data.points_per_hour
        self.steps_per_day = int(getattr(config.data, 'steps_per_day', self.points_per_hour * 24))
        self.data_range = data_range
        self.data_name = clean_data.data_name


        self.label = np.array(clean_data.label) # (T_total, V, D), where T_all means the total time steps in the data
        self.feature = np.array(clean_data.feature)  # (T_total, V, D)

        # Prepare samples
        self.idx_lst = self.get_idx_lst()
        print('sample num:', len(self.idx_lst))

    def __getitem__(self, index):

        recent_idx = self.idx_lst[index]

        start, end = recent_idx[1][0], recent_idx[1][1]
        label = self.label[start:end]

        start, end = recent_idx[0][0], recent_idx[0][1]
        node_feature = self.feature[start:end]
        pos_w, pos_d = self.get_time_pos(start)
        pos_w = np.array(pos_w, dtype=np.int32)
        pos_d = np.array(pos_d, dtype=np.int32)
        # NOTE:
        # Some environments have broken numpy<->torch bridge (torch.as_tensor /
        # torch.from_numpy failing on np.ndarray/np.scalar). Returning tensors
        # directly avoids DataLoader default-collate conversion issues.
        label_t = torch.tensor(label.tolist(), dtype=torch.float32)
        node_feature_t = torch.tensor(node_feature.tolist(), dtype=torch.float32)
        pos_w_t = torch.tensor(pos_w.tolist(), dtype=torch.long)
        pos_d_t = torch.tensor(pos_d.tolist(), dtype=torch.long)
        return label_t, node_feature_t, pos_w_t, pos_d_t

    def __len__(self):
        return len(self.idx_lst)

    def get_time_pos(self, idx):
        idx = np.array(range(self.T_h)) + idx
        pos_w = (idx // self.steps_per_day) % 7  # day of week
        pos_d = idx % self.steps_per_day  # time of day
        return pos_w, pos_d

    def get_idx_lst(self):
        idx_lst = []
        start = self.data_range[0]
        end = self.data_range[1] if self.data_range[1] != -1 else self.feature.shape[0]

        for label_start_idx in range(start, end):
            # only 6:00-24:00 for Metro data
            if 'Metro' in self.data_name:
                if label_start_idx % (24 * 6) < (7 * 6):
                    continue
                if label_start_idx % (24 * 6) > (24 * 6) - self.T_p:
                    continue

            recent = search_recent_data(self.feature, label_start_idx, self.T_p, self.T_h)  # recent data

            if recent and np.isfinite(self.label[recent[1][0]:recent[1][1]]).any():
                idx_lst.append(recent)
        return idx_lst

    #################################################


if __name__ == '__main__':
    pass
