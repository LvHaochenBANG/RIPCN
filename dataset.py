import os
import numpy as np
import heapq
import torch
from torch.utils.data import TensorDataset

# def get_road_feature_old(name):
#     """
#     从指定路径加载数据，计算道路特征，并将计算结果保存到指定路径的文件中。
#     Returns:
#         tuple: 包含道路特征的元组 (capacity, max_speed, ttu)
#     """
#     input_path=f'./data/dataset/{name}/flow.npy'
#     output_path=f'./data/dataset/{name}/road_f.npz'

#     # 检查输出文件是否存在
#     if os.path.exists(output_path):
#         data = np.load(output_path)
#         capacity = data['capacity']
#         max_speed = data['max_speed']
#         ttu = data['ttu']
#     else:
#         data = np.load(input_path)
#         data_train = data[:int(data.shape[0]*0.6), :, :]

#         mean_s = np.percentile(data_train[:,:,2], 25, 0)
#         mean_o = np.percentile(data_train[:,:,1], 25, 0)

#         max_value = np.max(data_train[:,:,0], axis=0)
#         max_index = np.argmax(data_train[:,:,0], axis=0)

#         max_f_s = []
#         max_f_o = []
#         for i in range(data_train.shape[1]):
#             max_f_s.append(data_train[max_index[i], i, 2])
#             max_f_o.append(data_train[max_index[i], i, 1])

#         max_f_s = np.array(max_f_s)
#         max_f_o = np.array(max_f_o)

#         a = max_f_s / mean_s
#         b = mean_o / max_f_o
#         c = max_value*(np.clip(a+b, 0.8, 1.8))

#         max_speed = []
#         speed_10 = []
#         speed_90 = []
#         for r_idx in range(data_train.shape[1]):
#             max_speed.append(np.mean(heapq.nlargest(100, data_train[:, r_idx, 2])))
#             speed_10.append(np.percentile(data_train[:, r_idx, 2], 10))
#             speed_90.append(np.percentile(data_train[:, r_idx, 2], 90))

#         max_speed = np.array(max_speed)
#         speed_10 = np.array(speed_10)
#         speed_90 = np.array(speed_90)

#         ttu = (1/speed_10 - 1/speed_90)*1000

#         capacity = c
#         np.savez(output_path, capacity=capacity, max_speed=max_speed, ttu = ttu)

#     return capacity, max_speed, ttu


def get_road_feature(name):
    """
    从指定路径加载数据，计算道路特征，并将计算结果保存到指定路径的文件中。
    Returns:
        tuple: 包含道路特征的元组 (capacity, max_speed, ttu)
    """
    input_path=f'./data/dataset/{name}/flow.npy'
    output_path=f'./data/dataset/{name}/road_feature.npz'

    # 检查输出文件是否存在
    if os.path.exists(output_path):
        data = np.load(output_path)
        capacity = data['capacity']
        max_speed = data['max_speed']
        ttu = data['ttu']
    else:
        data = np.load(input_path)
        
        # 判断最后一维是否为3维
        if data.shape[-1] < 3:
            # 特征数不足3，直接用最大流量和速度赋值
            data_train = data[:int(data.shape[0]*0.6), :, :] if data.shape[0] > 0 else data
            # 万一N小于1，防止错误
            if data_train.shape[0] > 0:
                if len(data_train.shape) == 3:
                    flow_col = 0 if data_train.shape[2] > 0 else -1
                    max_value = np.max(data_train[:,:,flow_col], axis=0)
                elif len(data_train.shape) == 2:
                    max_value = np.max(data_train, axis=0)
                else:
                    max_value = np.max(data_train)
                capacity = max_value
            else:
                capacity = np.array([])
            max_speed = np.full_like(capacity, 75, dtype=np.float32)
            # 计算 ttu 需要 cv
            # 此时 data 可能只有一个流量特征
            flow = data_train[..., 0] if data_train.ndim >= 2 else data_train
            if flow.ndim > 1:  # [T, N]
                mu = np.mean(flow, axis=0) + 1e-6  # shape: [N]
                sigma = np.std(flow, axis=0)
            else:
                mu = np.mean(flow) + 1e-6
                sigma = np.std(flow)
            cv = sigma / mu
            gamma = 1.0
            delta = 1.0
            ttu = 1 + gamma * (cv ** delta)
            # ttu 与 capacity 形状一致
            if ttu.shape != capacity.shape:
                ttu = np.full_like(capacity, ttu, dtype=np.float32)
            np.savez(output_path, capacity=capacity, max_speed=max_speed, ttu=ttu)
        else:
            data_train = data[:int(data.shape[0]*0.6), :, :]
            mean_s = np.percentile(data_train[:,:,2], 25, 0)
            mean_o = np.percentile(data_train[:,:,1], 25, 0)

            max_value = np.max(data_train[:,:,0], axis=0)
            max_index = np.argmax(data_train[:,:,0], axis=0)

            max_f_s = []
            max_f_o = []
            for i in range(data_train.shape[1]):
                max_f_s.append(data_train[max_index[i], i, 2])
                max_f_o.append(data_train[max_index[i], i, 1])

            max_f_s = np.array(max_f_s)
            max_f_o = np.array(max_f_o)

            a = max_f_s / mean_s
            b = mean_o / max_f_o
            c = max_value*(np.clip(a+b, 0.8, 1.8))

            max_speed = []
            speed_10 = []
            speed_90 = []
            for r_idx in range(data_train.shape[1]):
                max_speed.append(np.mean(heapq.nlargest(100, data_train[:, r_idx, 2])))
                speed_10.append(np.percentile(data_train[:, r_idx, 2], 10))
                speed_90.append(np.percentile(data_train[:, r_idx, 2], 90))

            max_speed = np.array(max_speed)
            # 使用变异系数计算 TTU = 1 + gamma * (CV)^delta
            gamma = 1.0
            delta = 1.0

            flow = data_train[:, :, 0]  # shape: [T, N]
            mu = np.mean(flow, axis=0) + 1e-6  # 防止除以0
            sigma = np.std(flow, axis=0)
            cv = sigma / mu
            ttu = 1 + gamma * (cv ** delta)

            capacity = c
            np.savez(output_path, capacity=capacity, max_speed=max_speed, ttu = ttu)

    return capacity, max_speed, ttu

# class FeatureWindowDataset:
#     def __init__(self, input_path, history_len, forecast_len,
#                  feature_indices=None,
#                  split_ratio=(0.6, 0.2, 0.2)):
#         """
#         参数:
#             input_path: numpy array, shape (T, N, F)
#             history_len: 滑窗长度
#             forecast_len: 预测长度
#             feature_indices: 要使用的特征维索引 (在 F 维上)，如 [0, 1, 3]
#             split_ratio: 训练/验证/测试划分比例
#         """
#         data = np.load(input_path)
#         assert data.ndim == 3, "输入数据维度必须为 (T, N, F)"
#         assert sum(split_ratio) <= 1.0, "划分比例和不能超过1"

#         self.data = data
#         self.history_len = history_len
#         self.forecast_len = forecast_len
#         self.split_ratio = split_ratio
#         self.feature_indices = feature_indices if feature_indices is not None else list(range(data.shape[2]))

#         # 保留指定特征
#         self.data = self.data[:, :, self.feature_indices]

#     def split_data(self):
#         """
#         根据时间维划分数据集
#         """
#         T = self.data.shape[0]
#         r_train, r_val, _ = self.split_ratio

#         train_end = int(T * r_train)
#         val_end = int(T * (r_train + r_val))

#         train_data = self.data[:train_end]
#         val_data = self.data[train_end:val_end]
#         test_data = self.data[val_end:]

#         return train_data, val_data, test_data

#     def generate_sliding_windows(self, data):
#         """
#         对给定数据生成滑动窗口样本
#         返回:
#             X: (num_samples, history_len, N, F_selected)
#             Y: (num_samples, forecast_len, N, F_selected)
#         """
#         T = data.shape[0]
#         N = data.shape[1]
#         F = data.shape[2]

#         X, Y = [], []
#         for t in range(T - self.history_len - self.forecast_len + 1):
#             x = data[t:t+self.history_len]     # shape: (history_len, N, F)
#             y = data[t+self.history_len:t+self.history_len+self.forecast_len]
#             X.append(x)
#             Y.append(y)
#         return np.array(X), np.array(Y)
    
#     def compute_train_mean_std(self):
#         train_data, _, _ = self.split_data()
#         mean = np.mean(train_data, axis=(0, 1), keepdims=True)  # shape (1, 1, F)
#         std = np.std(train_data, axis=(0, 1), keepdims=True)
#         std[std == 0] = 1e-8
#         return mean, std
    
#     def get_normalizer(self):
#         """
#         返回两个函数：
#             normalize(x): 归一化
#             denormalize(x): 反归一化
#         """
#         mean, std = self.compute_train_mean_std()

#         def normalize(x):
#             return (x - mean) / std

#         def denormalize(x):
#             return x * std + mean

#         return normalize, denormalize

#     def get_datasets(self):
#         """
#         返回：
#             (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
#         """
#         train_data, val_data, test_data = self.split_data()
#         X_train, Y_train = self.generate_sliding_windows(train_data)
#         X_val, Y_val = self.generate_sliding_windows(val_data)
#         X_test, Y_test = self.generate_sliding_windows(test_data)
#         return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
    

class TrafficDataset:
    def __init__(self, root_dir, dataset_name, device='cpu'):
        """
        参数：
            root_dir: 根目录路径（不含 dataset 名）
            dataset_name: 数据集名，对应 npz 文件夹名
            device: 'cpu' 或 'cuda'
        """
        self.root_dir = root_dir.rstrip('/')
        self.dataset_name = dataset_name
        self.device = torch.device(device)

        # 加载所有数据
        self.train_data = self._load_npz('train_')  # B, T, N, F
        self.val_data = self._load_npz('eval_')
        self.test_data = self._load_npz('test_')

        # 创建 Dataset
        self._build_normalizer(self.train_data['input'])
        self.train_dataset = self._to_dataset(self.train_data)
        self.val_dataset = self._to_dataset(self.val_data)
        self.test_dataset = self._to_dataset(self.test_data)
        
        self.num_vertices = self.train_data['input'].shape[2]
        self.num_features = self.train_data['input'].shape[3]
        self.adj = np.load(f"{self.root_dir}/{self.dataset_name}/adj.npy")

    def _load_npz(self, split_name):
        """加载 .npz 文件并返回 numpy 字典"""
        path = f"{self.root_dir}/{self.dataset_name}/{split_name}.npz"
        data = np.load(path)
        return {
            'input': torch.from_numpy(data['input']).float().to(self.device),
            'mean_prediction': torch.from_numpy(data['mean_prediction']).float().to(self.device),
            'target': torch.from_numpy(data['target']).float().to(self.device)
        }

    def _to_dataset(self, data_dict):
        """将 numpy 字典转为 PyTorch TensorDataset，先对数据进行标准化处理"""
        # 对 input、mean_prediction 和 target 进行标准化处理
        normalized_input = self.normalization(data_dict['input'])
        normalized_mean_prediction = self.normalization(data_dict['mean_prediction'])
        normalized_target = self.normalization(data_dict['target'])
        
        return TensorDataset(normalized_input, normalized_mean_prediction, normalized_target)

    def _build_normalizer(self, input_tensor):
        """
        基于输入张量构建标准化函数
        - input_tensor: shape (N, ...) 训练样本的输入
        """
        # 计算均值和标准差：通常在特征维度上（如最后一维）
        dims = tuple(i for i in range(input_tensor.ndim) if i != -1)
        self.mean = input_tensor.mean(dim=dims, keepdim=True)  # 保留特征维
        self.std = input_tensor.std(dim=dims, keepdim=True)
        self.std[self.std == 0] = 1e-8  # 防止除以0

    def normalization(self, x):
        return (x - self.mean) / self.std

    def reverse_normalization(self, x):
        return x * self.std + self.mean

    def get_datasets(self):
        """返回 PyTorch 数据集 (train, val, test)"""
        return self.train_dataset, self.val_dataset, self.test_dataset


if __name__ == "__main__":
    get_road_feature('PEMS03')