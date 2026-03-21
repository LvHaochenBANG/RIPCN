import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from multiprocessing import Pool, cpu_count

import torch
from torch.utils.data import TensorDataset, DataLoader


class SeriesWindowProcessor:
    def __init__(self, data_path, 
                 history_window=12, future_window=12,
                 train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        # 初始化参数
        self.data_path = data_path
        self.history_window = history_window
        self.future_window = future_window
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # 数据相关变量
        self.data = None          # 原始数据 (T, N)
        self.train_data = None; self.val_data = None; self.test_data = None
        self.train_targets = None; self.val_targets = None; self.test_targets = None

        # 标准化参数
        self.mean = None
        self.std = None
        
        # 执行处理流程
        self.load_data()
        self.split_datasets()              # 先划分
        self.fit_scaler()                  # 用训练集计算标准化参数
        self.apply_scaler()                # 对三部分标准化
        self.create_history_future_windows()  # 再划窗

    def load_data(self):
        # 加载时空数据并提取F的第0维度
        data = np.load(self.data_path)
        if len(data.shape) != 3:
            raise ValueError("时空数据必须是3维结构 (T, N, F)")
        self.data = data[..., 0]  # 提取F的第0维度，得到(T, N)

    def split_datasets(self):
        # 按时间顺序划分原始数据
        T = self.data.shape[0]
        train_split = int(T * self.train_ratio)
        val_split = int(T * (self.train_ratio + self.val_ratio))

        self.train_raw = self.data[:train_split]
        self.val_raw = self.data[train_split:val_split]
        self.test_raw = self.data[val_split:]

        print(f"数据划分完成: 训练集{len(self.train_raw)} | 验证集{len(self.val_raw)} | 测试集{len(self.test_raw)}")

    def fit_scaler(self):
        # 计算训练集的均值和标准差
        self.mean = self.train_raw.mean(axis=(0, 1), keepdims=False)
        self.std = self.train_raw.std(axis=(0, 1), keepdims=False) + 1e-6

    def apply_scaler(self):
        # 使用训练集参数标准化
        self.train_norm = (self.train_raw - self.mean) / self.std
        self.val_norm = (self.val_raw - self.mean) / self.std
        self.test_norm = (self.test_raw - self.mean) / self.std

    def reverse_normalization(self, data):
        # 反标准化
        return data * self.std + self.mean

    def create_windows(self, data):
        # 生成历史窗口与未来窗口对
        T, N = data.shape
        window_size = self.history_window + self.future_window
        n_samples = T - window_size + 1

        samples = np.zeros((n_samples, self.history_window, N))
        targets = np.zeros((n_samples, self.future_window, N))

        for i in range(n_samples):
            end_idx = i + window_size
            samples[i] = data[i:i+self.history_window, :]
            targets[i] = data[i+self.history_window:end_idx, :]

        return samples, targets

    def create_history_future_windows(self):
        # 分别对训练/验证/测试集划窗
        self.train_data, self.train_targets = self.create_windows(self.train_norm)
        self.val_data, self.val_targets = self.create_windows(self.val_norm)
        self.test_data, self.test_targets = self.create_windows(self.test_norm)

        print(f"窗口生成完成: 训练{len(self.train_data)} | 验证{len(self.val_data)} | 测试{len(self.test_data)}")


class WindowProcessorWithStats(SeriesWindowProcessor):
    def __init__(self, name, data_path,
                 history_window=12, future_window=12,
                 train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, device=None):
        super().__init__(data_path, 
                         history_window, future_window,
                         train_ratio, val_ratio, test_ratio)
        self.name = name
        self.device = device

    def _prepare_split_with_encoder_decoder(self, samples, targets):
        xs = torch.as_tensor(samples, dtype=torch.float32)[..., None]  # (B, T, N, 1)
        ys = torch.as_tensor(targets, dtype=torch.float32)[..., None]  # (B, T, N, 1)

        # decoder_inputs 由最后一个历史步骤和已知目标的一部分拼接而成
        # xs: (B, history_window, N, 1), ys: (B, future_window, N, 1)
        decoder_inputs = torch.cat(
            [
                xs[:, -1:, :, :],     # [B, 1, N, 1]，最后一个历史时间
                ys[:, :-1, :, :]     # [B, future_window-1, N, 1]，未来目标（除去最后一帧）
            ],
            dim=1
        )

        # encoder_inputs: (B, N, history_window, 1)
        encoder_inputs = xs[:, :self.history_window].transpose(1, 2).contiguous()
        # decoder_inputs: (B, N, future_window, 1)
        decoder_inputs = decoder_inputs.transpose(1, 2).contiguous()
        # ys: (B, N, future_window, 1)
        ys = ys.transpose(1, 2).contiguous()

        return TensorDataset(encoder_inputs, decoder_inputs, ys)
    
    def get_dynamic_dataset(self, split):
        if split == 'train':# 准备三份数据
            print("Preparing train dynamic dataset...")
            self.train_tensors = self._prepare_split_with_encoder_decoder(self.train_data, self.train_targets)
            return self.train_tensors
        elif split == 'val':
            print("Preparing val dynamic dataset...")
            self.val_tensors = self._prepare_split_with_encoder_decoder(self.val_data, self.val_targets)
            return self.val_tensors
        elif split == 'test':
            print("Preparing test dynamic dataset...")
            self.test_tensors = self._prepare_split_with_encoder_decoder(self.test_data, self.test_targets)
            return self.test_tensors
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")
    
    
    # ============ 主函数中测试静态、动态 Dataset ===========
if __name__ == '__main__':
    # ==== 数据处理流程演示 ====
    # 1. 初始化数据处理器
    processor = WindowProcessorWithStats(
        name='camels-13',
        data_path='/data/LvHaochen/Diffme/data/camels-13/hydrology.npy',
        start_time='1980-01-01',
        freq='1D',
        history_window=12,    # 历史窗口大小
        future_window=12,     # 预测窗口大小
        train_ratio=0.8,      # 训练集比例
        val_ratio=0.1,        # 验证集比例
        test_ratio=0.1        # 测试集比例
    )

    # 2. 获取静态分布 dataset（dynamic_split=全0）
    train_len = len(processor.train_data)
    val_len = len(processor.val_data)
    test_len = len(processor.test_data)
    # 构造与样本数量相同的全0 dynamic_split
    train_dynamic_zero = torch.zeros((train_len, processor.history_window + processor.future_window, 1))
    val_dynamic_zero = torch.zeros((val_len, processor.history_window + processor.future_window, 1))
    test_dynamic_zero = torch.zeros((test_len, processor.history_window + processor.future_window, 1))
    train_dataset = processor.get_dataset('train', dynamic_split=train_dynamic_zero)
    val_dataset = processor.get_dataset('val', dynamic_split=val_dynamic_zero)
    test_dataset = processor.get_dataset('test', dynamic_split=test_dynamic_zero)

    # 3. 获取动态生成（encoder/decoder） dataset
    train_dyn_dataset = processor.get_dynamic_dataset('train')
    val_dyn_dataset = processor.get_dynamic_dataset('val')
    test_dyn_dataset = processor.get_dynamic_dataset('test')

    # 4. 构建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    train_dyn_loader = DataLoader(train_dyn_dataset, batch_size=32, shuffle=True)
    val_dyn_loader = DataLoader(val_dyn_dataset, batch_size=32, shuffle=False)
    test_dyn_loader = DataLoader(test_dyn_dataset, batch_size=32, shuffle=False)

    # 6. 动态（encoder/decoder）Dataset示例
    print("=== 动态分布 (encoder/decoder) Dataset 示例 ===")
    for batch in train_dyn_loader:
        encoder_inputs, decoder_inputs, ys = batch
        print(f"encoder_inputs shape: {encoder_inputs.shape}")    # [batch, N, history_window, 1]
        print(f"decoder_inputs shape: {decoder_inputs.shape}")    # [batch, N, future_window, 1]
        print(f"ys shape: {ys.shape}")                            # [batch, N, future_window, 1]
        break
    
    # 5. 静态Dataset示例
    print("=== Dataset 示例 ===")
    for batch in train_loader:
        x_masked, x0, mean, var, dynamic, t = batch
        print(f"x_masked shape: {x_masked.shape}")     # [batch, N, history+future_window, 1]
        print(f"x0 shape: {x0.shape}")                 # [batch, N, history+future_window, 1]
        print(f"mean shape (均值): {mean.shape}")       # [batch, N, history+future_window, 1]
        print(f"var shape (方差): {var.shape}")         # [batch, N, history+future_window, 1]
        print(f"dynamic shape: {dynamic.shape}")       # [batch, N, history+future_window, 1]
        print(f"t shape (时间戳): {t.shape}")           # [batch]
        break