import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
from time import time
from torch.nn import MSELoss


def plot_loss_curve(loss_values, title='Loss Curve', save_path=None):
    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, 'b', label='Training loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)
        

def gram_schmidt(x):
    x_shape = x.shape  # (B,n_dirs,1,28,28) n_dirs个方向
    x = x.flatten(2)  # (B,n_dirs,28^2)

    x_orth = []
    proj_vec_list = []
    for i in range(x.shape[1]):
        w = x[:, i, :]  # (B,28^2)
        for w2 in proj_vec_list:
            w = w - w2 * torch.sum(w * w2, dim=-1, keepdim=True)  # 正交化
        w_hat = w.detach() / w.detach().norm(dim=-1, keepdim=True)  # 单位化 遍历过的w不再计算梯度，即正交方向确定了，后面的向量正交化不能影响前面的

        x_orth.append(w)  # 非单位化的正交方向
        proj_vec_list.append(w_hat)  # 单位正交基，旋转操作

    x_orth = torch.stack(x_orth, dim=1).view(*x_shape)  # 作为PC  (B,n_dirs,28^2) -> (B,n_dirs,1,28,28)
    proj_vec_list = torch.stack(proj_vec_list, dim=1).view(*x_shape)
    return x_orth, proj_vec_list  # 非单位化的正交方向，单位正交基


def binary_search_minimize(f, left, right, tolerance=0.001):
    while right - left > tolerance:
        mid1 = left + (right - left) / 3
        mid2 = right - (right - left) / 3
        f1 = f(t=mid1)
        f2 = f(t=mid2)
        
        if f1 < f2:
            right = mid2
        else:
            left = mid1
    best_t = (left + right) / 2
    return best_t, f(t=best_t)  

def f(x, x_hat, z, t, npc_idx):
    """
    Calculate the corrected mae by npc
    """
    samples = x_hat + t * z[:,npc_idx-1:npc_idx]
    samples = np.clip(samples, 0, np.inf)
    mask = (~np.isnan(x)).astype('float32')
    mask /= mask.mean()
    mae = np.abs(x - samples)
    return np.mean(np.nan_to_num(mask * mae))


class NPPCTrainer:
    '''
    Trainer for NPPC
    '''
    def __init__(self, dataset, train_loader, eval_loader, test_loader, model, config):
        self.dataset = dataset
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, betas=(0.9, 0.999))
        self.model_path = config.PATH_MODEL + '/nppc_net.model'
        self.loss_path = config.PATH_NPPC + '/nppc_loss'
        self.begin_eval = config.start_epoch  # start to eval
        self.patience = config.early_stop  # early stop patience
        self.counter = 0
        self.best_mae = np.inf
        self.best_t = None
        
        self.cap = torch.from_numpy(config.model.cap).to(self.config.device).to(torch.float32)[None,None,:,None]
        self.max_speed = torch.from_numpy(config.model.max_speed).to(self.config.device).to(torch.float32)[None,None,:,None]
        self.ttu = torch.from_numpy(config.model.ttu).to(self.config.device).to(torch.float32)[None,None,:,None]
        self.r_loss = MSELoss()
        self.metric = MetricCalculator(self.config.logger, dataset.dataset_name)
        
    def get_npc(self, batch):
        input, mean_prediction, target = [b.to(self.config.device).transpose(1, 3) for b in batch]
        history_reverse = self.dataset.reverse_normalization(input.transpose(1, 3))  # [B, 12, 323, 1]
        r_h = self.ttu * (1 + 0.15 * torch.pow(history_reverse/self.cap, 4)) / self.max_speed 
        x_masked = torch.cat((input, torch.zeros_like(target)), dim=3)
        x_mean = torch.cat((input, mean_prediction), dim=3)
        x_nppc = torch.cat((x_masked, x_mean), dim=1)
        r_p, w_mat = self.model(x_nppc, r_h, t=None, c=(None, None, None))  # (B,n_dirs,170,24)
        return target, mean_prediction, r_p, w_mat

        
    def get_loss_weights(self, nppc_step):
        def ramp(x, start, end):
            value = (x - start) / (end - start)
            value = max(min(value, 1), 0)
            return max(value ** 2, 1e-6)
        lambda_1 = 1 - ramp(nppc_step, 50, 150)
        lambda_2 = ramp(nppc_step, 20, 40)   
        lambda_3 = ramp(nppc_step, 120, 140)
        lambda_3 = ramp(nppc_step, 100, 120)
        return 0, 1, lambda_3

                
    def train_epoch(self):
        loss_b = []
        lambda_1, lambda_2, lambda_3 = self.get_loss_weights(self.nppc_step)
        for i, batch in enumerate(tqdm(self.train_loader, leave=False, desc='Training batch')):
            x, restore_data, r_p, w_mat = self.get_npc(batch)

            future_reverse = self.dataset.reverse_normalization(x[...,-self.config.T_p:])
            r_true = self.ttu * (1 + 0.15 * torch.pow(future_reverse/self.cap, 4)) / self.max_speed
            
            w_mat, _ = gram_schmidt(w_mat)  # 正交化，(B,n_dirs,1,170,24)
            w_mat_ = w_mat.flatten(2)  # (B,n_dirs,170*24)
            w_norms = w_mat_.norm(dim=2)
            w_hat_mat = w_mat_ / w_norms[:, :, None]  # 单位化，单位正交基，代表旋转矩阵

            err = (x - restore_data).flatten(1)
            # Normalizing by the error's norm
            err_norm = err.norm(dim=1)
            err = err / err_norm[:, None]
            w_norms = w_norms / err_norm[:, None]

            ## W hat loss
            err_proj = torch.einsum('bki,bi->bk', w_hat_mat, err)  # (B,n_dirs,170*24) * (B,170*24) -> (B,n_dirs) 实现了特征降维，28*28 -> n_dirs
            reconst_err = 1 - err_proj.pow(2).sum(dim=1)  # 目标是样本新特征的方差最大，则说明主成分的准确

            ## W norms loss
            second_moment_mse = (w_norms.pow(2) - err_proj.detach().pow(2)).pow(2)  # 希望非单位化的正交方向的模就是方差，即方向是数据变化最大的方向（旋转），方向的模表示变化的幅度（拉伸），这也是我们希望得到目标主成分。
            
            objective = lambda_1 * self.r_loss(r_p, r_true) + lambda_2 * reconst_err.mean() + lambda_3 * second_moment_mse.mean()

            self.optimizer.zero_grad()
            objective.backward()
            self.optimizer.step()
            loss_b.append(objective.item())
        return np.mean(loss_b)
    
    
    def train(self):
        self.model.train()
        loss_e = []
        self.nppc_step = 0
        best_epoch = 0
        for e in tqdm(range(self.config.nppc_epoch), desc='Training epoch'):
            self.nppc_step += 1
            # self.nppc_step = 0
            l_e = self.train_epoch()
            loss_e.append(l_e)
            message = f"epoch {e+1}: {l_e}\n"
            self.config.logger.message_buffer += f"{message}"
            self.config.logger.write_message_buffer()
            
            if e < self.begin_eval: continue
            
            t, mae = self.eval()
            if mae < self.best_mae:
                best_epoch = e + 1
                self.best_mae = mae
                self.best_t = t
                self.counter = 0
                torch.save(self.model, self.model_path)
            else:
                self.counter += 1
                
            if self.counter > self.patience:
                break
            
        message = f"\nbest_epoch: {best_epoch}\n"
        self.config.logger.message_buffer += f"{message}"
        self.config.logger.write_message_buffer()
        plot_loss_curve(loss_e, save_path=self.loss_path)
    
    
    def eval(self):
        self.model.eval()
        wzd_list = []
        wz_list = []
        x_list = []
        x_hat_list = []

        for i, batch in enumerate(self.eval_loader):
            
            x, restore_data, r_p, w_mat = self.get_npc(batch)
            
            w_mat = w_mat[:,:,:,-self.config.T_p:].detach().cpu()
            
            w_z, w_zd = gram_schmidt(w_mat)  # 正交化，(B,n_dirs,1,170,24)
            x = self.dataset.reverse_normalization(x)
            restore_data = self.dataset.reverse_normalization(restore_data)
            wz_list.append(w_z.cpu().detach().numpy())
            wzd_list.append(w_zd.cpu().detach().numpy())
            x_list.append(x.cpu().detach().numpy())
            x_hat_list.append(restore_data.cpu().detach().numpy())
            
        
        wz_list = np.concatenate(wz_list)
        wzd_list = np.concatenate(wzd_list)
        x_list = np.concatenate(x_list)
        x_hat_list = np.concatenate(x_hat_list)
        
        x = x_list[:,:,:,-self.config.T_p:]
        x_hat = x_hat_list[:,:,:,-self.config.T_p:]
        f_b = partial(f, x=x, x_hat=x_hat, z=wz_list, npc_idx=1)
        best_t, best_mae = binary_search_minimize(f_b, -1000, 1000)  # 二分法找最优 t
        message = f"best_t:{best_t}, best_mae:{best_mae}\n"
        self.config.logger.message_buffer += f"{message}"
        self.config.logger.write_message_buffer()
        return best_t,  best_mae
        
        
    
    def test(self):
        self.model.eval()
        wzd_list = []
        wz_list = []
        x_list = []
        x_hat_list = []

        time0 = time()
        for i, batch in enumerate(self.test_loader):
            x, restore_data, r_p, w_mat = self.get_npc(batch)
            
            w_mat = w_mat[:,:,:,-self.config.T_p:].detach().cpu()
            
            w_z, w_zd = gram_schmidt(w_mat)  # 正交化，(B,n_dirs,1,170,24)
            x = self.dataset.reverse_normalization(x)
            restore_data = self.dataset.reverse_normalization(restore_data)
            wz_list.append(w_z.cpu().detach().numpy())
            wzd_list.append(w_zd.cpu().detach().numpy())
            x_list.append(x.cpu().detach().numpy())
            x_hat_list.append(restore_data.cpu().detach().numpy())
        gen_time = time() - time0
        wz_list = np.concatenate(wz_list)
        wzd_list = np.concatenate(wzd_list)
        x_list = np.concatenate(x_list)
        x_hat_list = np.concatenate(x_hat_list)
        path = self.config.PATH_NPPC + '/output.npz'
        np.savez(path, x=x_list, x_hat=x_hat_list, z=wz_list, npc=wzd_list)
        
        x = x_list[:,:,:,-self.config.T_p:]
        x_hat = x_hat_list[:,:,:,-self.config.T_p:]
        f_b = partial(f, x=x, x_hat=x_hat, z=wz_list, npc_idx=1)
        test_mae = f_b(t=self.best_t)
        message = f"best_t:{self.best_t}, test_mae:{test_mae}, test_time:{gen_time}\n"
        self.config.logger.message_buffer += f"{message}"
        self.config.logger.write_message_buffer()
        self.metric.calculate(x, x_hat, wz_list, wzd_list, self.best_t)
        
        
        
    def load_model(self, model_path=None):
        save_path = self.model_path if model_path==None else model_path
        self.model = torch.load(save_path, map_location=self.config.device)
        print('Best model loaded from: <<', self.model_path)
        
    
    def save_model(self, model_path=None):
        save_path = self.model_path if model_path==None else model_path
        torch.save(self.model, save_path)
        print('Save model to: >>' + save_path)
        
    def get_t(self):
        return self.best_t





class MetricCalculator:
    """
    计算一系列指标的类
    用法:
        metric_calculator = MetricCalculator(logger=logger, dataset_name='Seattle')
        results = metric_calculator.calculate(x, x_hat, z, npc, t)
    """
    def __init__(self, logger=None, dataset_name='Seattle'):
        self.logger = logger
        self.dataset_name = dataset_name

    def mae(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred)).item()

    def rmse(self, y_true, y_pred):
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()

    def mape(self, y_true, y_pred):
        if self.dataset_name == 'Seattle':
            mask = y_true > 10
        else:
            mask = y_true > 1
        masked_true = y_true[mask]
        masked_pred = y_pred[mask]
        if masked_true.numel() == 0:
            return float('nan')
        return (torch.mean(torch.abs((masked_true - masked_pred) / masked_true)) * 100).item()

    def quantile_loss(self, target, forecast, q: float, eval_points):
        return 2 * torch.sum(
            torch.abs((forecast - target) * eval_points * ((target <= forecast).float() - q))
        )

    def calc_denominator(self, target, eval_points):
        return torch.sum(torch.abs(target * eval_points))

    def calc_quantile_CRPS(self, target, forecast, eval_points):
        """
        target: (B, T, V), torch.Tensor
        forecast: (B, n_sample, T, V), torch.Tensor
        eval_points: (B, T, V): which values should be evaluated,
        """
        import numpy as np
        quantiles = np.arange(0.05, 1.0, 0.05)
        denom = self.calc_denominator(target, eval_points)
        CRPS = 0
        for i in range(len(quantiles)):
            q_pred = []
            for j in range(len(forecast)):
                q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
            q_pred = torch.cat(q_pred, 0)
            q_loss = self.quantile_loss(target, q_pred, quantiles[i], eval_points)
            CRPS += q_loss / denom
        return (CRPS / len(quantiles)).item()

    def calculate_MISE(self, predicted_lower, predicted_upper, observed_values, rho=0.05):
        predicted_lower_flat = predicted_lower.flatten()
        predicted_upper_flat = predicted_upper.flatten()
        observed_values_flat = observed_values.flatten()
        interval_width = predicted_upper_flat - predicted_lower_flat
        over_penalty = (observed_values_flat - predicted_upper_flat) * (observed_values_flat > predicted_upper_flat) * (2 / rho)
        under_penalty = (predicted_lower_flat - observed_values_flat) * (observed_values_flat < predicted_lower_flat) * (2 / rho)
        total_MIS = torch.sum(interval_width + over_penalty + under_penalty)
        N = predicted_lower_flat.shape[0]
        MISE = total_MIS / N
        return MISE.item()

    def calculate(self, x, x_hat, z, npc, t):
        """
        计算一系列指标
        参数：
            x: 观测值, torch.Tensor 或 np.ndarray
            x_hat: 预测值, torch.Tensor 或 np.ndarray
            z: 正交向量, torch.Tensor 或 np.ndarray
            npc: npc向量, torch.Tensor 或 np.ndarray
            t: 外部给定的 float 数值
        返回:
            mae_val, rmse_val, mape_val, crps_val, mise_val
        """
        logger = self.logger

        if logger is not None:
            logger.message_buffer += f"cal_metric:\n"
            logger.write_message_buffer()

        # 数据预处理
        if not torch.is_tensor(z):
            z = torch.tensor(z)
        z = z.norm(2, dim=(2, 3))
        if not torch.is_tensor(x_hat):
            x_hat = torch.tensor(x_hat)
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if not torch.is_tensor(npc):
            npc = torch.tensor(npc)

        x_hat = torch.clamp(x_hat, 0, torch.inf)
        x = torch.clamp(x, 0, torch.inf)
        npc_idx = 1

        # 预测结果
        prediction_re = x_hat + t * (z[:, npc_idx-1:npc_idx, None, None] * npc[:, npc_idx-1:npc_idx, :, :])
        prediction_re = torch.clamp(prediction_re, 0, torch.inf)
        target_re = torch.clamp(x, 0, torch.inf)

        # 全部先挤掉无用的batch维
        prediction_re_sq = prediction_re.squeeze()
        target_re_sq = target_re.squeeze()

        mae_val = self.mae(target_re_sq, prediction_re_sq)
        rmse_val = self.rmse(target_re_sq, prediction_re_sq)
        mape_val = self.mape(target_re_sq, prediction_re_sq)
        if logger is not None:
            logger.message_buffer += (
                f"MAE: {mae_val:.6f}\n"
                f"RMSE: {rmse_val:.6f}\n"
                f"MAPE: {mape_val:.6f}\n"
            )

        # 计算CRPS
        t_list = torch.linspace(t - 500, t + 500, 5, dtype=x_hat.dtype, device=x_hat.device)
        samples = x_hat[:, None, None, :, :, :] + t_list[None, :, None, None, None, None] * (
            z[:, None, npc_idx-1:npc_idx, None, None, None] * npc[:, None, npc_idx-1:npc_idx, None, :, :]
        )
        samples = samples.flatten(1, 2)
        samples = torch.clamp(samples, 0, torch.inf)
        samples = samples.squeeze()
        eval_points = torch.ones_like(target_re_sq)
        crps_val = self.calc_quantile_CRPS(target_re_sq, samples, eval_points)
        if logger is not None:
            logger.message_buffer += f"CRPS: {crps_val:.6f}\n"

        # 计算 MISE
        t_list_large = torch.linspace(t - 2000, t + 2000, 5, dtype=x_hat.dtype, device=x_hat.device)
        samples_large = x_hat[:, None, None, :, :, :] + t_list_large[None, :, None, None, None, None] * (
            z[:, None, npc_idx-1:npc_idx, None, None, None] * npc[:, None, npc_idx-1:npc_idx, None, :, :]
        )
        samples_large = samples_large.flatten(1, 2)
        samples_large = torch.clamp(samples_large, 0, torch.inf)
        samples_large = samples_large.squeeze()
        lower_percentile = torch.quantile(samples_large, 0.025, dim=1)
        upper_percentile = torch.quantile(samples_large, 0.975, dim=1)
        mise_val = self.calculate_MISE(lower_percentile, upper_percentile, target_re_sq)
        if logger is not None:
            logger.message_buffer += f"MISE: {mise_val:.6f}\n"
            logger.write_message_buffer()

        return mae_val, rmse_val, mape_val, crps_val, mise_val

