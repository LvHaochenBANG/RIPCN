import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from time import time
from scipy.sparse.linalg import eigs
import pickle


def re_normalization(x, mean, std):
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


def dir_check(path):
    """
    check weather the dir of the given path exists, if not, then create it
    """
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)
    return path


def get_real_mu_bounds(y_pred, processor):
    """
    将网络输出的三个通道拆分成均值和区间，并反归一化到原始尺度
    ----------------------------------------------------------------
    y_pred: [B, N, T, 3] - 网络输出 (mu_raw, width_raw, shift_raw)
    processor: 数据归一化与反归一化的处理对象

    返回:
        mu_real: [B, N, T, 1] - 均值（原始尺度）
        lower_real: [B, N, T, 1] - 下界（原始尺度）
        upper_real: [B, N, T, 1] - 上界（原始尺度）
        all_bounds: [B, N, T, 3] - 堆叠后的 [mu, lower, upper]（原始尺度）
    """

    # ======== 拆分输出通道 ========
    mu = y_pred[..., 0:1]         # [B, N, T, 1]
    width_raw = y_pred[..., 1:2]  # [B, N, T, 1]
    shift_raw = y_pred[..., 2:3]  # [B, N, T, 1]

    # ======== 结构化区间构造 ========
    width = np.log1p(np.exp(width_raw))        # softplus，保证宽度大于0
    shift = np.tanh(shift_raw)                 # 映射到[-1, 1]
    y_lower = mu - width * (1 + shift) / 2     # [B, N, T, 1]
    y_upper = mu + width * (1 - shift) / 2     # [B, N, T, 1]

    # ======== 反归一化到原始尺度 ========
    mu_real = processor.reverse_normalization(mu)
    lower_real = processor.reverse_normalization(y_lower)
    upper_real = processor.reverse_normalization(y_upper)

    return np.concatenate([mu_real, lower_real, upper_real], axis=-1)


def get_real_mu_bounds_torch(y_pred):
    """
    将网络输出的三个通道拆分成均值和区间
    ----------------------------------------------------------------
    y_pred: [B, N, T, 3] - 网络输出 (mu_raw, width_raw, shift_raw) (tensor)

    返回:
        mu: [B, N, T, 1] - 均值
        lower: [B, N, T, 1] - 下界
        upper: [B, N, T, 1] - 上界
        all_bounds: [B, N, T, 3] - 堆叠后的 [mu, lower, upper]
    """

    # ======== 拆分输出通道 ========
    mu = y_pred[..., 0:1]         # [B, N, T, 1]
    width_raw = y_pred[..., 1:2]  # [B, N, T, 1]
    shift_raw = y_pred[..., 2:3]  # [B, N, T, 1]

    # ======== 结构化区间构造 ========
    width = F.softplus(width_raw)         # softplus，保证宽度大于0
    shift = torch.tanh(shift_raw)                    # 映射到[-1, 1]
    y_lower = mu - width * (1 + shift) / 2           # [B, N, T, 1]
    y_upper = mu + width * (1 - shift) / 2           # [B, N, T, 1]
    return torch.cat([mu, y_lower, y_upper], dim=-1)


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        # distance file中的id并不是从0开始的 所以要进行重新的映射；id_filename是节点的顺序
        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()  # 略过表头那一行
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:  # distance file中的id直接从0开始

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA


def get_adjacency_matrix_2direction(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        # distance file中的id并不是从0开始的 所以要进行重新的映射；id_filename是节点的顺序
        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()  # 略过表头那一行
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    A[id_dict[j], id_dict[i]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
                    distaneA[id_dict[j], id_dict[i]] = distance
            return A, distaneA

        else:  # distance file中的id直接从0开始

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    A[j, i] = 1
                    distaneA[i, j] = distance
                    distaneA[j, i] = distance
            return A, distaneA


def get_Laplacian(A):
    '''
    compute the graph Laplacian, which can be represented as L = D − A

    Parameters
    ----------
    A: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    Laplacian matrix: np.ndarray, shape (N, N)

    '''

    assert (A-A.transpose()).sum() == 0  # 首先确保A是一个对称矩阵

    D = np.diag(np.sum(A, axis=1))  # D是度矩阵，只有对角线上有元素

    L = D - A  # L是实对称矩阵A，有n个不同特征值对应的特征向量是正交的。

    return L


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))  # D是度矩阵，只有对角线上有元素

    L = D - W  # L是实对称矩阵A，有n个不同特征值对应的特征向量是正交的。

    lambda_max = eigs(L, k=1, which='LR')[0].real  # 求解拉普拉斯矩阵的最大奇异值

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def sym_norm_Adj(W):
    '''
    compute Symmetric normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    Symmetric normalized Laplacian: (D^hat)^1/2 A^hat (D^hat)^1/2; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    W = W + np.identity(N) # 为邻居矩阵加上自连接
    D = np.diag(np.sum(W, axis=1))
    sym_norm_Adj_matrix = np.dot(np.sqrt(D),W)
    sym_norm_Adj_matrix = np.dot(sym_norm_Adj_matrix,np.sqrt(D))

    return sym_norm_Adj_matrix


def norm_Adj(W):
    '''
    compute  normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    normalized Adj matrix: (D^hat)^{-1} A^hat; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    W = W + np.identity(N)  # 为邻接矩阵加上自连接
    D = np.diag(1.0/np.sum(W, axis=1))
    norm_Adj_matrix = np.dot(D, W)

    return norm_Adj_matrix


def trans_norm_Adj(W):
    '''
    compute  normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    Symmetric normalized Laplacian: (D^hat)^1/2 A^hat (D^hat)^1/2; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    W = W.transpose()
    N = W.shape[0]
    W = W + np.identity(N)  # 为邻居矩阵加上自连接
    D = np.diag(1.0/np.sum(W, axis=1))
    trans_norm_Adj = np.dot(D, W)

    return trans_norm_Adj


def compute_val_loss(net, val_loader, criterion, num_for_predict, logger, DEVICE):
    """
    计算验证集上的平均损失
    """
    net.train(False)  # ensure dropout layers are in evaluation mode
    with torch.no_grad():
        val_loader_length = len(val_loader)  # nb of batch
        tmp = []  # 记录了所有batch的loss
        start_time = time()
        for batch_index, batch_data in enumerate(val_loader):
            encoder_inputs, decoder_inputs, labels  = [m.to(DEVICE, non_blocking=True) for m in batch_data]
            predict_length = labels.shape[2]  # T
            # encode
            encoder_output = net.encode(encoder_inputs)
            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]
            decoder_input_list = [decoder_start_inputs]
            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, predict_output[:, :, :, :1]]

            loss = criterion(predict_output, labels)  # 计算误差
            tmp.append(loss.item())
            if batch_index % 100 == 0:
                logger.write('validation batch %s / %s, loss: %.2f\n' % (batch_index + 1, val_loader_length, loss.item()), is_terminal=False)

        logger.write('validation cost time: %.4fs\n' %(time()-start_time), is_terminal=False)

        validation_loss = sum(tmp) / len(tmp)
        logger.write(f'validation_loss: {validation_loss}\n', is_terminal=False)

    return validation_loss


def predict_and_save_results(net, data_loader, processor, num_for_predict, forecast_path, type, logger, DEVICE):
    net.train(False)  # ensure dropout layers are in test mode
    start_time = time()
    with torch.no_grad():
        loader_length = len(data_loader)  # nb of batch
        prediction = []
        input = []  # 存储所有batch的input
        target = []  # 存储所有batch的target
        start_time = time()
        for batch_index, batch_data in enumerate(data_loader):
            encoder_inputs, decoder_inputs, labels = [m.to(DEVICE, non_blocking=True) for m in batch_data]
            # encode
            encoder_output = net.encode(encoder_inputs)
            input.append(encoder_inputs[:, :, :, 0:1].cpu().numpy())  # (batch, N, T, 1)
            target.append(labels.cpu().numpy())
            # decode
            predict_length = labels.shape[2]  # T   
            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]
            decoder_input_list = [decoder_start_inputs]
            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, predict_output[:, :, :, :1]]
            
            prediction.append(predict_output.detach().cpu().numpy())
            if batch_index % 100 == 0:
                logger.write('predicting testing set batch %s / %s, time: %.2fs\n' % (batch_index + 1, loader_length, time() - start_time), is_terminal=False)

        logger.write('test time on whole data:%.2fs\n' % (time() - start_time), is_terminal=False)
        input = np.concatenate(input, 0)
        input = processor.reverse_normalization(input)

        prediction = np.concatenate(prediction, 0)  # (batch, N, T', 1)
        prediction = get_real_mu_bounds(prediction, processor)

        target = np.concatenate(target, 0)  # (batch, N, T)
        # 反归一化target
        target = processor.reverse_normalization(target)

        logger.write(f'input: {input.shape}\n', is_terminal=False)
        logger.write(f'prediction: {prediction.shape}\n', is_terminal=False)
        logger.write(f'target: {target.shape}\n', is_terminal=False)

        # 在forecast_path的文件名前面加入type字符串
        forecast_dir = os.path.dirname(forecast_path)
        forecast_base = os.path.basename(forecast_path)
        new_forecast_path = os.path.join(forecast_dir, f"{type}_{forecast_base}")
        with open(new_forecast_path, 'wb') as f:
            pickle.dump({'input': input, 'prediction': prediction, 'target': target}, f)
            
        # print overall results
        mae = mean_absolute_error(target.reshape(-1, 1), prediction[:, :, :, 0].reshape(-1, 1))
        rmse = mean_squared_error(target.reshape(-1, 1), prediction[:, :, :, 0].reshape(-1, 1)) ** 0.5
        mape = masked_mape_threshold_np(target.reshape(-1, 1), prediction[:, :, :, 0].reshape(-1, 1), 1, 0)
        logger.write('all MAE: %.2f\n' % (mae), is_terminal=False)
        logger.write('all RMSE: %.2f\n' % (rmse), is_terminal=False)
        logger.write('all MAPE: %.2f\n' % (mape), is_terminal=False)


def load_graphdata_normY_channel1(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, shuffle=True, percent=1.0):
    '''
    将x,y都处理成归一化到[-1,1]之前的数据;
    每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
    该函数会把hour, day, week的时间串起来；
    注： 从文件读入的数据，x,y都是归一化后的值
    :param graph_signal_matrix_filename: str
    :param num_of_hours: int
    :param num_of_days: int
    :param num_of_weeks: int
    :param DEVICE:
    :param batch_size: int
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)

    '''

    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]

    dirpath = os.path.dirname(graph_signal_matrix_filename)

    filename = os.path.join(dirpath,
                            file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks) + '.npz')

    print('load file:', filename)

    file_data = np.load(filename)
    train_x = file_data['train_x']  # (10181, 307, 3, 12)
    train_x = train_x[:, :, 0:1, :]
    train_target = file_data['train_target']  # (10181, 307, 12)
    train_timestamp = file_data['train_timestamp']  # (10181, 1)

    train_x_length = train_x.shape[0]
    scale = int(train_x_length*percent)
    print('ori length:', train_x_length, ', percent:', percent, ', scale:', scale)
    train_x = train_x[:scale]
    train_target = train_target[:scale]
    train_timestamp = train_timestamp[:scale]

    val_x = file_data['val_x']
    val_x = val_x[:, :, 0:1, :]
    val_target = file_data['val_target']
    val_timestamp = file_data['val_timestamp']

    test_x = file_data['test_x']
    test_x = test_x[:, :, 0:1, :]
    test_target = file_data['test_target']
    test_timestamp = file_data['test_timestamp']

    _max = file_data['mean']  # (1, 1, 3, 1)
    _min = file_data['std']  # (1, 1, 3, 1)

    # 统一对y进行归一化，变成[-1,1]之间的值
    train_target_norm = max_min_normalization(train_target, _max[:, :, 0, :], _min[:, :, 0, :])
    test_target_norm = max_min_normalization(test_target, _max[:, :, 0, :], _min[:, :, 0, :])
    val_target_norm = max_min_normalization(val_target, _max[:, :, 0, :], _min[:, :, 0, :])

    #  ------- train_loader -------
    train_decoder_input_start = train_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    train_decoder_input_start = np.squeeze(train_decoder_input_start, 2)  # (B,N,T(1))
    train_decoder_input = np.concatenate((train_decoder_input_start, train_target_norm[:, :, :-1]), axis=2)  # (B, N, T)

    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_decoder_input_tensor = torch.from_numpy(train_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    train_target_tensor = torch.from_numpy(train_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_decoder_input_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    #  ------- val_loader -------
    val_decoder_input_start = val_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    val_decoder_input_start = np.squeeze(val_decoder_input_start, 2)  # (B,N,T(1))
    val_decoder_input = np.concatenate((val_decoder_input_start, val_target_norm[:, :, :-1]), axis=2)  # (B, N, T)

    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_decoder_input_tensor = torch.from_numpy(val_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    val_target_tensor = torch.from_numpy(val_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_decoder_input_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    #  ------- test_loader -------
    test_decoder_input_start = test_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    test_decoder_input_start = np.squeeze(test_decoder_input_start, 2)  # (B,N,T(1))
    test_decoder_input = np.concatenate((test_decoder_input_start, test_target_norm[:, :, :-1]), axis=2)  # (B, N, T)

    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_decoder_input_tensor = torch.from_numpy(test_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    test_target_tensor = torch.from_numpy(test_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_decoder_input_tensor, test_target_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # print
    print('train:', train_x_tensor.size(), train_decoder_input_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_decoder_input_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_decoder_input_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _max, _min





class MISLossStructured(torch.nn.Module):
    """
    结构化 MIS 损失函数（融合 chunk + 动态趋势约束）
    -------------------------------------------------------
    alpha: float           - 置信区间参数（一般取 0.05 或 0.1）
    lambda_dyn: float      - 动态趋势约束系数
    -------------------------------------------------------
    模型输出3个通道：
        y_pred[..., 0] → 均值 mu
        y_pred[..., 1] → 区间宽度 raw（通过 softplus 保证 >0）
        y_pred[..., 2] → 偏移 raw（通过 tanh 限制在 [-1,1]）
    -------------------------------------------------------
    """

    def __init__(self, alpha=0.05, lambda_dyn=0.0):
        super(MISLossStructured, self).__init__()
        self.alpha = alpha
        self.lambda_dyn = lambda_dyn

    def forward(self, y_pred, y_true):
        # ======== 拆分输出通道 ========
        mu, width_raw, shift_raw = torch.chunk(y_pred, 3, dim=-1)

        # ======== 结构化区间构造 ========
        width = F.softplus(width_raw)           # 保证正值
        shift = torch.tanh(shift_raw)           # 限制在 [-1, 1]
        y_lower = mu - width * (1 + shift) / 2
        y_upper = mu + width * (1 - shift) / 2

        # ======== MIS 主体 ========
        width_eff = (y_upper - y_lower).clamp(min=1e-6)
        lower_penalty = (y_lower - y_true).clamp(min=0)
        upper_penalty = (y_true - y_upper).clamp(min=0)
        mis = width_eff + (2 / self.alpha) * (lower_penalty + upper_penalty)
        loss_mis = mis.mean()

        # ======== 动态趋势约束 ========
        if self.lambda_dyn > 0:
            dy_true = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
            dy_pred = mu[:, :, 1:, :] - mu[:, :, :-1, :]
            loss_dyn = F.l1_loss(dy_pred, dy_true)
        else:
            loss_dyn = torch.tensor(0.0, device=y_true.device)

        # ======== 综合损失 ========
        loss = loss_mis + self.lambda_dyn * loss_dyn
        return loss_dyn
    
    
class MISLossStructuredMAE(torch.nn.Module):
    """
    结构化 MIS 损失函数（融合 chunk + MAE 损失约束）
    -------------------------------------------------------
    alpha: float           - 置信区间参数（一般取 0.05 或 0.1）
    lambda_dyn: float      - MAE 约束系数
    -------------------------------------------------------
    模型输出3个通道：
        y_pred[..., 0] → 均值 mu
        y_pred[..., 1] → 区间宽度 raw（通过 softplus 保证 >0）
        y_pred[..., 2] → 偏移 raw（通过 tanh 限制在 [-1,1]）
    -------------------------------------------------------
    """

    def __init__(self, alpha=0.05, lambda_dyn=0.0):
        super(MISLossStructuredMAE, self).__init__()
        self.alpha = alpha
        self.lambda_dyn = lambda_dyn

    def forward(self, y_pred, y_true):
        # ======== 拆分输出通道 ========
        mu, width_raw, shift_raw = torch.chunk(y_pred, 3, dim=-1)

        # ======== 结构化区间构造 ========
        width = F.softplus(width_raw)           # 保证正值
        shift = torch.tanh(shift_raw)           # 限制在 [-1, 1]
        y_lower = mu - width * (1 + shift) / 2
        y_upper = mu + width * (1 - shift) / 2

        # ======== MIS 主体 ========
        width_eff = (y_upper - y_lower).clamp(min=1e-6)
        lower_penalty = (y_lower - y_true).clamp(min=0)
        upper_penalty = (y_true - y_upper).clamp(min=0)
        mis = width_eff + (2 / self.alpha) * (lower_penalty + upper_penalty)
        loss_mis = mis.mean()

        # ======== MAE 约束 ========
        if self.lambda_dyn > 0:
            loss_mae = F.l1_loss(mu, y_true)
        else:
            loss_mae = torch.tensor(0.0, device=y_true.device)

        # ======== 综合损失 ========
        loss = loss_mis + self.lambda_dyn * loss_mae
        return loss



def get_dataloader_dynoutput(dataloader, dyn_model, T_p, device):
    """
    For each batch in the dataloader, runs encoder and decoder of dyn_model to obtain dyn_output.

    Args:
        dataloader: torch.utils.data.DataLoader that yields (x_masked, x0, vars, means, ts_tensor, encoder_inputs, decoder_inputs)
        dyn_model: model with encode and decode methods
        T_p: number of prediction steps

    Returns:
        List of dyn_output tensors for each batch
    """
    mean_prediction = []
    for batch in dataloader:
        # Unpack batch to match the network input
        encoder_inputs, decoder_inputs, labels = [m.to(device, non_blocking=True) for m in batch]  # for dynamic dataset
        encoder_output = dyn_model.encode(encoder_inputs)
        decoder_start_inputs = decoder_inputs[:, :, :1, :]
        decoder_input_list = [decoder_start_inputs]
        for step in range(T_p):
            decoder_inputs_step = torch.cat(decoder_input_list, dim=2)
            predict_output = dyn_model.decode(decoder_inputs_step, encoder_output)
            decoder_input_list = [decoder_start_inputs, predict_output[:, :, :, :1]]
        mean_prediction.append(predict_output.permute(0, 2, 1, 3).contiguous())  # (B, T, V, F)
    if len(mean_prediction) > 0:
        mean_prediction = torch.cat(mean_prediction, dim=0)
    return mean_prediction.cpu().numpy()


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100
    
    
def masked_mape_threshold_np(y_true, y_pred, threshold=1, null_val=np.nan):
    """
    计算y_true大于某个阈值(如threshold)时的MAPE
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = np.logical_and(mask, y_true >= threshold)
        mask = mask.astype('float32')
        if np.mean(mask) == 0:
            return np.nan  # 没有满足条件的点
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100
