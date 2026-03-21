# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.sparse as sp

import os
import torch
def get_workspace():
    """
    get the workspace path, i.e., the root directory of the project
    """
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    return file
ws =  get_workspace()


def gather(consts: torch.Tensor, t: torch.Tensor):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

def dir_check(path):
    """
    check weather the dir of the given path exists, if not, then create it
    """
    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)
    return path

def to_device(batch, device):
    batch = [x.to(device) for x in batch]
    return batch


# merge all the dict in the list
def dict_merge(dict_list = []):
    dict_ =  {}
    for dic in dict_list:
        assert isinstance(dic, dict), "object is not a dict!"
        dict_ = {**dict_, **dic}
    return dict_

def unfold_dict(in_dict):
    from easydict import EasyDict as edict
    # convert 2 level easydict to 1 level, mainly for record the results.
    out_dict = {}
    for k1, v1 in in_dict.items():
        if isinstance(v1, edict) or isinstance(v1, dict):
            for k2, v2 in v1.items():
                out_dict[f'{k1}.{k2}'] = v2
        else:
            out_dict[k1] = v1
    return out_dict


def save2file_meta(params, file_name, head):
    def timestamp2str(stamp):
        utc_t = int(stamp)
        utc_h = utc_t // 3600
        utc_m = (utc_t // 60) - utc_h * 60
        utc_s = utc_t % 60
        hour = (utc_h + 8) % 24
        t = f'{hour}:{utc_m}:{utc_s}'
        return t
    import csv, time, os
    dir_check(file_name)
    if not os.path.exists(file_name):
        f = open(file_name, "w", newline='\n', encoding='utf-8')
        csv_file = csv.writer(f)
        csv_file.writerow(head)
        f.close()

    df = pd.read_csv(file_name, encoding='utf-8')
    old_head = df.columns
    if len(set(head)) > len(set(old_head)):
        f = open(file_name, "w", newline='\n')
        csv_file = csv.writer(f)
        csv_file.writerow(head) # write new head
        for idx, data_df in df.iterrows():
            data = [data_df[k] if k in old_head else -1 for k in head]
            csv_file.writerow(data)
        f.close()

    with open(file_name, "a", newline='\n', encoding='utf-8') as file:
        csv_file = csv.writer(file)
        params['log_time'] = timestamp2str(time.time())
        data = [params[k] for k in head]
        csv_file.writerow(data)


def GpuId2CudaId(gpu_id):

    # return {0:8, 1:4, 2:0, 3:5, 4:1, 5:6, 6:9, 7:7, 8:2, 9:3}[gpu_id]# for server in NUS, where the gpu id does not equal the cuda id in the server.
    return {i:i for i in range(8)}.get(gpu_id, 0)



# print logger
class Logger(object):
    def __init__(self):
        import sys
        self.terminal = sys.stdout  #stdout
        self.file = None
        self.message_buffer = ''

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=True, is_file=True):
        if '\r' in message: is_file=False

        if is_terminal:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file:
            self.file.write(message)
            self.file.flush()

    def write_message_buffer(self):
        self.write(self.message_buffer, is_terminal=False)
        self.message_buffer = ''


def shape_correct(x: torch.Tensor, expected_shape: list):
    # automatic convert a tensor to the expected_shape
    # e.g., x: (B, T, V, F) --> (B, V, T, F)
    dim4idx = {d: i for i, d in enumerate(x.shape)}
    assert len(expected_shape) == len(dim4idx), "length of expected shape does not equal to the input shape"
    permute_idx = [dim4idx[d] for d in expected_shape]
    x = x.permute(tuple(permute_idx))
    return x



from multiprocessing import Pool
def multi_thread_work(parameter_queue,function_name,thread_number=5):
    pool = Pool(thread_number)
    result = pool.map(function_name, parameter_queue)
    pool.close()
    pool.join()
    return  result

import matplotlib.pyplot as plt
def draw_predicted_distribution(samples, target, observed_flag, evaluate_flag, config={}):

    """
    All should be torch.Tensor
    :param samples: (B, n_samples, T, V, F)
    :param label: (B, T, V, F)
    :param observed_flag: (B, T, V, F), equals 1 if the data is observed in the data
    :param evaluate_flag: (B, T, V, F), equals 1 if the data if we want to draw the distribution of this data
    :return:
    """

    def get_quantile(samples, q, dim=1):
        return torch.quantile(samples, q, dim=dim).cpu().numpy()

    B, n_samples, T, V, F = samples.shape
    print('node number:', V)
    # take out the last feature
    samples = samples[:, :, :, :, 0]
    all_target_np = target[:, :, :, 0].cpu().numpy()
    all_observed_np = observed_flag[:, :, :, 0].cpu().numpy()
    all_evalpoint_np = evaluate_flag[:, :, :, 0].cpu().numpy()

    all_given_np = all_observed_np - all_evalpoint_np

    qlist = [0.05, 0.25, 0.5, 0.75, 0.95]
    quantiles_imp = []
    for q in qlist:
        quantiles_imp.append(get_quantile(samples, q, dim=1) * (1 - all_given_np) + all_target_np * all_given_np)

    dataind = config.get('dataind', 10)  # change to visualize a different sample

    plt.rcParams["font.size"] = 16
    fig, axes = config.get('fig_axes', (None, None))
    w,h = 6.8, 4.2# 6, 4.5
    # nrow, ncol = 9, 4
    nrow, ncol = 2, 4
    if fig is None:
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(ncol * w, nrow * h ))
        print('create fig')
    # fig.delaxes(axes[-1][-1])
    V_lst = config.get('V_lst', None)
    if V_lst is None:
        V = min([V,  nrow * ncol])
        V_lst = list(range(V))
    for v_idx, v in enumerate(V_lst):
        df = pd.DataFrame({"x": np.arange(0, T), "val": all_target_np[dataind, :, v], "y": all_evalpoint_np[dataind, :, v]})
        df = df[df.y != 0]
        df2 = pd.DataFrame({"x": np.arange(0, T), "val": all_target_np[dataind, :, v], "y": all_given_np[dataind, :, v]})
        df2 = df2[df2.y != 0]
        row = v_idx // ncol
        col = v_idx % ncol

        color, linestyle, label, alpha = config.get('color', 'g'), config.get('linestyle', 'solid'), config.get('label', 'label'), config.get('alpha', 0.3)
        axes[row][col].fill_between(range(0, T), quantiles_imp[0][dataind, :, v], quantiles_imp[4][dataind, :, v], color=color, alpha=alpha, label=label + "")# 90% interval


        #axes[row][col].fill_between(range(0, T), quantiles_imp[0][dataind, :, v]+1, quantiles_imp[4][dataind, :, v]-1, color=color, alpha=alpha+0.4, label=label)
        observations_label = config.get('observations_label', None)
        axes[row][col].plot(df.x, df.val, color='b', marker='o', linestyle='None', label=observations_label)
        axes[row][col].plot(df2.x, df2.val, color='r', marker='x', linestyle='None')
        axes[row][col].set_title(f'node:{v}')

        if col == 0:
            plt.setp(axes[row, 0], ylabel='PM 2.5')
        # if row == -1:
        #     plt.setp(axes[-1, col], xlabel='time')
        if row == -1:
            plt.setp(axes[-1, col], xlabel='time')
    # plt.legend()
    # plt.show()
    axes[0][0].legend()
    # fig.tight_layout()
    # plt.subplots_adjust(wspace=0, hspace=1)
    return fig, axes





# def max_min_normalization(x, _max, _min):
#     x = 1. * (x - _min)/(_max - _min)
#     x = x * 2. - 1.
#     return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


def max_min_normalization(x, _max, _min):
    mean = 230.26390847345718
    std = 145.5800822595608
    return (x - mean) / std


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

    print(_max[:, :, 0, :], _min[:, :, 0, :])

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
    train_target_tensor = torch.from_numpy(train_target_norm).type(torch.FloatTensor)[:,:,None,:].to(DEVICE)  # (B, N, T)
    pw = torch.zeros((train_x_length, 1)).to(DEVICE)
    pt = torch.zeros((train_x_length, 1)).to(DEVICE)

    train_dataset = torch.utils.data.TensorDataset(train_target_tensor.permute(0, 3, 1, 2), train_x_tensor.permute(0, 3, 1, 2), pw, pt)

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    #  ------- val_loader -------
    val_decoder_input_start = val_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    val_decoder_input_start = np.squeeze(val_decoder_input_start, 2)  # (B,N,T(1))
    val_decoder_input = np.concatenate((val_decoder_input_start, val_target_norm[:, :, :-1]), axis=2)  # (B, N, T)

    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_decoder_input_tensor = torch.from_numpy(val_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    val_target_tensor = torch.from_numpy(val_target_norm).type(torch.FloatTensor)[:,:,None,:].to(DEVICE)  # (B, N, T)
    pw = torch.zeros((val_x_tensor.shape[0], 1)).to(DEVICE)
    pt = torch.zeros((val_x_tensor.shape[0], 1)).to(DEVICE)

    val_dataset = torch.utils.data.TensorDataset(val_target_tensor.permute(0, 3, 1, 2), val_x_tensor.permute(0, 3, 1, 2), pw, pt)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    #  ------- test_loader -------
    test_decoder_input_start = test_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    test_decoder_input_start = np.squeeze(test_decoder_input_start, 2)  # (B,N,T(1))
    test_decoder_input = np.concatenate((test_decoder_input_start, test_target_norm[:, :, :-1]), axis=2)  # (B, N, T)

    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_decoder_input_tensor = torch.from_numpy(test_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    test_target_tensor = torch.from_numpy(test_target_norm).type(torch.FloatTensor)[:,:,None,:].to(DEVICE)  # (B, N, T)
    pw = torch.zeros((test_x_tensor.shape[0], 1)).to(DEVICE)
    pt = torch.zeros((test_x_tensor.shape[0], 1)).to(DEVICE)

    test_dataset = torch.utils.data.TensorDataset(test_target_tensor.permute(0, 3, 1, 2), test_x_tensor.permute(0, 3, 1, 2), pw, pt)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # print
    print('train:', train_x_tensor.size(), train_decoder_input_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_decoder_input_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_decoder_input_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _max, _min






