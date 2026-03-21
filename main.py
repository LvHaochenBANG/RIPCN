import os
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from utils.common_utils import get_workspace, dir_check, Logger, load_graphdata_normY_channel1
from algorithm.nppc_trainer import NPPCTrainer
from algorithm.RIPCN.RGnet import RGnet
from dataset import TrafficDataset, get_road_feature
from algorithm.preprocess import WindowProcessorWithStats
from algorithm.ASTGNN.train_ASTGNN import train_main as ASTGNN_train
from algorithm.ASTGNN.utils import get_dataloader_dynoutput as get_dynoutput

ws =  get_workspace()

def get_params():
    parser = argparse.ArgumentParser(description='Entry point of the code')
    parser.add_argument("--train_mean", type=bool, default=True)
    parser.add_argument("--train_nppc", type=bool, default=True)
    parser.add_argument("--load_nppc", type=bool, default=False)
    parser.add_argument("--T_h", type=int, default=12)  # history window
    parser.add_argument("--T_p", type=int, default=12)  # prediction window
    parser.add_argument("--cuda", type=int, default=1)
    parser.add_argument("--data", type=str, default='PEMS08')  # 还要改clean_dataset!!!
    parser.add_argument("--lr", type=float, default=1e-4) # 0.005
    parser.add_argument("--batch_size", type=int, default=12)  # Seattle PEMS04 12,PEMS08 32
    parser.add_argument("--n_dirs", type=int, default=3)  # 1,3,5,7
    parser.add_argument("--hidden_size", type=int, default=32)  # Seattle 128, other 32
    parser.add_argument("--epoch", type=int, default=200)  # 200
    
    args, _ = parser.parse_known_args()
    return args


def default_config(params):
    config = edict()
    config.data = params['data']
    config.key = time.strftime('%m-%d %H:%M', time.localtime())
    config.PATH_MOD = ws + f'/output_{config.key}/model/' 
    config.PATH_LOG = ws + f'/output_{config.key}/log/'
    config.PATH_FORECAST = ws + f'/output_{config.key}/forecast/'
    config.PATH_METRIC = ws + f'/output_{config.key}/metrics/'
    config.trial_name = '+'.join([f"{v}" for k, v in params.items()])
    config.log_path = f"{config.PATH_LOG}/{config.trial_name}.log"

    config.device = torch.device(f"cuda:{params['cuda']}" if torch.cuda.is_available() else 'cpu')
    config.T_h = params['T_h']
    config.T_p = params['T_p']
    config.early_stop = 20
    config.start_epoch = 100  # stat to eval 20
    config.n_dirs = params['n_dirs']
    config.lr = params['lr']
    config.batch_size = params['batch_size']
    
    config.nppc_epoch = params['epoch']
    config.PATH_BASE = os.path.join(ws, f'NPPC_{config.key}')
    config.PATH_NPPC = config.PATH_BASE + '/npc'
    config.PATH_MODEL = config.PATH_BASE + '/model'
    config.log_path = f"{config.PATH_BASE}/n_dirs={config.n_dirs}-epoch={config.nppc_epoch}.log"
    config.logger = Logger()
    
    # RIPCN model config
    config.model = edict()
    config.model.T_h = params['T_h']
    config.model.T_p =  params['T_p']
    config.model.T = params['T_h'] + params['T_p']
    config.model.d_h = params['hidden_size']
    config.model.n_channels = params['hidden_size']
    config.model.channel_multipliers = [1, 2]
    config.model.device = config.device
    config.model.n_dirs = params['n_dirs']
    config.model.for_NPPC = True
    config.model.final_channel = params['n_dirs']
    
    config.dyn_model = edict(
        device=config.device,
        log_path = os.path.join(config.PATH_BASE, f"dyn.log"),
        model_path = os.path.join(config.PATH_BASE, 'dyn_model.pt'),
        forecast_path = os.path.join(config.PATH_BASE, 'dyn_forecast.pkl'),
        points_per_hour=12,
        num_for_predict=params['T_p'],
        len_input=params['T_h'],
        dataset_name=config.data,
        learning_rate=0.001,
        epochs=40,  # 40
        fine_tune_epochs=20,  # 20
        batch_size=params['batch_size'],
        num_of_weeks=0,
        num_of_days=0,
        num_of_hours=1,
        encoder_input_size=1,  # num_features
        decoder_input_size=1,  # num_features
        dropout=0.0,
        kernel_size=3,
        num_layers=4,
        d_model=64,
        nb_head=8,
        ScaledSAt=1,
        SE=1,
        smooth_layer_num=1,
        aware_temporal_context=1,
        TE=1,
        use_LayerNorm=1,
        residual_connection=1,
        logger = Logger(),
        load_path = None
    )
    return config

    

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
if __name__ == '__main__':
    params = vars(get_params())
    setup_seed(2025)
    torch.set_num_threads(1)
    config = default_config(params)
    
    if params['train_mean']:
         # ==== 初始化数据处理类 ====
        processor = WindowProcessorWithStats(
            name = config.data,
            data_path=f'./data/dataset/{config.data}/flow.npy',
            history_window=config.T_h,   # 12个时间步历史数据
            future_window=config.T_p,    # 12个时间步预测目标
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            device=config.device
        )
        config.dyn_model.A = np.load(f'./data/dataset/{config.data}/adj.npy')
        
        # ==== 获取动态特征类的数据集 ====
        train_dataset_dynamic = processor.get_dynamic_dataset('train')
        val_dataset_dynamic = processor.get_dynamic_dataset('val')
        test_dataset_dynamic = processor.get_dynamic_dataset('test')

        # ==== 构造用于ASTGNN训练的数据加载器 ====
        train_loader_dynamic = DataLoader(train_dataset_dynamic, batch_size=32, shuffle=True)
        val_loader_dynamic = DataLoader(val_dataset_dynamic, batch_size=32, shuffle=False)
        test_loader_dynamic = DataLoader(test_dataset_dynamic, batch_size=32, shuffle=False)
        
        # ==== 训练ASTGNN模型 ====
        dyn_model = ASTGNN_train(processor, train_loader_dynamic, val_loader_dynamic, test_loader_dynamic, config.dyn_model)
        
        train_loader_dynamic = DataLoader(train_dataset_dynamic, batch_size=32, shuffle=False)
        dyn_output_train = get_dynoutput(train_loader_dynamic, dyn_model, config.T_p, config.device)  # (B, T, V, F)
        dyn_output_val = get_dynoutput(val_loader_dynamic, dyn_model, config.T_p, config.device)
        dyn_output_test = get_dynoutput(test_loader_dynamic, dyn_model, config.T_p, config.device)
        
        # 反归一化并保存训练、验证和测试数据
        for split_name, data, targets, output in [
            ('train_', processor.train_data[...,None], processor.train_targets[...,None], dyn_output_train),
            ('eval_',  processor.val_data[...,None],    processor.val_targets[...,None],   dyn_output_val),
            ('test_',  processor.test_data[...,None],   processor.test_targets[...,None],  dyn_output_test)
        ]:
            input_denorm = processor.reverse_normalization(data)
            target_denorm = processor.reverse_normalization(targets)
            mean_prediction_denorm = processor.reverse_normalization(output)

            np.savez(
                f'./data/dataset/{config.data}/{split_name}.npz',
                input=input_denorm,
                target=target_denorm,
                mean_prediction=mean_prediction_denorm
            )
        
        
    if params['train_nppc']:
        if not os.path.exists(config.PATH_NPPC):
            os.makedirs(config.PATH_NPPC)
        if not os.path.exists(config.PATH_MODEL):
            os.makedirs(config.PATH_MODEL)

        dir_check(config.log_path)
        config.logger.open(config.log_path, mode="w")
        config.logger.write(config.__str__()+'\n', is_terminal=False)  # log parameters
    
    
    dataset = TrafficDataset("./data/dataset", params['data'], device=config.device)
    config.model.A = dataset.adj
    config.model.V = dataset.num_vertices
    config.model.F = dataset.num_features
    config.model.cap, config.model.max_speed, config.model.ttu = \
        get_road_feature(params['data'])  # load road feature
    
    train_dataset, val_dataset, test_dataset = dataset.get_datasets()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size)
        
    # RIPCN
    if params['train_nppc'] or params['load_nppc']:
        nppc_model = RGnet(config.model)
        nppc_model.to(config.device)
        nppc_trainer = NPPCTrainer(dataset, train_loader, val_loader, test_loader, nppc_model, config)
        if params['load_nppc']:
            nppc_path = ''
            nppc_trainer.load_model(nppc_path)
        if params['train_nppc']:
            nppc_trainer.train()
            nppc_trainer.load_model()
            nppc_trainer.test()
        else:
            nppc_trainer.test()
            