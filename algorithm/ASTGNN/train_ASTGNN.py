#!/usr/bin/env python
# coding: utf-8
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from time import time
import shutil
import argparse
import configparser
from .ASTGNN import make_model
from .utils import compute_val_loss, predict_and_save_results, dir_check
# from tensorboardX import SummaryWriter


def train_main(processor, train_loader, val_loader, test_loader, config):
    logger = config.logger
    dir_check(config.log_path)
    logger.open(config.log_path, mode="w")
    logger.write(config.__str__()+'\n', is_terminal=False)
    
    logger.write(f"model path:{config.model_path}\n", is_terminal=False)
    dir_check(config.model_path)

    logger.write(f"forecast_path:{config.forecast_path}\n", is_terminal=False)
    dir_check(config.forecast_path)
    
    # 读取参数
    DEVICE = config.device

    model_path = config.model_path
    forecast_path = config.forecast_path
    
    points_per_hour = config.points_per_hour
    num_for_predict = config.num_for_predict

    learning_rate = config.learning_rate
    epochs = config.epochs
    fine_tune_epochs = config.fine_tune_epochs
    batch_size = config.batch_size
    num_of_weeks = config.num_of_weeks
    num_of_days = config.num_of_days
    num_of_hours = config.num_of_hours
    encoder_input_size = config.encoder_input_size
    decoder_input_size = config.decoder_input_size
    dropout = config.dropout
    kernel_size = config.kernel_size
    num_layers = config.num_layers
    d_model = config.d_model
    nb_head = config.nb_head
    ScaledSAt = bool(config.ScaledSAt)
    SE = bool(config.SE)
    smooth_layer_num = config.smooth_layer_num
    aware_temporal_context = bool(config.aware_temporal_context)
    TE = bool(config.TE)
    use_LayerNorm = bool(config.use_LayerNorm)
    residual_connection = bool(config.residual_connection)
    adj_mx = config.A

    logger.write('total training epoch, fine tune epoch: {}, {}\n'.format(epochs, fine_tune_epochs), is_terminal=False)
    logger.write('batch_size: {}\n'.format(batch_size), is_terminal=False)


    net = make_model(
        DEVICE, num_layers, encoder_input_size, decoder_input_size, d_model, adj_mx, nb_head, num_of_weeks, num_of_days,
        num_of_hours, points_per_hour, num_for_predict, dropout=dropout, aware_temporal_context=aware_temporal_context,
        ScaledSAt=ScaledSAt, SE=SE, TE=TE, kernel_size=kernel_size, smooth_layer_num=smooth_layer_num,
        residual_connection=residual_connection, use_LayerNorm=use_LayerNorm
    )

    logger.write(f"{str(net)}\n", is_terminal=False)
    
    if config.load_path != None:
        logger.write('load weight from: {}\n'.format(config.load_path), is_terminal=False)
        net.load_state_dict(torch.load(config.load_path))
        predict_and_save_results(net, test_loader, processor, num_for_predict, forecast_path, 'test', logger, DEVICE)
        for param in net.parameters():
            param.requires_grad = False
        return net


    criterion = nn.L1Loss().to(DEVICE)  # 定义损失函数
    # criterion = MISLossStructuredMAE(alpha, lambda_dyn).to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # 定义优化器，传入所有网络参数
    
    total_param = 0
    logger.write("Net's state_dict:\n", is_terminal=False)
    for param_tensor in net.state_dict():
        logger.write(f"{param_tensor}\t{net.state_dict()[param_tensor].size()}\n", is_terminal=False)
        total_param += np.prod(net.state_dict()[param_tensor].size())
    logger.write("Net's total params: {}\n".format(total_param), is_terminal=False)

    logger.write("Optimizer's state_dict:\n", is_terminal=False)
    for var_name in optimizer.state_dict():
        logger.write(f"{var_name}\t{optimizer.state_dict()[var_name]}\n", is_terminal=False)

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    # train model
    start_time = time()

    for epoch in range(epochs):
        net.train()  # ensure dropout layers are in train mode
        train_start_time = time()
        for batch_index, batch_data in enumerate(train_loader):
            encoder_inputs, decoder_inputs, labels = [m.to(DEVICE, non_blocking=True) for m in batch_data]
            optimizer.zero_grad()
            outputs = net(encoder_inputs, decoder_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss = loss.item()
            global_step += 1
            
        logger.write('epoch: %s, train time every whole data:%.2fs\n' % (epoch, time() - train_start_time), is_terminal=False)
        logger.write('epoch: %s, total time:%.2fs\n' % (epoch, time() - start_time), is_terminal=False)
        
        val_loss = compute_val_loss(net, val_loader, criterion, num_for_predict, logger, DEVICE)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), model_path)
            logger.write('save parameters to file: %s\n' % model_path, is_terminal=False)

    logger.write('best epoch: {}\n'.format(best_epoch), is_terminal=False)
    logger.write('apply the best val model on the test data set ...\n', is_terminal=False)
    
    logger.write('load weight from: {}\n'.format(model_path), is_terminal=False)
    net.load_state_dict(torch.load(model_path))
    predict_and_save_results(net, test_loader, processor, num_for_predict, forecast_path, 'test', logger, DEVICE)

    # fine tune the model
    optimizer = optim.Adam(net.parameters(), lr=learning_rate*0.1)
    logger.write('fine tune the model ... \n', is_terminal=False)
    for epoch in range(epochs, epochs+fine_tune_epochs):
        net.train()  # ensure dropout layers are in train mode
        train_start_time = time()
        for batch_index, batch_data in enumerate(train_loader):
            encoder_inputs, decoder_inputs, labels  = [m.to(DEVICE, non_blocking=True) for m in batch_data]
            optimizer.zero_grad()
            encoder_output = net.encode(encoder_inputs)

            # decode
            predict_length = labels.shape[2]  # T   
            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]
            decoder_input_list = [decoder_start_inputs]
            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, predict_output[:, :, :, :1]]

            loss = criterion(predict_output, labels)
            loss.backward()
            optimizer.step()
            training_loss = loss.item()
            global_step += 1

        logger.write('epoch: %s, train time every whole data:%.2fs\n' % (epoch, time() - train_start_time), is_terminal=False)
        logger.write('epoch: %s, total time:%.2fs\n' % (epoch, time() - start_time), is_terminal=False)

        val_loss = compute_val_loss(net, val_loader, criterion, num_for_predict, logger, DEVICE)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), model_path)
            logger.write('save parameters to file: %s\n' % model_path, is_terminal=False)

    logger.write('best epoch: {}\n'.format(best_epoch), is_terminal=False)
    logger.write('apply the best val model on the test data set ...\n', is_terminal=False)
    
    logger.write('load weight from: {}\n'.format(model_path), is_terminal=False)
    net.load_state_dict(torch.load(model_path))
    predict_and_save_results(net, test_loader, processor, num_for_predict, forecast_path, 'test', logger, DEVICE)
    for param in net.parameters():
        param.requires_grad = False
    return net



if __name__ == "__main__":

    train_main()

    # predict_main(0, test_loader, test_target_tensor, _max, _min, 'test')

















