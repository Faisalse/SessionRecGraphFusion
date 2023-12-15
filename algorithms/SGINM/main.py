# -*- coding: utf-8 -*-

import pickle
import torch
import datetime
import time
import argparse
import numpy as np
import os
import random

from data import *
from model import *


def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda: 0" if USE_CUDA else "cpu")

def train_test(model, tra_data, tes_data, opt):
    print('start training: ', datetime.datetime.now())
    model.train()
    model.scheduler.step()
    total_loss = []
    # function to generate the batches.....
    slices = tra_data.generate_batch_slices(opt.batch_size)
    
   
    for i in range(len(slices)):
        inp_var, lengths, mask0, mask1, maskinf, out_var = tra_data.batch2TrainData(slices[i])
        inp_var = inp_var.to(device)
        lengths = lengths.to(device)
        mask0 = mask0.to(device)
        mask1 = mask1.to(device)
        maskinf = maskinf.to(device)
        out_var = out_var.to(device)

        model.optimizer.zero_grad()

        score = model(inp_var, lengths, mask0, mask1, maskinf)
        loss = model.loss_function(score, out_var - 1)
        loss.backward()
        model.optimizer.step()
        total_loss.append(loss.item())
    print('Loss: %0.4f\tlr: %0.8f' % (np.mean(total_loss), model.optimizer.param_groups[0]['lr']))

    print('start predicting: ', datetime.datetime.now())
    
    
    
    
    
    
    model.eval()
    hit_dic, mrr_dic = {}, {}
    for k in opt.k:
        hit_dic[k] = []
        mrr_dic[k] = []
    slices = tes_data.generate_batch_slices(opt.batch_size)
    for i in range(len(slices)):
        te_inp_var, te_lengths, te_mask0, te_mask1, te_maskinf, te_out_var = tes_data.batch2TrainData(slices[i])
        te_inp_var = te_inp_var.to(device)
        te_lengths = te_lengths.to(device)
        te_mask0 = te_mask0.to(device)
        te_mask1 = te_mask1.to(device)
        te_maskinf = te_maskinf.to(device)
        test_score = model(te_inp_var, te_lengths, te_mask0, te_mask1, te_maskinf)

        for k in opt.k:
            predict = test_score.topk(k)[1]
            print("predict   ", predict.shape)
            
            predict = predict.cpu()
            for pred, target in zip(predict, te_out_var):
                
                hit_dic[k].append(np.isin(target - 1, pred))
                if len(np.where(pred == target - 1)[0]) == 0:
                    mrr_dic[k].append(0)
                else:
                    mrr_dic[k].append(1 / (np.where(pred == target - 1)[0][0] + 1))


    for k in opt.k:
        hit_dic[k] = np.mean(hit_dic[k]) * 100
        mrr_dic[k] = np.mean(mrr_dic[k]) * 100
    # hit = np.mean(hit) * 100
    # mrr = np.mean(mrr) * 100
    return hit_dic, mrr_dic

def train_epochs(opt):
    dataset = opt.dataset
    train_data = pickle.load(open('./dataset/' + dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('./dataset/' + dataset + '/test.txt', 'rb'))

    # # if vcf data
    # if opt.is_vcf:
    #     print('before vcf len(train_data): %d' % (len(train_data)))
    #     train_data = plus_vcf_data(train_data, num=opt.num)
    #     print('after vcf len(train_data): %d' % (len(train_data)))

    if opt.short_or_long == 0:
        all_data = train_data + test_data
    else:
        print("Hello")
        tra_short, tra_long = split_short_long(train_data, thred=5)
        tes_short, tes_long = split_short_long(test_data, thred=5)
        if opt.short_or_long == 1:
            all_data = tra_short + tes_short
            train_data = tra_short
            test_data = tes_short
        else:
            all_data = tra_long + tes_long
            train_data = tra_long
            test_data = tes_long
            
            
            

    all_data = Data(all_data, status='all')  # 用来统计训练集和测试集一共的word2index, index2word, num_words
    train_data = Data(train_data, status='train')
    test_data = Data(test_data, status='test')

    train_data.word2index, train_data.index2word, train_data.num_words = all_data.word2index, all_data.index2word, all_data.num_words
    test_data.word2index, test_data.index2word, test_data.num_words = all_data.word2index, all_data.index2word, all_data.num_words

    model = SGINM(opt, train_data.num_words)

    if opt.pretrain is not None:
        checkpoint = torch.load(opt.pretrain)
        model.load_state_dict(checkpoint['model'])

    model = model.to(device)
    print(model)

    epochs = opt.epochs
    best_result = {}
    best_epoch = {}
    for k in opt.k:
        best_result[k] = [0, 0]
        best_epoch[k] = [0, 0]
    t0 = time.time()
    for epoch in range(epochs):
        st = time.time()
        print('------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data, opt)
        for k in opt.k:
            if hit[k] > best_result[k][0]:
                best_result[k][0] = hit[k]
                best_epoch[k][0] = epoch
            if mrr[k] > best_result[k][1]:
                best_result[k][1] = mrr[k]
                best_epoch[k][1] = epoch
            print('Hit@%d: %0.4f %%\tMRR@%d: %0.4f %%\t[%0.2f]' % (k, hit[k], k, mrr[k], (time.time() - st)))
    print('------------------best result-------------------')
    for k in opt.k:
        print('Best Result: Hit@%d: %0.4f %%\tMRR@%d: %0.4f %%\t[%0.2f]' %
              (k, best_result[k][0], k, best_result[k][1], (time.time() - t0)))
        print('Best Epoch: Hit@%d: %d\tMRR@%d: %d\t[%0.2f]' % (k, best_epoch[k][0], k, best_epoch[k][1], (time.time() - t0)))
    print('------------------------------------------------')
    print('saving model: ', datetime.datetime.now())
    now = time.strftime('%Y%m%d_%H%M%S', time.localtime(int(time.time())))
    model_name = now + '_' + dataset + '.tar'
    model_name = os.path.join('./model_save', dataset, model_name)
    # torch.save({
    #     'model': model.state_dict(),
    #     'opt': model.optimizer,
    #     'loss_func': model.loss_function,
    #     'best_result': best_result,
    #     'best_epoch': best_epoch
    # }, model_name)
    print('Run time: %0.2f s' % (time.time() - t0))

def main():
    init_seed(2021)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='diginetica', help='dataset name: yoochoose1_64/diginetica/yoochoose1_4.')
    parser.add_argument('--batch_size', default=512, help='Input data batch.')
    parser.add_argument('--short_or_long', default=0, help='0 is all dataset, 1 is short, 2 is long.')
    parser.add_argument('--k', default=[20], help='top k recommendation.')
    parser.add_argument('--embedding_size', default=50, help='Item embedding dimension.')
    parser.add_argument('--hidden_size', default=100, help='RNN hidden state size')
    parser.add_argument('--out_size', default=32, help='model out size')
    parser.add_argument('--n_layers', default=1, help='Number of RNN layers.')
    parser.add_argument('--h', default=5, help='head num of attn.')
    parser.add_argument('--dropout', default=0.1, help='Dropout.')
    parser.add_argument('--epochs', default=20, help='Number of train epoch.')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2-penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    parser.add_argument('--pretrain', default=None, help='pretrain file path')
    parser.add_argument('--lr_dc', type=list, default=[8, 10], help='lr decay')  # [8, 10] for yoo, [5, 7] for dig
    # parser.add_argument('--is_vcf', type=bool, default=False, help='if add virtual data.')
    # parser.add_argument('--num', type=int, default=0, help='if add virtual data.')
    opt = parser.parse_args()
    print(opt)
    train_epochs(opt)


if __name__ == '__main__':
    main()

