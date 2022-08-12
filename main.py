import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
from model import *
import pandas as pd
import itertools
import torch
import random
import numpy as np
import os


def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


seed_torch(42)

parser = argparse.ArgumentParser()
parser.add_argument('--base_path', default='./datasets', help='root')
parser.add_argument('--dataset', default='gowalla', help='dataset name: gowalla/lastfm/yoochoose1_64')
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=256, help='hidden state size')
parser.add_argument('--epoch', type=int, default=1, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.3, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=5, help='the number of epoch to wait before early stop ')
parser.add_argument('--gat_head', type=int, default=8, help='gat head')
parser.add_argument('--transformer_head', type=int, default=8, help='transformer head')
parser.add_argument('--order_list', type=list, default=[1,2,3], help='order_list')
parser.add_argument('--memory_size', type=int, default=40, help='memory_size')
opt = parser.parse_args()
print(opt)
def main():

    with open('%s/%s/train.pkl'%(opt.base_path, opt.dataset), 'rb') as f1:
        train_data = pickle.load(f1)
    with open('%s/%s/test.pkl'%(opt.base_path, opt.dataset), 'rb') as f2:
        test_data = pickle.load(f2)
    with open('%s/%s/num_items.txt'%(opt.base_path, opt.dataset)) as f3:
        num_items = int(f3.read())
    print('train_data num:', len(train_data[1]))
    print('test_data num:', len(test_data[1]))

    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    n_node = num_items + 1

    model = trans_to_cuda(SessionGraph(opt, n_node))
    state_dict = torch.load('./weights/%s_bestHR.pth'%(opt.dataset)) # map_location=torch.device('cpu')
    model.load_state_dict(state_dict['model'])

    start = time.time()
    hit, mrr, model = train_test(opt, model, train_data, test_data)
    results = 'Recall@20: %.4f, MRR@20: %.4f' % (hit, mrr)
    print('Final result: %s' % (results))
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()