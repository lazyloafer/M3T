# -*- coding: utf-8 -*-
import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F


class GATcell(Module):
    def __init__(self, opt):
        super(GATcell, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.att_hidden_size = int(opt.hiddenSize / opt.gat_head)
        self.linear_head_in = nn.Linear(self.hidden_size, self.att_hidden_size, bias=False)
        self.linear_head_out = nn.Linear(self.hidden_size, self.att_hidden_size, bias=False)
        self.linear_edge_in = nn.Linear(self.hidden_size, self.att_hidden_size, bias=False)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.att_hidden_size, bias=False)
        # self.a_in = Parameter(torch.Tensor(self.hidden_size*2, 1))
        # self.a_out = Parameter(torch.Tensor(self.hidden_size*2, 1))
        self.a_in = nn.Linear(self.att_hidden_size, 1, bias=False)
        self.a_out = nn.Linear(self.att_hidden_size, 1, bias=False)
        self.bias_iah = Parameter(torch.Tensor(self.att_hidden_size))
        self.bias_oah = Parameter(torch.Tensor(self.att_hidden_size))
        # self.linear_tail = nn.Linear(self.hidden_size*2, self.hidden_size, bias=True)
        self.drop = nn.Dropout(0.1)
        self.leakyRelu = nn.LeakyReLU(0.2)
        # self.PRelu = torch.nn.PReLU(self.hidden_size)

    def get_AttScore(self, input_, a):
        node_num = input_.size(1)
        q = input_.repeat(1, 1, node_num).view(input_.size(0), node_num * node_num, -1) * input_.repeat(1, node_num, 1)
        att_score = self.drop(self.leakyRelu(a(q))).view(input_.size(0), node_num, node_num)
        return att_score

    def forward(self, A_in_0, A_out_0, A_I, order, input_in, input_out):
        # print(input_in[0,:,:])
        input_in_att = self.linear_head_in(input_in)
        input_out_att = self.linear_head_out(input_out)
        att_score_in = self.get_AttScore(input_in_att, self.a_in)
        att_score_out = self.get_AttScore(input_out_att, self.a_out)

        A_in_temp = A_in_0
        A_out_temp = A_out_0
        A_in = A_in_0
        A_out = A_out_0

        for i in range(order - 1):
            A_in_temp = torch.matmul(A_in_temp, A_in_0)
            A_out_temp = torch.matmul(A_out_temp, A_out_0)
            A_in = A_in + A_in_temp
            A_out = A_out + A_out_temp

        ones_in = torch.ones_like(A_in)
        ones_out = torch.ones_like(A_out)

        A_in = torch.where(A_in > 0, ones_in, A_in)
        A_out = torch.where(A_out > 0, ones_out, A_out)

        zeros_in = -1e12 * torch.ones_like(att_score_in)
        zeros_out = -1e12 * torch.ones_like(att_score_out)

        A_in = torch.softmax(torch.where(A_in > 0, att_score_in, zeros_in), dim=-1) * A_in
        A_out = torch.softmax(torch.where(A_out > 0, att_score_out, zeros_out), dim=-1) * A_out

        output_in = self.linear_edge_in(torch.matmul(A_in, input_in)) + self.bias_iah
        output_out = self.linear_edge_out(torch.matmul(A_out, input_out)) + self.bias_oah

        return output_in, output_out


class GAT(Module):
    def __init__(self, opt):
        super(GAT, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.head = opt.gat_head
        self.GAT_set = nn.ModuleList()
        for i in range(self.head):
            self.GAT_set.append(GATcell(opt))
        self.leakyRelu = nn.LeakyReLU(0.2)

    def forward(self, A_in_0, A_out_0, A_I, order, input_in, input_out):
        outputs_in_multi = []
        outputs_out_multi = []
        for i in range(self.head):
            outputs_in, outputs_out = self.GAT_set[i](A_in_0, A_out_0, A_I, order, input_in, input_out)
            if self.head == 1:
                return self.leakyRelu(outputs_in + input_in), self.leakyRelu(outputs_out + input_out)
            else:  # multi head
                outputs_in_multi.append(outputs_in)
                outputs_out_multi.append(outputs_out)
        return self.leakyRelu(torch.cat(outputs_in_multi, dim=-1) + input_in), self.leakyRelu(
            torch.cat(outputs_out_multi, dim=-1) + input_out)


class GNN(Module):
    def __init__(self, opt):
        super(GNN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.step = opt.step
        self.hidden_size = opt.hiddenSize
        self.head = opt.gat_head
        self.order_list = opt.order_list

        self.gnn_set = nn.ModuleList()
        for i in range(len(self.order_list)):
            # self.gnn_set.append(GCNcell(opt))
            self.gnn_set.append(GAT(opt))

        self.feat_drop = nn.Dropout(0.5)

    def forward(self, A, A_I, hidden, alias_inputs, mask, item_flag):
        hidden = self.feat_drop(hidden)
        A_in_1 = A[:, :, :A.shape[1]]
        A_out_1 = A[:, :, A.shape[1]:]
        A_I = torch.eye(A_in_1.size(1)).unsqueeze(0).to(self.device)

        # new_hidden_0, _ = self.GRU_0(hidden, alias_inputs, mask)
        # hidden_in_0 = new_hidden_0[:, :, :self.hidden_size] + hidden
        # hidden_out_0 = new_hidden_0[:, :, self.hidden_size:] + hidden
        # hidden_0 = new_hidden_0 + hidden

        out_set = []
        for i in range(len(self.order_list)):
            output_in, output_out = self.gnn_set[i](A_in_1, A_out_1, A_I, self.order_list[i], hidden, hidden)
            # out = self.gate_set[i](output_in, output_out, hidden)
            out = output_in + output_out
            out_set.append(out)
        return out_set  # [order, batch, node, hidden_size]


class PositionalEncoding(Module):
    def __init__(self, d_hid, mask):
        super(PositionalEncoding, self).__init__()
        self.mask = mask
        self.d_hid = d_hid
        self.n_position = self.mask.size(1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def pos_emb(self, n_position, d_hid):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        pos_emb = torch.FloatTensor(sinusoid_table)
        return pos_emb.to(self.device)

    def forward(self, x, pos=None):
        if pos == None:
            last_item_idx = torch.sum(self.mask, 1).view(-1, 1) - 1
            flag = torch.arange(self.mask.size(1), device=self.device, dtype=torch.int32).unsqueeze(0)
            flag = torch.abs(flag.repeat(self.mask.size(0), 1) - last_item_idx)
            # return torch.cat([x, self.pos_emb(self.n_position, self.d_hid)[flag]], dim=-1)
            return x + self.pos_emb(self.n_position, self.d_hid)[flag]
        else:
            # return torch.cat([x, self.pos_emb(self.n_position, self.d_hid)[0].unsqueeze(0).repeat(x.size(0), 1)], dim=-1)
            return x + self.pos_emb(self.n_position, self.d_hid)[pos].unsqueeze(0).repeat(x.size(0), 1)


class Self_Attention(Module):
    def __init__(self, opt):
        super(Self_Attention, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.head = opt.transformer_head
        self.K = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.Q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, inputs, mask, pos_emb=False):
        if pos_emb == True:
            get_posEmb = PositionalEncoding(self.hidden_size, mask)
            seq_Q = get_posEmb(self.Q(inputs))
            seq_K = get_posEmb(self.K(inputs))
            seq_V = get_posEmb(self.V(inputs))
        else:
            seq_Q = self.Q(inputs)
            seq_K = self.K(inputs)
            seq_V = self.V(inputs)

        return seq_Q, seq_K, seq_V  # [batch, head, seq, hidden/head]


class ScoerModel(Module):
    def __init__(self, opt, embedding_1, embedding_2):
        super(ScoerModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mem_size =opt.memory_size
        self.hidden_size = opt.hiddenSize
        self.head = opt.transformer_head
        self.embedding_1 = embedding_1
        self.embedding_2 = embedding_2
        self.GRU = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=2,
                          batch_first=True, bidirectional=False)
        self.linear_one_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_one_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three_1 = nn.Linear(self.hidden_size * 2, 1, bias=False)
        self.linear_three_2 = nn.Linear(self.hidden_size * 2, 1, bias=False)
        self.linear_transform_1 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_transform_2 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_tail_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_tail_2 = nn.Linear(self.hidden_size, 2, bias=False)

        self.self_att_1 = Self_Attention(opt)
        self.self_att_2 = Self_Attention(opt)

        self.FFN_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.FFN_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.memory_k_embedding = nn.Parameter(torch.FloatTensor(1, self.mem_size, self.hidden_size))
        self.memory_v_embedding = nn.Parameter(torch.FloatTensor(1, self.mem_size, self.hidden_size))

    def co_attention(self, seq_Q, seq_K, seq_V, mask, memory=False):  # seq_Q [batch, head, seq, hidden/head]

        if memory:
            m_k = self.memory_k_embedding.repeat(seq_K.size(0), 1, 1)  # batch, mem_size, hidden
            m_v = self.memory_v_embedding.repeat(seq_V.size(0), 1, 1)  # batch, mem_size, hidden

            seq_K = torch.cat([m_k, seq_K], dim=1)  # batch, mem_size+seq, hidden
            seq_V = torch.cat([m_v, seq_V], dim=1)  # batch, mem_size+seq, hidden

            att = torch.matmul(seq_Q, seq_K.transpose(2, 1)) / np.sqrt(self.hidden_size)  # batch, seq, mem_size+seq
            # att = torch.matmul(seq_Q, seq_K.transpose(3, 2)) / np.sqrt(self.hidden_size) # [batch, head, seq, seq]

            f_l = torch.tril(torch.ones([seq_Q.size(0), seq_Q.size(1), seq_Q.size(1)], device=self.device))
            flag = torch.cat([torch.ones([seq_Q.size(0), seq_Q.size(1), self.mem_size], device=self.device), f_l], dim=-1)  # batch, seq, mem_size+seq
            att = att.masked_fill(~flag.bool(), -1e12)

            # bool_mask = mask.unsqueeze(1).repeat(1, seq_Q.size(2), 1).unsqueeze(1).repeat(1, self.head, 1, 1).bool() # [batch, head, seq, seq]
            bool_mask = torch.cat([torch.ones([seq_Q.size(0), seq_Q.size(1), self.mem_size], device=self.device), mask.unsqueeze(1).repeat(1, seq_Q.size(1), 1)], dim=-1).bool()  # [batch, seq, mem_size+seq]
            att = torch.softmax(att.masked_fill(~bool_mask, -1e12), dim=-1)
            att_seq_emb = torch.matmul(att, seq_V) * mask.view(mask.shape[0], -1, 1).float()  # batch, seq, hidden
            # att_seq_emb = torch.cat(torch.chunk(att_seq_emb, self.head, dim=1), dim=-1).squeeze(1) * mask.view(mask.shape[0], -1, 1).float()
        else:
            att = torch.matmul(seq_Q, seq_K.transpose(2, 1)) / np.sqrt(self.hidden_size)  # batch, seq, seq
            # att = torch.matmul(seq_Q, seq_K.transpose(3, 2)) / np.sqrt(self.hidden_size) # [batch, head, seq, seq]

            f_l = torch.tril(torch.ones([seq_Q.size(0), seq_Q.size(1), seq_Q.size(1)], device=self.device))
            att = att.masked_fill(~f_l.bool(), -1e12)

            bool_mask = mask.unsqueeze(1).repeat(1, seq_Q.size(1), 1).bool() # [batch, head, seq, seq]
            att = torch.softmax(att.masked_fill(~bool_mask, -1e12), dim=-1)
            att_seq_emb = torch.matmul(att, seq_V) * mask.view(mask.shape[0], -1, 1).float()  # batch, seq, hidden

        return att_seq_emb

    def RENorm(self, sess_emb, score, item_flag):
        score_in = torch.softmax(score.masked_fill(~item_flag.bool(), float('-inf')), dim=-1)
        score_ex = torch.softmax(score.masked_fill(item_flag.bool(), float('-inf')), dim=-1)

        score_in = score_in.unsqueeze(1)
        score_ex = score_ex.unsqueeze(1)

        beta = torch.softmax(self.linear_tail_2(torch.relu(self.linear_tail_1(sess_emb))), dim=-1)
        boundary = torch.mean(torch.abs(beta[:, 0] - beta[:, 1]), dim=0)
        # print(boundary)

        finall_score = (torch.cat((score_in, score_ex), dim=1) * beta.unsqueeze(2)).sum(1)
        return finall_score, boundary

    def forward(self, seq_emb_1, seq_emb_2, mask, item_flag):  # mask = batch x seq
        seq_emb_2, _ = self.GRU(seq_emb_2)

        get_posEmb = PositionalEncoding(self.hidden_size, mask)

        sr_l_1 = seq_emb_1[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]

        q1_1 = get_posEmb(self.linear_one_1(sr_l_1), pos=0)  # batch hidden

        seq_emb_1 = seq_emb_1 * mask.view(mask.shape[0], -1, 1).float()
        seq_Q_1, seq_K_1, seq_V_1 = self.self_att_1(seq_emb_1, mask, pos_emb=True)  # batch, seq, hidden
        att_seq_emb_1 = self.FFN_1(self.co_attention(seq_Q_1, seq_K_1, seq_V_1, mask))

        seq_emb_2 = seq_emb_2 * mask.view(mask.shape[0], -1, 1).float()
        seq_Q_2, seq_K_2, seq_V_2 = self.self_att_2(seq_emb_2, mask, pos_emb=False)  # batch, seq, hidden
        att_seq_emb_2 = self.FFN_2(self.co_attention(seq_Q_1, seq_K_2, seq_V_2, mask, memory=True))
        att_seq_emb = torch.relu(att_seq_emb_1 + att_seq_emb_2)

        alpha_1 = self.linear_three_1(torch.cat([att_seq_emb, q1_1.unsqueeze(1).repeat(1, att_seq_emb.size(1), 1)], dim=-1))  # batch, seq, 1
        alpha_1 = torch.softmax(torch.where(mask.view(mask.shape[0], -1, 1).double() > 0, alpha_1.double(), -1e12), dim=1).float()
        sess_emb_1 = torch.sum(alpha_1 * seq_emb_1 * mask.view(mask.shape[0], -1, 1).float(), 1)  # batch_size x latent_size
        sess_emb_1 = self.linear_transform_1(torch.cat([sess_emb_1, sr_l_1], 1))  # batch_size x latent_size

        sess_emb = sess_emb_1
        item_emb = self.embedding_1.weight[1:] # n_nodes x latent_size

        score = torch.matmul(sess_emb, item_emb.transpose(1, 0))  # batch x n_nodes

        finall_score, boundary = self.RENorm(sess_emb, score, item_flag)
        # finall_score = torch.softmax(score, dim=-1)
        # boundary = 1

        return finall_score.view(score.size(0), 1, score.size(1)), boundary


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize

        self.embedding_1 = nn.Embedding(self.n_node, self.hidden_size)
        self.embedding_2 = nn.Embedding(self.n_node, self.hidden_size)

        self.gnn = GNN(opt)
        # self.get_score_set = ScoerModel(opt, self.embedding)
        self.get_score_set = nn.ModuleList()
        for i in range(len(opt.order_list)):
            self.get_score_set.append(ScoerModel(opt, self.embedding_1, self.embedding_2))
        self.loss_function = nn.NLLLoss()  # CrossEntropyLoss, NLLLoss
        self.reset_parameters()
        bias_p = []
        weight_p = []
        for name, p in self.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        self.optimizer = torch.optim.Adam([{'params': weight_p, 'weight_decay': opt.l2},
                                           {'params': bias_p, 'weight_decay': 0}],
                                          lr=opt.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.alpha = nn.Parameter(torch.zeros(len(opt.order_list)))
        self.alpha.data[0] = torch.tensor(1.0)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, weight in self.named_parameters():
            if 'embedding' in name:
                weight.data.normal_(mean=0.0, std=stdv)
            else:
                if 'weight' in name:
                    weight.data.uniform_(-stdv, stdv)
                else:
                    weight.data = nn.Parameter(torch.zeros(weight.data.shape))

    def get_seq(self, results, alias_inputs):
        seq_embs = []
        get = lambda index: item_embs[index][alias_inputs[index]]
        for item_embs in results:
            # item_embs, hidden = result
            seq_emb = torch.stack(
                [get(index) for index in torch.arange(len(alias_inputs)).long()])
            seq_embs.append(seq_emb)  # order x batch x item_num x hidesize
        return seq_embs

    def get_item_flag_in_sess(self, seq_embs, items):
        item_flag_in_sess = trans_to_cuda(torch.zeros(seq_embs[0].size(0), self.n_node - 1))
        for i in range(len(item_flag_in_sess)):  # len(item_flag)
            items_idx = items[i][items[i] != 0]
            item_flag_in_sess[i, items_idx - 1] = 1
        return item_flag_in_sess

    def compute_scores(self, seq_embs_1, seq_emb_2, mask, item_flag_in_sess):
        scores = []
        total_boundary = 0
        for i in range(len(seq_embs_1)):  # seq_embs = order x batch x item_num x hidesize
            score, boundary = self.get_score_set[i](seq_embs_1[i], seq_emb_2, mask,
                                                    item_flag_in_sess)  # seq_embs[i] = batch x item_num x hidesize
            # score, boundary = self.get_score_set(seq_embs[i], mask, items)  # seq_embs[i] = batch x item_num x hidesize
            scores.append(score)
            total_boundary += boundary

        fusion_scores = torch.cat(scores, 1)
        alpha = torch.softmax(self.alpha.unsqueeze(0), dim=-1).view(1, self.alpha.size(0), 1)
        g = alpha.repeat(fusion_scores.size(0), 1, 1)
        scores = (fusion_scores * g).sum(1)

        # scores = torch.squeeze(scores[0], dim=1)
        # print(scores)
        return torch.log(scores), total_boundary / len(seq_embs_1)

    def forward(self, items, A, A_I, alias_inputs, mask):
        item_flag_in_graph = torch.where(items > 0, 1, 0).unsqueeze(2).float()  # is zero?
        hidden_1 = self.embedding_1(items) * item_flag_in_graph
        results = self.gnn(A, A_I, hidden_1, alias_inputs, mask, item_flag_in_graph)
        seq_embs_1 = self.get_seq(results, alias_inputs)

        hidden_2 = self.embedding_2(items) * item_flag_in_graph
        seq_emb_2 = self.get_seq([hidden_2], alias_inputs)[0]

        item_flag_in_sess = self.get_item_flag_in_sess(seq_embs_1, items)

        scores, total_boundary = self.compute_scores(seq_embs_1, seq_emb_2, mask, item_flag_in_sess)

        return scores, total_boundary


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A, A_I, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    A_I = trans_to_cuda(torch.Tensor(A_I).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())

    scores, total_boundary = model(items, A, A_I, alias_inputs, mask)  # order x batch x item_num x hidesize
    return targets, scores, total_boundary


def train_test(opt, model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    # print(slices)
    for i, j in zip(slices, np.arange(len(slices))):  
        model.optimizer.zero_grad()
        targets, scores, total_boundary = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets - 1).long())
        # reg = 0
        # for name, param in model.named_parameters():
        #     if 'bias' not in name:
        #         reg += 0.5 * (param ** 2).sum()
        loss = model.loss_function(scores, targets)  # - 0.1*total_boundary  # + opt.l2 * reg
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 10 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores, _ = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100

    return hit, mrr, model