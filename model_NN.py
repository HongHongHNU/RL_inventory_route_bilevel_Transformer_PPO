import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch.distributions.categorical import Categorical
import math
from inventory_route_env import action_update, action_return, c_dist
from route_PPO_train import Route_Train

INIT = True


class ActorRetailer(nn.Module):
    def __init__(self, n_state_type, d_transformer, num_encoder_layers, num_decoder_layers, n_actions):
        super(ActorRetailer, self).__init__()
        self.embedding_src = nn.Linear(n_state_type, d_transformer)
        self.embedding_tgt = nn.Embedding(n_actions, d_transformer)
        self.transformer = Transformer(d_model=d_transformer, num_decoder_layers=num_decoder_layers,
                                       num_encoder_layers=num_encoder_layers)
        self.predictor = nn.Linear(d_transformer, n_actions)
        self.n_actions = n_actions
        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.xavier_uniform_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, src, tgt):
        src = src.view(self.n_actions, -1)
        src = self.embedding_src(src)
        tgt = self.embedding_tgt(tgt)
        out = self.transformer(src, tgt)
        out=out[-1,:]
        out = self.predictor(out)
        out = torch.softmax(out, dim=0)
        return out


def actor_decision(model, src, retailer_num,device):
    log_ps = 0.0
    retailer_actions = []
    for i in range(retailer_num):
        if i == 0:
            tgt = torch.LongTensor([0]).to(device)
        model.eval()
        out = model(src, tgt)
        dist = Categorical(out)
        index = dist.sample()
        retailer_action = index.data.unsqueeze(0)
        retailer_actions.append(retailer_action)
        log_p = dist.log_prob(index)
        log_ps += log_p.item()
        tgt = torch.concat([tgt, retailer_action],dim=0)
    retailer_actions = torch.cat(retailer_actions, dim=0)
    return retailer_actions, log_ps


def actor_decision_back(model, src, old_actions, retailer_num,device):
    log_ps = 0.0
    for i in range(retailer_num):
        if i == 0:
            tgt = torch.LongTensor([0]).to(device)
        out = model(src, tgt)
        log_ps += torch.log(out.gather(0, old_actions[i]))
        tgt = torch.concat([tgt, old_actions[i].unsqueeze(0)], dim=0)
    return log_ps


class ActorModel(nn.Module):
    def __init__(self, retailer_num, order_max, n_state_type, d_transformer, num_encoder_layers, num_decoder_layers,
                 hidden_node_dim,
                 hidden_edge_dim, conv_laysers, lr_route, epoch_route, entropy_value_route, eps_clip_route,
                 timestep_route, ppo_epoch_route, batch_n_train_route, iter_per_batch_route, nodes,
                 train_n_route, batch_n_valid_route, is_trained, device):
        super(ActorModel, self).__init__()
        self.device = device
        self.retailer_num = retailer_num
        self.order_max = order_max
        self.actor_retailer = ActorRetailer(n_state_type, d_transformer, num_encoder_layers, num_decoder_layers,
                                            retailer_num).to(device)
        self.actor_route_train = [Route_Train(lr_route, hidden_node_dim,
                                              hidden_edge_dim, epoch_route, conv_laysers, entropy_value_route,
                                              eps_clip_route, timestep_route, ppo_epoch_route,
                                              batch_n_train_route,
                                              batch_n_valid_route, iter_per_batch_route, nodes, n_nodes_route + 4,
                                              train_n_route) for n_nodes_route in
                                  range(self.retailer_num - 2)]
        if is_trained is False:
            print("开始训练路径\n")
            self.actor_route = [self.actor_route_train[i].train() for i in range(self.retailer_num - 2)]
            print("路径训练完毕\n")
            '''
            range(self.retailer_num-2)中-2是由于如果网络只有1、2或3个节点，不需要进行路径规划，所以actor_route_train[0]是指有网络有4个节点（包括supplier）
            '''
        else:
            self.actor_route = [self.actor_route_train[i].trainppo[self.actor_route_train[i].best_trainppo].agent.policy.to(self.device)
                                for i in range(self.retailer_num - 2)]

    def forward(self, state, nodes, batch_size, action_old, greedy, _action):
        if _action is False:
            retailer_actions, logp_retailer = actor_decision(self.actor_retailer, state, self.retailer_num,self.device)
            retailer_actions_np = retailer_actions.unsqueeze(0).cpu().detach().numpy()
            retailer_actions_np = np.array(retailer_actions_np)
            routes = []
            # logb_routes = []
            for i in range(batch_size):
                nodes_update_num = len(list(np.where(retailer_actions_np[i] != 0)[0]))  # nodes_update_num不包含supplier点
                if nodes_update_num == 0:
                    route, logb_route = torch.IntTensor([[0]]).to(self.device).type(torch.int64), torch.FloatTensor(
                        0).to(
                        self.device).type(torch.float32)
                elif nodes_update_num == 1:
                    route, logb_route = torch.IntTensor([[0, 1]]).to(self.device).type(torch.int64), torch.FloatTensor(
                        0).to(
                        self.device).type(torch.float32)
                elif nodes_update_num == 2:
                    route, logb_route = torch.IntTensor([[0, 1, 2]]).to(self.device).type(
                        torch.int64), torch.FloatTensor(
                        0).to(
                        self.device).type(torch.float32)
                else:
                    mask_actions = np.where(retailer_actions_np[i] == 0)[0]
                    nodes_update = np.delete(nodes, list(mask_actions + 1), 0)
                    edges = np.zeros((nodes_update_num + 1, nodes_update_num + 1, 1))
                    edges_index = []
                    for j, (x1, y1) in enumerate(nodes_update):
                        for k, (x2, y2) in enumerate(nodes_update):
                            d = c_dist((x1, y1), (x2, y2))
                            edges[j][k][0] = d
                            edges_index.append([j, k])
                    edges = edges.reshape(-1, 1)
                    edges_index = torch.LongTensor(edges_index).to(self.device)
                    edges_index = edges_index.transpose(dim0=0, dim1=1)
                    data = Data(x=torch.from_numpy(nodes_update).to(self.device).float(), edge_index=edges_index,
                                edge_attr=torch.from_numpy(edges).to(self.device).float())
                    batch = DataLoader([data], batch_size=1)
                    for ii, datas in enumerate(batch):
                        route, logb_route = self.actor_route[nodes_update_num - 3].act(datas, 0, nodes_update_num + 1,
                                                                                       1, False, False)
                routes.append(route)
                # logb_routes.append(logb_route)
            logb_route = logb_route.detach()
            logp_out = torch.cat(
                [torch.FloatTensor([logp_retailer]).to(self.device).unsqueeze(0), logb_route.unsqueeze(0)],
                dim=1)
            actions = []
            for i in range(batch_size):
                actions.append(list(retailer_actions_np[i]) + routes[i].cpu().detach().tolist()[0])
            return actions, logp_out

        else:
            retailer_actions = action_old[0:self.retailer_num]
            routes = action_old[self.retailer_num:]
            logp = 0.0
            logp += actor_decision_back(self.actor_retailer, state,retailer_actions, self.retailer_num,self.device)
            retailer_actions = retailer_actions.cpu().numpy()
            nodes_update_num = len(list(np.where(retailer_actions != 0)[0]))

            if nodes_update_num >= 3:
                mask_actions = np.where(retailer_actions == 0)[0]
                nodes_update = np.delete(nodes, list(mask_actions + 1), 0)
                edges = np.zeros((nodes_update_num + 1, nodes_update_num + 1, 1))
                edges_index = []
                for j, (x1, y1) in enumerate(nodes_update):
                    for k, (x2, y2) in enumerate(nodes_update):
                        d = c_dist((x1, y1), (x2, y2))
                        edges[j][k][0] = d
                        edges_index.append([j, k])
                edges = edges.reshape(-1, 1)
                edges_index = torch.LongTensor(edges_index).to(self.device)
                edges_index = edges_index.transpose(dim0=0, dim1=1)
                data = Data(x=torch.from_numpy(nodes_update).to(self.device).float(), edge_index=edges_index,
                            edge_attr=torch.from_numpy(edges).to(self.device).float())
                batch = DataLoader([data], batch_size=1)
                for ii, datas in enumerate(batch):
                    _, logb_route, _ = self.actor_route[nodes_update_num - 3].evaluate(datas, routes,
                                                                                       nodes_update_num + 1,
                                                                                       1, greedy, _action)
                logp += logb_route.item()
            return logp


class CriticModel(nn.Module):
    def __init__(self, retailer_num, obs_size, hidden_size, device):
        super(CriticModel, self).__init__()
        self.retailer_num = retailer_num
        self.device = device
        self.ls1 = nn.Linear(obs_size, hidden_size // 2)
        self.ls2 = nn.Linear(hidden_size // 2, hidden_size)
        self.ls3 = nn.Linear(hidden_size, hidden_size // 2)
        self.ls4 = nn.Linear(hidden_size // 2, 1)

    def forward(self, state):
        x = F.relu(self.ls1(state))
        x = F.relu(self.ls2(x))
        x = F.relu(self.ls3(x))
        x = self.ls4(x)
        return x
