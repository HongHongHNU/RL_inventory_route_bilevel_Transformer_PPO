import os
import time
import torch
import torch.nn as nn
from route_PPO_Model import Agentppo, Memory
from route_create_tsp_instance import creat_data, reward, reward1, combination_num
from collections import OrderedDict
from collections import namedtuple
from itertools import product
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# device = torch.device('cpu')


def rollout(model, dataset, batch_size, n_nodes):
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model.act(bat, 0, n_nodes, batch_size, False, False)
            cost = reward1(bat.x, cost.detach(), n_nodes)
        return cost.cpu()

    totall_cost = torch.cat([eval_model_bat(bat.to(device)) for bat in dataset], 0)
    return totall_cost


class TrainPPO:
    def __init__(self, steps, greedy, lr, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, epoch=40,
                 batch_size=32, conv_laysers=3, entropy_value=0.01, eps_clip=0.2, timestep=4, ppo_epoch=2):

        self.steps = steps
        self.greedy = greedy
        self.batch_size = batch_size
        self.update_timestep = timestep
        self.epoch = epoch
        self.memory = Memory()
        self.agent = Agentppo(steps, greedy, lr, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim,
                              ppo_epoch, batch_size, conv_laysers, entropy_value, eps_clip)

    def run_train(self, data_loader, batch_size, valid_loder, n_nodes):
        memory = Memory()
        self.agent.old_polic.to(device)

        folder = 'tsp-{}-GAT'.format(n_nodes)
        # filename = 'ppomodel'
        # filepath = os.path.join(folder, filename)

        costs = []
        for i in range(self.epoch):
            print('old_epoch:', i, '***************************************')
            self.agent.old_polic.train()
            times, losses, rewards2, critic_rewards = [], [], [], []
            epoch_start = time.time()
            start = epoch_start
            mean_reward_temp = []
            global is_break
            is_break = False

            for batch_idx, batch in enumerate(data_loader):
                '''if batch_idx%self.update_timestep == 0:
                    print('old_batch:',batch_idx,'***************************************')'''
                x, attr = batch.x, batch.edge_attr
                # print(x.size(),index.size(),attr.size())

                x, attr = x.view(batch_size, n_nodes, 2), attr.view(batch_size, n_nodes * n_nodes, 1)
                batch = batch.to(device)
                actions, log_p = self.agent.old_polic.act(batch, 0, self.steps, batch_size, self.greedy, False)
                rewards = reward(batch.x, actions, n_nodes, batch_size)

                actions = actions.to(torch.device('cpu')).detach()
                log_p = log_p.to(torch.device('cpu')).detach()
                rewards = rewards.to(torch.device('cpu')).detach()

                # print(actions.size(),log_p.size(),entropy.size())

                for i_batch in range(self.batch_size):
                    memory.input_x.append(x[i_batch])
                    # memory.input_index.append(index[i_batch])
                    memory.input_attr.append(attr[i_batch])
                    memory.actions.append(actions[i_batch])
                    memory.log_probs.append(log_p[i_batch])
                    memory.rewards.append(rewards[i_batch])
                if (batch_idx + 1) % self.update_timestep == 0:
                    self.agent.update(memory, i, n_nodes)
                    memory.def_memory()
                rewards2.append(torch.mean(rewards.detach()).item())
                time_Space = 100
                if (batch_idx + 1) % time_Space == 0:
                    print('epoch:', i, '-----------------------------------------------')
                    end = time.time()
                    times.append(end - start)
                    start = end
                    mean_reward = np.mean(rewards2[-time_Space:])
                    mean_reward_temp.append(mean_reward)
                    print('  Batch %d/%d, reward: %2.3f,took: %2.4fs' %
                          (batch_idx, len(data_loader), mean_reward,
                           times[-1]))
                    if len(mean_reward_temp) > 4:
                        if np.std(np.array(mean_reward_temp[-4:])) < 0.003:
                            '''
                            注意，此处设置参数值4和阀值0.003，用于控制由于波动较小退出情况
                            '''
                            print("近4次波动较小，结束")
                            costs.append(mean_reward)
                            is_break = True

                            break
            if is_break:
                break

            cost = rollout(self.agent.policy, valid_loder, batch_size, n_nodes)
            cost = cost.mean()
            costs.append(cost.item())
            print('Problem:TSP''%s' % n_nodes, '/ Average distance:', cost.item())
            print(costs)
            epoch_dir = os.path.join(folder, '%s' % i)
            if not os.path.exists(epoch_dir):
                os.makedirs(epoch_dir)
            save_path = os.path.join(epoch_dir, 'actor.pt')
            torch.save(self.agent.old_polic.state_dict(), save_path)
            if i > 0:
                if costs[-1] == min(costs) and costs[-1] / min(costs[:-1]) > 0.995:
                    print("最近一次epoch为历史最优，且离次优差别较小，结束")
                    break
        if not os.path.exists(folder):
            os.makedirs(folder)
        save_path = os.path.join(folder, 'actor.pt')
        torch.save(self.agent.policy.state_dict(), save_path)
        return self.agent.policy, costs[-1]


class Route_Train():
    def __init__(self, lr, hidden_node_dim,
                 hidden_edge_dim, epoch, conv_laysers, entropy_value, eps_clip, timestep, ppo_epoch, batch_n_train,
                 batch_n_valid, iter_per_batch, nodes, n_nodes, train_n):
        self.lr = lr
        self.hidden_node_dim = hidden_node_dim
        self.hidden_edge_dim = hidden_edge_dim
        self.epoch = epoch
        self.conv_laysers = conv_laysers
        self.entropy_value = entropy_value
        self.eps_clip = eps_clip
        self.timestep = timestep
        self.batch_n_train = batch_n_train
        self.batch_n_valid = batch_n_valid
        self.iter_per_batch = iter_per_batch
        self.nodes = nodes
        self.n_nodes = n_nodes
        self.ppo_epoch = ppo_epoch
        self.train_n = train_n
        batch_size_per = combination_num(len(nodes), n_nodes - 1)
        self.batch_size = (iter_per_batch // batch_size_per) * batch_size_per
        self.trainppo = [
            TrainPPO(self.n_nodes, False, self.lr, 2, self.hidden_node_dim, 1, self.hidden_edge_dim, self.epoch,
                     self.batch_size, self.conv_laysers,
                     self.entropy_value, self.eps_clip, self.timestep, self.ppo_epoch) for _ in range(self.train_n)]
        self.best_trainppo=0

    def train(self):
        data_loader, batch_size = creat_data(self.nodes, self.n_nodes, self.iter_per_batch, self.batch_n_train)
        valid_loader, _ = creat_data(self.nodes, self.n_nodes, self.iter_per_batch, self.batch_n_valid)

        print('DATA CREATED/Problem size:', self.n_nodes)

        cost_min = 0.0
        for i in range(self.train_n):
            print("trian: %d =============================" % i)
            if i == 0:
                net_act, cost_min = self.trainppo[i].run_train(data_loader, batch_size, valid_loader, self.n_nodes)
                self.best_trainppo=i
            else:
                net_act_temp, cost_temp = self.trainppo[i].run_train(data_loader, batch_size, valid_loader,
                                                                     self.n_nodes)
                if cost_temp < cost_min:
                    net_act = net_act_temp
                    cost_min = cost_temp
                    self.best_trainppo=i
        print("cost_min: %.3f" % cost_min)
        folder = 'tsp-{}-GAT'.format(self.n_nodes)
        if not os.path.exists(folder):
            os.makedirs(folder)
        save_path = os.path.join(folder, 'actor.pt')
        torch.save(net_act.state_dict(), save_path)
        return net_act

#
# lr = 3e-4
# hidden_node_dim = 64
# hidden_edge_dim = 8
# epoch = 30
# conv_laysers = 2
# entropy_value = 0.001
# eps_clip = 0.2
# timestep = 1
# ppo_epoch = 1
# batch_n_train = 1000
# batch_n_valid = 1
# iter_per_batch = 50
# nodes = [[0, 0], [2, 1], [2, -1], [-2, 1], [-2, -1], [5, 0]]
# n_nodes = 4  # 包含起始
# train_n = 3  # 考虑存在局部最优解，对求解过程重复train_n次，取最小值
# route_train_try = Route_Train(lr, hidden_node_dim,
#                               hidden_edge_dim, epoch, conv_laysers, entropy_value, eps_clip, timestep, ppo_epoch,
#                               batch_n_train,
#                               batch_n_valid, iter_per_batch, nodes, n_nodes, train_n)
# _ = route_train_try.train()
