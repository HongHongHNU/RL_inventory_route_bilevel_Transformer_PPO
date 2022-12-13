import gym
import torch.cuda
from gym import spaces
import numpy as np
import ptan
from itertools import permutations


def c_dist(x1, x2):
    return ((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2) ** 0.5


def action_update(order_max, action, retailer_num):
    action_updated = []
    for i in range(retailer_num):
        temp = action
        if i == 0:
            action_r = temp // (order_max ** (retailer_num - 1 - i))
        elif i != retailer_num - 1:
            for j in range(i):
                temp -= action_updated[j] * order_max ** (retailer_num - 1 - j)
            action_r = temp // (order_max ** (retailer_num - 1 - i))
        else:
            for j in range(i - 1):
                temp -= action_updated[j] * order_max ** (retailer_num - 1 - j)
            action_r = temp % (order_max ** (retailer_num - i))
        action_updated.append(action_r)

    return action_updated


def action_return(order_max, action, retailer_num):
    action_back = 0
    for i in range(retailer_num):
        action_back += order_max ** (retailer_num - i - 1) * action[i]
    return action_back


class InventoryRouteEnv(gym.Env):
    def __init__(self, retailer_num, demand_mu, demand_sig, inventory_init, order_max, order_per, inventory_cost,
                 shortage_cost, transport_cost,
                 nodes, total_life_time, transport_cost_lead_time, lead_time):
        super(InventoryRouteEnv, self).__init__()
        self.retailer_num = retailer_num
        self.demand_mu = demand_mu
        self.demand_sig = demand_sig
        self.demand = [0 for _ in range(retailer_num)]
        self.inventory_init = inventory_init
        self.order_max = order_max
        self.order_per = order_per
        self.inventory_cost = inventory_cost
        self.shortage_cost = shortage_cost
        self.transport_cost = transport_cost
        self.nodes = nodes
        self.nodes_num = np.array(nodes).shape[0]
        edges = np.zeros((self.nodes_num, self.nodes_num, 1))
        for j, (x1, y1) in enumerate(self.nodes):
            for k, (x2, y2) in enumerate(self.nodes):
                d = c_dist((x1, y1), (x2, y2))
                edges[j][k][0] = d
        self.retailer_distance = edges
        self.total_life_time = total_life_time
        self.lead_time = lead_time
        self.transport_cost_lead_time = transport_cost_lead_time
        self.transport_cost_pay = [0.0 for _ in range(self.transport_cost_lead_time)]

        self.order = [[0 for _ in range(lead_time)] for _ in range(retailer_num)]
        self.inventory = [inventory_init for _ in range(retailer_num)]
        self.shortage = [0 for _ in range(retailer_num)]
        self.life_time = 0
        self.state_retailer_every = [list(i)[0] + list(i[1:])+[self.life_time] for i in zip(self.order, self.inventory, self.shortage)]
        self.state_retailer = np.array([i for j in self.state_retailer_every for i in j])
        self.is_done = False
        self.action_space = spaces.MultiDiscrete(
            [self.order_max for _ in range(self.retailer_num)] + [self.retailer_num + 1 for _ in
                                                                  range(self.retailer_num)])

    def reset(self):
        self.order = [[0 for _ in range(self.lead_time)] for _ in range(self.retailer_num)]
        self.inventory = [self.inventory_init for _ in range(self.retailer_num)]
        self.shortage = [0 for _ in range(self.retailer_num)]
        self.life_time = 0
        self.state_retailer_every = [list(i)[0] + list(i[1:])+[self.life_time] for i in zip(self.order, self.inventory, self.shortage)]
        self.transport_cost_pay = [0.0 for _ in range(self.transport_cost_lead_time + 1)]
        self.state_retailer = np.array(
            [i for j in self.state_retailer_every for i in j])
        self.is_done = False
        return self.state_retailer

    def creat_demand(self):
        self.demand = np.array(
            [np.around(np.random.normal(self.demand_mu[i], self.demand_sig[i])) for i in range(self.retailer_num)])
        np.clip(self.demand, a_min=0, a_max=1e3)

    def step(self, action):
        self.creat_demand()
        reward_temp = 0.0
        # Retailer step()
        for i in range(self.retailer_num):
            if self.inventory[i] > 0:
                self.inventory[i] += self.order[i][0]
            elif self.shortage[i] - self.order[i][0] > 0:
                self.shortage[i] -= self.order[i][0]
            else:
                self.shortage[i] = 0
                self.inventory[i] += self.order[i][0]
            if self.inventory[i] - self.demand[i] >= 0:
                self.inventory[i] -= self.demand[i]
            elif self.inventory[i] > 0:
                self.shortage[i] = self.demand[i] - self.inventory[i]
                self.inventory[i] = 0
            else:
                self.shortage[i] += self.demand[i]
            for j in range(self.lead_time):
                if j < self.lead_time - 1:
                    self.order[i][j] = self.order[i][j + 1]
                else:
                    self.order[i][j] = action[i] * self.order_per
            # Retailer reward
            reward_temp += -self.inventory_cost[i] * self.inventory[i]
            reward_temp += -self.shortage_cost[i] * self.shortage[i]

        # Manufacturer step()
        route = action[self.retailer_num:]
        for i in range(self.retailer_num):
            if action[i] == 0:
                for j in range(len(route)):
                    if route[j] >= i + 1:
                        route[j] += 1

        transport_cost_temp = 0.0
        for i in range(np.size(route)):
            if i == 0:
                transport_cost_temp += -self.retailer_distance[route[-1]][route[i]][0] * self.transport_cost
            else:
                transport_cost_temp += -self.retailer_distance[route[i - 1]][route[i]][0] * self.transport_cost

        for i in range(self.transport_cost_lead_time + 1):
            if i < self.transport_cost_lead_time:
                self.transport_cost_pay[i] = self.transport_cost_pay[i + 1]
            else:
                self.transport_cost_pay[i] = transport_cost_temp
        reward_temp += self.transport_cost_pay[0]
        self.transport_cost_pay[0] = 0  # index为0，代表当天，这表示当天需要支付的已付完

        # Creat state
        self.state_retailer_every = [list(i)[0] + list(i[1:]) +[self.life_time] for i in zip(self.order, self.inventory, self.shortage)]
        self.state_retailer = np.array(
            [i for j in self.state_retailer_every for i in j])
        inventory_new = self.inventory
        shortage_new = self.shortage
        order_new = self.order
        state_retailer_every_new = [list(i)[0] + list(i[1:]) + [self.life_time] for i in zip(order_new, inventory_new, shortage_new)]
        state_retailer_new = np.array(
            [i for j in state_retailer_every_new for i in j])

        # Update life_time
        self.life_time += 1
        is_done_new = True if self.life_time >= self.total_life_time else False

        return state_retailer_new, reward_temp, is_done_new, {"prob": 1}


class Agent(ptan.agent.BaseAgent):
    def __init__(self, net_act, net_crt, device, env, nodes, greedy):
        self.net_act = net_act
        self.net_crt = net_crt
        self.device = device
        self.env = env
        self.nodes = nodes
        self.greedy = greedy

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)
        if torch.cuda.is_available():
            states_v = states_v.type(torch.float32)
        action, _ = self.net_act(states_v, self.nodes, 1, False, self.greedy, False)
        return action, agent_states


def test_net(net_act, test_env, nodes, count=10, device="cpu", greedy=False):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        states = test_env.reset()
        while True:
            states_v = ptan.agent.float32_preprocessor(states)
            states_v = states_v.to(device).unsqueeze(0)
            action, _ = net_act(states_v, nodes, 1, 0, greedy, False)
            states, reward, done, _ = test_env.step(list(action)[0])
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


def calc_final_result_net_act(net_act, nodes, state, retailer_action, device):
    n_retailer_action = len(retailer_action)
    n_routes = 0
    state_v = torch.Tensor(state).to(device).unsqueeze(0)
    outcome = []
    for i in retailer_action:
        if i != 0:
            n_routes += 1
    route_content = [i + 1 for i in range(n_routes)]
    for route in permutations(route_content):
        prob = 0.0
        route_list = list(route)
        for i in range(n_routes + 1):
            route_updated = route_list[n_routes - i:] + [0] + route_list[:n_routes - i]
            action = retailer_action + list(route_updated)
            action_v = torch.IntTensor(action).to(device).type(torch.int64)
            prob += torch.exp(net_act(state_v, nodes, 1, action_v, False, True)).cpu().item()
        outcome.append(retailer_action + route_list + [prob])
    return outcome


def print_all_action(net_act, test_env, nodes, device, greedy):
    states = test_env.reset()
    rewards = 0.0
    steps = 0
    while True:
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(device).unsqueeze(0)
        action, logp = net_act(states_v, nodes, 1, 0, greedy, False)
        prob = torch.exp(logp.sum(dim=1)).item()
        states, reward, done, _ = test_env.step(list(action)[0])
        rewards += reward
        steps += 1
        print("step %d: \n reward: %f rewards: %f prob: %f \n action: " % (steps, reward, rewards, prob))
        print(action)
        print("state:")
        print(states)
        if done:
            break
