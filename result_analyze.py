import torch
import main
from model_NN import ActorModel, CriticModel
import numpy as np
from torch_geometric.data import Data, DataLoader
import inventory_route_env

GAMMA = 0.99
GAE_LAMBDA = 0.95
TRAJECTORY_SIZE = 1025
LEARNING_RATE_ACTOR = 8e-6
LEARNING_RATE_CRITIC = 6e-5

PPO_EPS = 0.2
PPO_EPOCHES = 1
PPO_BATCH_SIZE = 512

TEST_ITERS = 2000
MAX_ITERATION_NUM = 1000000

RETAILER_NUM = 5
LEAD_TIME = 1
TRANSPORT_COST_LEAD_TIME = 0  # 指订货后几天记运输费，不超过LEAD_TIME
DEMAND_MU = [20, 20, 20, 20]
DEMAND_SIG = [1, 1, 1, 1]
ORDER_MAX = 5
ORDER_PER = 10  # 每个单位order有多少货物
INVENTORY_INIT = 20
INVENTORY_COST = [1, 1, 1, 1]
SHORTAGE_COST = [2, 2, 2, 2]
TRANSPORT_COST = 5
NODES = [[0, 0], [1, 1], [1, -1], [-1, -1], [-1, 1],[4,0]]
TOTAL_LIFE_TIME = 6

TEST_COUNT = 32
D_TRANSFORMER = 1024
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
HIDDEN_SIZE_CRT = 512
HIDDEN_NODE_DIM_ACT = 64
HIDDEN_EDGE_DIM_ACT = 8
CONV_LAYERS_ACT = 2

GREEDY = False

LEARNING_RATE_ROUTE = 1e-4
EPOCH_ROUTE = 2
ENTROPY_VALUE_ROUTE = 0.001
EPS_CLIP_ROUTE = 0.2
TIMESTEP_ROUTE = 1
PPO_EPOCH_ROUTE = 1
BATCH_N_TRAIN_ROUTE = 500
ITER_PER_BATCH_ROUTE = 512
TRAIN_N_ROUTE = 3  # 考虑存在局部最优解，对求解过程重复train_n次，取最小值
BATCH_N_VALID_ROUTE = 1

dir = "results\\1212225815\\tr_om_GNN_PP0net_act_-371_121371247.dat"
checkpoint = torch.load(dir)
device = "cuda"
net = ActorModel(RETAILER_NUM, ORDER_MAX,
                 int(LEAD_TIME + 3),
                 D_TRANSFORMER, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, HIDDEN_NODE_DIM_ACT,
                 HIDDEN_EDGE_DIM_ACT,
                 CONV_LAYERS_ACT, LEARNING_RATE_ROUTE, EPOCH_ROUTE, ENTROPY_VALUE_ROUTE, EPS_CLIP_ROUTE,
                 TIMESTEP_ROUTE, PPO_EPOCH_ROUTE, BATCH_N_TRAIN_ROUTE, ITER_PER_BATCH_ROUTE, NODES,
                 TRAIN_N_ROUTE, BATCH_N_VALID_ROUTE, True, device).to(device)
net.load_state_dict(checkpoint)

test_env = inventory_route_env.InventoryRouteEnv(RETAILER_NUM, DEMAND_MU, DEMAND_SIG, INVENTORY_INIT, ORDER_MAX,
                                                 ORDER_PER, INVENTORY_COST, SHORTAGE_COST, TRANSPORT_COST, NODES,
                                                 TOTAL_LIFE_TIME, TRANSPORT_COST_LEAD_TIME, LEAD_TIME)
'''
test_env = inventory_route_om_tr.InventoryRouteEnv(RETAILER_NUM, DEMAND_MU, DEMAND_SIG,
                                                   INVENTORY_INIT, ORDER_MAX,
                                                   ORDER_PER, INVENTORY_COST,
                                                   SHORTAGE_COST, TRANSPORT_COST, NODES,
                                                   TOTAL_LIFE_TIME, LEAD_TIME)
'''

inventory_route_env.print_all_action(net, test_env, NODES, "cuda", False)
