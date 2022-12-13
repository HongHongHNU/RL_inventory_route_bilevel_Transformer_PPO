import os.path
import math
import ptan
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
from tensorboardX import SummaryWriter
from inventory_route_env import InventoryRouteEnv, Agent, test_net
from model_NN import ActorModel, CriticModel

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
LEAD_TIME = 2
TRANSPORT_COST_LEAD_TIME = 0  # 指订货后几天记运输费，不超过LEAD_TIME
DEMAND_MU = [20, 20, 20, 20, 20]
DEMAND_SIG = [1, 1, 1, 1, 1]
ORDER_MAX = 5
ORDER_PER = 10  # 每个单位order有多少货物
INVENTORY_INIT = 20
INVENTORY_COST = [1, 1, 1, 1, 1]
SHORTAGE_COST = [2, 2, 2, 2, 2]
TRANSPORT_COST = 5
NODES = [[0, 0], [1, 1], [1, -1], [-1, -1], [-1, 1], [4, 0]]
TOTAL_LIFE_TIME = 8

TEST_COUNT = 32
D_TRANSFORMER = 1024
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
HIDDEN_SIZE_CRT = 512
HIDDEN_NODE_DIM_ACT = 64
HIDDEN_EDGE_DIM_ACT = 8
CONV_LAYERS_ACT = 2

GREEDY = False

LEARNING_RATE_ROUTE = 2e-4
EPOCH_ROUTE = 2
ENTROPY_VALUE_ROUTE = 0.001
EPS_CLIP_ROUTE = 0.2
TIMESTEP_ROUTE = 1
PPO_EPOCH_ROUTE = 1
BATCH_N_TRAIN_ROUTE = 500
ITER_PER_BATCH_ROUTE = 512
TRAIN_N_ROUTE = 3  # 考虑存在局部最优解，对求解过程重复train_n次，取最小值
BATCH_N_VALID_ROUTE = 1


def calc_adv_ref(trajectory, net_crt, states_v, device="cpu"):
    values_v = net_crt(states_v)
    values = values_v.squeeze().data.cpu().numpy()
    # generalized advantage estimator: smoothed version of the advantage
    last_gae = 0.0
    result_adv = []
    result_ref = []
    for val, next_val, (exp,) in zip(reversed(values[:-1]),
                                     reversed(values[1:]),
                                     reversed(trajectory[:-1])):
        if exp.done:
            delta = exp.reward - val
            last_gae = delta
        else:
            delta = exp.reward + GAMMA * next_val - val
            last_gae = delta + GAMMA * GAE_LAMBDA * last_gae
        result_adv.append(last_gae)
        result_ref.append(last_gae + val)

    adv_v = torch.FloatTensor(np.array(list(reversed(result_adv)))).to(device)
    ref_v = torch.FloatTensor(np.array(list(reversed(result_ref)))).to(device)
    return adv_v.type(torch.float32), ref_v.type(torch.float32)


def calc_logprob(traj_states_v, net_act, nodes, traj_actions_v, _action, greedy):
    old_logprob_v = []
    for i, traj_state in enumerate(traj_states_v):
        traj_state = traj_state.unsqueeze(0)
        old_logp = net_act(traj_state, nodes, 1, traj_actions_v[i], greedy, _action)
        old_logprob_v.append(old_logp.unsqueeze(0))
    old_logprob_v = torch.cat(old_logprob_v)
    return old_logprob_v


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--cuda", default=True, help="is cuda")
    parse.add_argument("--lrc", default=LEARNING_RATE_CRITIC, help="learning rate of critic")
    parse.add_argument("--lra", default=LEARNING_RATE_ACTOR, help="learning rate of actor")
    parse.add_argument("--name", default="tr_om_GNN_PP0", help="name of the project")
    args = parse.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    time_content = time.localtime(time.time())
    time_str = str(time_content.tm_mon) + str(time_content.tm_mday) + str(time_content.tm_hour) + str(
        time_content.tm_min) + str(time_content.tm_sec)
    save_path = os.path.join("results", time_str)
    os.makedirs(save_path, exist_ok=True)

    write_content = {"TIME": [time.asctime(time.localtime(time.time()))], "GAMMA": [GAMMA], "GAE_LAMBDA": [GAE_LAMBDA],
                     "TRAJECTORY_SIZE": [TRAJECTORY_SIZE],
                     "LEARNING_RATE_CRITIC": [LEARNING_RATE_CRITIC],
                     "LEARNING_RATE_ACTOR": [LEARNING_RATE_ACTOR], "PPO_EPS": [PPO_EPS], "PPO_EPOCHES": [PPO_EPOCHES],
                     "PPO_BATCH_SIZE": [PPO_BATCH_SIZE],
                     "TEST_ITERS": [TEST_ITERS], "MAX_ITERATION_NUM": [MAX_ITERATION_NUM],
                     "RETAILER_NUM": [RETAILER_NUM], "LEAD_TIME": [LEAD_TIME],
                     "TRANSPORT_COST_LEAD_TIME": [TRANSPORT_COST_LEAD_TIME], "DEMAND_MU": DEMAND_MU,
                     "DEMAND_SIG": DEMAND_SIG, "INVENTORY_INIT": [INVENTORY_INIT], "ORDER_MAX": [ORDER_MAX],
                     "ORDER_PER": [ORDER_PER], "INVENTORY_COST": INVENTORY_COST,
                     "SHORTAGE_COST": SHORTAGE_COST, "TRANSPORT_COST": [TRANSPORT_COST], "NODES": NODES,
                     "TOTAL_LIFE_TIME": [TOTAL_LIFE_TIME],
                     "TEST_COUNT": [TEST_COUNT], "D_TRANSFORMER": [D_TRANSFORMER],
                     "NUM_ENCODER_LAYERS": [NUM_ENCODER_LAYERS], "NUM_DECODER_LAYERS": [NUM_DECODER_LAYERS],
                     "HIDDEN_SIZE_CRT": [HIDDEN_SIZE_CRT],
                     "HIDDEN_NODE_DIM_ACT": [HIDDEN_NODE_DIM_ACT], "HIDDEN_EDGE_DIM_ACT": [HIDDEN_EDGE_DIM_ACT],
                     "CONV_LAYERS_ACT": [CONV_LAYERS_ACT], "GREEDY": [GREEDY],
                     "LEARNING_RATE_ROUTE": [LEARNING_RATE_ROUTE],
                     "EPOCH_ROUTE": [EPOCH_ROUTE], "ENTROPY_VALUE_ROUTE": [ENTROPY_VALUE_ROUTE],
                     "EPS_CLIP_ROUTE": [EPS_CLIP_ROUTE], "TIMESTEP_ROUTE": [TIMESTEP_ROUTE],
                     "PPO_EPS": [PPO_EPS], "BATCH_N_TRAIN_ROUTE": [BATCH_N_TRAIN_ROUTE],
                     "ITER_PER_BATCH_ROUTE": [ITER_PER_BATCH_ROUTE], "TRAIN_N_ROUTE": [TRAIN_N_ROUTE],
                     "BATCH_N_VALID_ROUTE": [BATCH_N_VALID_ROUTE]}
    write_excel = pd.DataFrame.from_dict(write_content, orient='index')
    write_excel.to_csv("try.csv", mode="a", header=True)

    env = InventoryRouteEnv(RETAILER_NUM, DEMAND_MU, DEMAND_SIG, INVENTORY_INIT, ORDER_MAX,
                            ORDER_PER, INVENTORY_COST, SHORTAGE_COST, TRANSPORT_COST, NODES,
                            TOTAL_LIFE_TIME, TRANSPORT_COST_LEAD_TIME, LEAD_TIME)

    test_env = InventoryRouteEnv(RETAILER_NUM, DEMAND_MU, DEMAND_SIG, INVENTORY_INIT, ORDER_MAX,
                                 ORDER_PER, INVENTORY_COST, SHORTAGE_COST, TRANSPORT_COST, NODES,
                                 TOTAL_LIFE_TIME, TRANSPORT_COST_LEAD_TIME, LEAD_TIME)
    temp_env = InventoryRouteEnv(RETAILER_NUM, DEMAND_MU, DEMAND_SIG, INVENTORY_INIT, ORDER_MAX,
                                 ORDER_PER, INVENTORY_COST, SHORTAGE_COST, TRANSPORT_COST, NODES,
                                 TOTAL_LIFE_TIME, TRANSPORT_COST_LEAD_TIME, LEAD_TIME)

    net_act = ActorModel(RETAILER_NUM, ORDER_MAX,
                         int(LEAD_TIME + 3),
                         D_TRANSFORMER, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, HIDDEN_NODE_DIM_ACT,
                         HIDDEN_EDGE_DIM_ACT,
                         CONV_LAYERS_ACT, LEARNING_RATE_ROUTE, EPOCH_ROUTE, ENTROPY_VALUE_ROUTE, EPS_CLIP_ROUTE,
                         TIMESTEP_ROUTE, PPO_EPOCH_ROUTE, BATCH_N_TRAIN_ROUTE, ITER_PER_BATCH_ROUTE, NODES,
                         TRAIN_N_ROUTE, BATCH_N_VALID_ROUTE, False, device).to(device)
    net_crt = CriticModel(RETAILER_NUM,
                          int((LEAD_TIME + 3) * RETAILER_NUM),
                          HIDDEN_SIZE_CRT, device).to(device)  # 3是node的坐标点(x,y)+Linear每个节点输出1个数构成

    env = [env]
    writer = SummaryWriter(comment="-ppo")
    agent = Agent(net_act, net_crt, device=device, env=test_env, nodes=NODES, greedy=GREEDY)
    exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)

    opt_act = optim.Adam(net_act.parameters(), LEARNING_RATE_ACTOR)
    opt_crt = optim.Adam(net_crt.parameters(), LEARNING_RATE_CRITIC)
    scheduler_act = torch.optim.lr_scheduler.LambdaLR(opt_act,
                                                      lr_lambda=lambda step_idx: 1 - step_idx / MAX_ITERATION_NUM)
    scheduler_crt = torch.optim.lr_scheduler.LambdaLR(opt_crt,
                                                      lr_lambda=lambda step_idx: 1 - step_idx / MAX_ITERATION_NUM)

    trajectory = []
    best_reward = None

    print("\n开始inventory_route训练：++++++++++++++++++++++++++++++++")
    with ptan.common.utils.RewardTracker(writer) as tracker:
        for step_idx, exp in enumerate(exp_source):
            reward_steps = exp_source.pop_rewards_steps()
            if reward_steps:
                rewards, steps = zip(*reward_steps)
                tracker.reward(np.mean(rewards), step_idx)

            if step_idx % TEST_ITERS == 0:
                rewards, _ = test_net(net_act, test_env, NODES, TEST_COUNT, device, GREEDY)
                writer.add_scalar("test_rewards", rewards, step_idx)
                if best_reward is None or best_reward < rewards:
                    if best_reward is not None:
                        print("Best reward update: %.3f -> %.3f" % (best_reward, rewards))
                        local_time = time.localtime(time.time())
                        torch.save(net_act.state_dict(),
                                   save_path + "\\" + args.name + "net_act_%d_%d%d%d%d%d.dat" % (int(rewards),
                                                                                                 local_time.tm_mon,
                                                                                                 local_time.tm_mday,
                                                                                                 local_time.tm_hour,
                                                                                                 local_time.tm_min,
                                                                                                 local_time.tm_sec))
                        torch.save(net_crt.state_dict(),
                                   save_path + "\\" + args.name + "net_crt_%d_%d%d%d%d%d.dat" % (int(rewards),
                                                                                                 local_time.tm_mon,
                                                                                                 local_time.tm_mday,
                                                                                                 local_time.tm_hour,
                                                                                                 local_time.tm_min,
                                                                                                 local_time.tm_sec))
                    best_reward = rewards

            trajectory.append(exp)
            if len(trajectory) < TRAJECTORY_SIZE:
                continue

            traj_states = [t[0].state for t in trajectory]
            traj_actions = [t[0].action for t in trajectory]
            traj_states_v = torch.FloatTensor(np.array(traj_states)).to(device)
            traj_states_v = traj_states_v.type(torch.float32)
            traj_actions_v = [torch.IntTensor(np.array(traj)).to(device).type(torch.int64) for traj in traj_actions]
            # traj_actions_v = traj_actions_v.type(torch.int64)
            traj_adv_v, traj_ref_v = calc_adv_ref(trajectory, net_crt, traj_states_v, device)
            traj_adv_v = traj_adv_v.unsqueeze(-1)
            traj_ref_v = traj_ref_v.unsqueeze(-1)
            # traj_actions_v = traj_actions_v.unsqueeze(-1)
            old_logprob_v = calc_logprob(traj_states_v, net_act, NODES, traj_actions_v, True, GREEDY).detach()
            old_logprob_v = old_logprob_v[:-1]
            trajectory = trajectory[:-1]

            traj_adv_v = traj_adv_v - torch.mean(traj_adv_v)
            traj_adv_v /= torch.std(traj_adv_v)

            sum_loss_value = 0.0
            sum_loss_policy = 0.0
            count_steps = 0

            for epoch in range(PPO_EPOCHES):
                for batch_ofs in range(0, len(trajectory), PPO_BATCH_SIZE):
                    batch_l = batch_ofs + PPO_BATCH_SIZE
                    states_v = torch.index_select(traj_states_v, 0, torch.arange(batch_ofs, batch_l).to(device))
                    actions_v = traj_actions_v[batch_ofs:batch_l]
                    batch_adv_v = torch.index_select(traj_adv_v, 0, torch.arange(batch_ofs, batch_l).to(device))
                    batch_ref_v = torch.index_select(traj_ref_v, 0, torch.arange(batch_ofs, batch_l).to(device))
                    batch_old_logprob_v = torch.index_select(old_logprob_v, 0,
                                                             torch.arange(batch_ofs, batch_l).to(device))

                    opt_crt.zero_grad()
                    value_v = net_crt(states_v)
                    loss_value_v = F.mse_loss(value_v.squeeze(-1), batch_ref_v.squeeze(-1))
                    loss_value_v.backward()
                    opt_crt.step()
                    scheduler_crt.step()

                    opt_act.zero_grad()

                    log_prob_v = calc_logprob(states_v, net_act, NODES, actions_v, True, GREEDY)
                    ratio_v = torch.exp(log_prob_v - batch_old_logprob_v)
                    surr_obj_v = batch_adv_v * ratio_v
                    c_ratio_v = torch.clamp(ratio_v, 1.0 - PPO_EPS, 1.0 + PPO_EPS)
                    clipped_surr_v = batch_adv_v * c_ratio_v
                    loss_policy_v = -torch.min(surr_obj_v.squeeze(-1), clipped_surr_v.squeeze(-1)).mean()
                    loss_policy_v.backward()
                    opt_act.step()
                    scheduler_act.step()

                    sum_loss_value += loss_value_v.item()
                    sum_loss_policy += loss_policy_v.item()
                    count_steps += 1

            trajectory.clear()
            writer.add_scalar("advantage", traj_adv_v.mean().item(), step_idx)
            writer.add_scalar("value", traj_ref_v.mean().item(), step_idx)
            writer.add_scalar("loss_policy", sum_loss_policy / count_steps, step_idx)
            writer.add_scalar("loss_value", sum_loss_value / count_steps, step_idx)

            if step_idx > MAX_ITERATION_NUM:
                break
        writer.close()
        local_time = time.localtime(time.time())
        torch.save(net_act.state_dict(), args.name + "net_act_final_%d%d%d%d%d.dat" % (
            local_time.tm_mon, local_time.tm_mday, local_time.tm_hour, local_time.tm_min, local_time.tm_sec))
        torch.save(net_crt.state_dict(), args.name + "net_crt_final_%d%d%d%d%d.dat" % (
            local_time.tm_mon, local_time.tm_mday, local_time.tm_hour, local_time.tm_min, local_time.tm_sec))
        writer_result_dir = "result" + "_%d%d%d%d%d_final.csv" % (
            local_time.tm_mon, local_time.tm_mday, local_time.tm_hour, local_time.tm_min, local_time.tm_sec)
        # results = inventory_route.calc_final_result_net_act(net_act, NODES,
        #                                                           [0 for _ in range((LEAD_TIME + 2) * RETAILER_NUM)]+[0],
        #                                                           [ORDER_MAX - 1 for _ in range(RETAILER_NUM)], device)
        # resutls_pd = pd.DataFrame(results)
        # resutls_pd.columns = ["order_1", "order_2", "order_3", "order_4", "route_1", "route_2", "route_3", "route_4",
        #                       "prob"]
        # resutls_pd.to_csv(writer_result_dir)
