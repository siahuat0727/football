import gfootball.env as football_env

import operator
import functools
import time
import signal
import pickle
import argparse
import os
from os.path import join
import random

import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from model import DQN, DDQN, DuelingDQN, DuelingDDQN
from replay_pool import ReplayPool

import pdb

def np2flattensor(d):
    return torch.from_numpy(d).view(-1).float()


def avg(d):
    return sum(d)/len(d)



def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch DQN Training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Experiment setting
    parser.add_argument('--arch', type=str, default='DQN',
                        help='decrease learning rate at these epochs')
    parser.add_argument('--cpu', action='store_true', help='no gpu')

    # Hyperparams
    parser.add_argument('--update_freq', type=int, default=2500,
                        help='decrease learning rate at these epochs')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='decrease learning rate at these epochs')
    parser.add_argument('--memory_size', type=int, default=1000000,
                        help='decrease learning rate at these epochs')
    parser.add_argument('--decay_step', type=int, default=1000000,
                        help='decrease learning rate at these epochs')

    # Show
    parser.add_argument('--min_show', type=int, default=100,
                        help='decrease learning rate at these epochs')

    args = parser.parse_args()

    args.dir = '{}_{}'.format(args.arch, int(time.time()))
    args.gpu = not args.cpu

    return args

def main():
    args = parse_args()
    os.makedirs(args.dir)

    env = football_env.create_environment(env_name="academy_3_vs_1_with_keeper",
                                          representation='simple115',
                                          number_of_left_players_agent_controls=1,
                                          stacked=False,
                                          logdir=join('/tmp/football', args.dir),
                                          write_goal_dumps=False,
                                          write_full_episode_dumps=False,
                                          render=False)

    n_s = functools.reduce(operator.mul, env.observation_space.shape, 1)
    n_a = env.action_space.n

    models = {
            'DQN': DQN,
            'DDQN': DDQN,
            'DuelingDQN': DuelingDQN,
            'DuelingDDQN': DuelingDDQN,
    }

    model = models[args.arch]([n_s, 256, n_a])
    if args.gpu:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.net.parameters(), lr=0.00001475)  # TODO


    # Signal handling (soft terminate)
    keep_running = [True]
    def signal_handler(sig, frame):
        print("Last training...")
        keep_running[0] = False
    signal_old = signal.signal(signal.SIGINT, signal_handler)


    episodes = int(3e9)
    eps = 1.0
    steps = 0

    rewards = []
    pool = ReplayPool(args.memory_size)

    best_result, cur_result = 0, 0

    start = time.time()
    for episode in range(episodes):
        if keep_running[0] == False:
            break
        eps = max(eps - (1/args.decay_step), 0.01)

        done = False
        s = env.reset()

        while not done:
            steps += 1
            print('\r{}: {}'.format(episode, steps), end='')

            # a = env.action_space.sample()
            a = model.eps_greedy(np2flattensor(s), n_a, eps, args.gpu)

            s_, r, done, _ = env.step(a)

            pool.append([np2flattensor(s), torch.tensor(a),
                         torch.tensor(r), np2flattensor(s_), torch.tensor(done)])
            s = s_

            if len(pool) >= args.batch_size:
                batch = pool.sample(args.batch_size, args.gpu)
                model.train(batch, optimizer, gamma=0.999)

            if steps % args.update_freq == 0:
                end = time.time()
                print('\nTime elapsed for {} steps: {:.2f} s'.format(args.update_freq, end-start))
                print('10M step left {:.2f} hours'.format((end-start)*(1e7-steps)/args.update_freq/3600))
                start = end

                model.update_net_t()


        if done:
            rewards.append(int(r))

        info = ', eps={:.2f}, avg score {:.2f}'.format(eps, avg(rewards))

        if len(rewards) > args.min_show:
            win = sum(r==1 for r in rewards[-args.min_show:])
            lose = sum(r==-1 for r in rewards[-args.min_show:])

            info += ', last win {}/{}'.format(win, args.min_show)
            info += ', last lose {}/{}'.format(lose, args.min_show)

        print(info)


        if len(rewards) >= args.min_show:
            cur_result = avg(rewards[-args.min_show:])

        if cur_result > best_result:
            best_result = cur_result

            model_best = join(args.dir, 'model_best.pt')
            torch.save(model.net_t.state_dict(), model_best)
            print('save', model_best)

    signal.signal(signal.SIGINT, signal_old)

    model_last = join(args.dir, 'model_last.pt')
    torch.save(model.net_t.state_dict(), model_last)

    rewards_path = join(args.dir, 'rewards.pkl')
    with open(rewards_path, 'wb') as f:
        pickle.dump(rewards, f)
        print('save', rewards_path)

    os.rename(args.dir, '{}_{}steps_{:.2f}goal'.format(args.arch, steps, cur_result))


if __name__ == '__main__':
    main()
