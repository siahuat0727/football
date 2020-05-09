import time
import signal
import pickle
import argparse
import operator
import functools
import os
from os.path import join
from collections import namedtuple

import torch

import gfootball.env as football_env
from replay_pool import ReplayPool
from average_meter import AverageMeter
from model import DQN, DDQN, DuelingDQN, DuelingDDQN, PPO, DDPG


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
    parser.add_argument('--env', type=str, default='academy_empty_goal',
                        help='no gpu')

    # Hyperparams
    parser.add_argument('--batch_size', type=int,
                        default=512, help='batch size')
    parser.add_argument('--hidden_size', type=int, nargs='+',
                        default=[1024], help='hidden size')
    parser.add_argument('--memory_size', type=int,
                        default=1000000, help='memory size')
    parser.add_argument('--update_freq', type=int,
                        default=10000, help='update freq')
    parser.add_argument('--init_episode', type=int,
                        default=100, help='init episode')
    parser.add_argument('--lr', type=float, default=0.00001475, help='lr')
    parser.add_argument('--gamma', type=float,
                        default=0.999, help='gamma for q_value')
    parser.add_argument('--max_eps', type=float, default=1.0, help='max eps')
    parser.add_argument('--min_eps', type=float, default=0.01, help='min eps')
    parser.add_argument('--eps_decay', type=int,
                        default=1000, help='eps decay per episode')

    # Show
    parser.add_argument('--min_show', type=int, default=100, help='')
    parser.add_argument('--test', action='store_true', help='')

    args = parser.parse_args()

    # Checkpoint directory
    args.dir = 'checkpoints/{}_{}_{}'.format(
        args.arch, args.env, int(time.time()))
    if args.test:
        args.dir = join('/tmp', args.dir)

    if args.arch == 'PPO' or args.arch == 'DDPG':
        args.cumulated_reward=True
    else:
        args.cumulated_reward=False #TODO clean

    args.gpu = not args.cpu
    args.eps_greedy = args.arch in ['DQN', 'DDQN', 'DuelingDQN', 'DuelingDDQN']
    return args


def get_env(args):
    if args.env == 'CartPole-v0':
        import gym
        env = gym.make(args.env)
    else:
        env = football_env.create_environment(env_name=args.env,
                                              representation='simple115',
                                              number_of_left_players_agent_controls=1,
                                              stacked=False,
                                              logdir=join(
                                                  '/tmp/football', args.dir),
                                              write_goal_dumps=False,
                                              write_full_episode_dumps=False,
                                              render=False)
    n_s = functools.reduce(operator.mul, env.observation_space.shape, 1)
    n_a = env.action_space.n
    return env, n_s, n_a


def last_info(rewards, name, min_show):
    if len(rewards) < min_show:
        return ''
    win = sum(r == 1 for r in rewards[-min_show:])
    lose = sum(r == -1 for r in rewards[-min_show:])
    return f', {name} last {min_show} win={win} lose={lose}'


def main():
    args = parse_args()
    print(args)

    os.makedirs(args.dir)

    env, n_s, n_a = get_env(args)

    models = {
        'DQN': DQN,
        'DDQN': DDQN,
        'DuelingDQN': DuelingDQN,
        'DuelingDDQN': DuelingDDQN,
        'PPO': PPO,
        'DDPG': DDPG,
    }
    print(models[args.arch])
    model = models[args.arch](lr=args.lr, dims=[n_s, *args.hidden_size, n_a])
    if args.gpu:
        model = model.cuda()
    # optimizer = torch.optim.Adam(model.net.parameters(), lr=args.lr)

    # Signal handling (soft terminate)
    keep_running = [True]

    def signal_handler(sig, frame):
        print("Last training...")
        keep_running[0] = False
    signal.signal(signal.SIGINT, signal_handler)

    episodes = int(3e9)
    eps_ = args.max_eps

    explore_rewards = []
    exploit_rewards = []

    if args.arch == 'PPO':
        pool = ReplayPool(namedtuple('Data', ['s', 'a', 'r', 's_', 'logprobs']),
                          capacity=args.memory_size,
                          cumulated_reward=args.cumulated_reward) # TODO
    else:
        pool = ReplayPool(namedtuple('Data', ['s', 'a', 'r', 's_']),
                          capacity=args.memory_size,
                          cumulated_reward=args.cumulated_reward) # TODO

    best_result, cur_result = 0, 0

    start = time.time()
    steps = []
    for episode in range(episodes):
        if not keep_running[0]:
            break

        eps_ = max(eps_ - (1/args.eps_decay), args.min_eps)

        if episode % 10 != 0 or episode < args.init_episode:
            rewards = explore_rewards
            eps = eps_
        else:
            rewards = exploit_rewards
            eps = 0.0

        s = env.reset()
        losses = AverageMeter()

        done = False
        step = 0
        stack = []
        while not done:
            step += 1
            print(f'\r{episode}: {sum(steps)+step}', end='')
            if args.env == 'CartPole-v0':
                env.render()

            # a = env.action_space.sample()
            if args.eps_greedy:
                a = model.eps_greedy(np2flattensor(s), n_a, eps, args.gpu)
                logprobs = np2flattensor(s)  # TODO del
            else:
                # a, _ = model.choose_act(np2flattensor(s)) # TODO cuda if need
                a, logprobs = model.choose_act(np2flattensor(s)) # TODO cuda if need


            s_, r, done, _ = env.step(a)

            # if args.env == 'CartPole-v0' and done:
            #     r = -1.0

            if args.arch == 'PPO':
                pool.record([np2flattensor(s), torch.tensor(a), torch.tensor(r), np2flattensor(s_), logprobs], done)
            else:
                pool.record([np2flattensor(s), torch.tensor(a), torch.tensor(r), np2flattensor(s_)], done)
            s = s_

            if len(pool) < args.batch_size:  # TODO change to more that init steps
                continue

            batch = pool.sample(args.batch_size, args.gpu)
            loss = model.step(batch, gamma=args.gamma)
            # loss = model.step(batch, optimizer, gamma=args.gamma)
            losses.update(loss, args.batch_size)

            if (sum(steps) + step) % args.update_freq == 0:
                # model.update_net_t()  # TODO

                end = time.time()
                # print('\nTime elapsed for {} steps: {:.2f} s'.format(
                #     args.update_freq, end-start))
                # print('10M step left {:.2f} hours'.format(
                #     (end-start)*(1e7-steps)/args.update_freq/3600))
                start = end

        rewards.append(int(r))
        steps.append(step)

        print_info(step, eps, losses, explore_rewards, exploit_rewards, args)

        if len(exploit_rewards) >= args.min_show:
            cur_result = avg(exploit_rewards[-args.min_show:])

        if cur_result > best_result:
            best_result = cur_result
            save_model(model, args.dir, 'model_best.pt')

    save_model(model, args.dir, 'model_last.pt')
    save_pickle(explore_rewards, args.dir, 'explore.pkl')
    save_pickle(exploit_rewards, args.dir, 'exploit.pkl')
    save_pickle(args, args.dir, 'args.pkl')
    rename_dir = ''.join(args.dir.split('_')[:-1])
    rename(args.dir, f'{rename_dir}_{sum(steps)}steps_{cur_result:.2f}goal')


def rename(src, dst):
    os.rename(src, dst)
    print(f'rename {src} -> {dst}')


def print_info(step, eps, losses, explore_rewards, exploit_rewards, args):
    info = f', step={step}, eps={eps:.2f}, loss={losses.avg:.2e}'
    info += last_info(explore_rewards, 'explore', args.min_show)
    info += last_info(exploit_rewards, 'exploit', args.min_show)
    print(info)


def save_model(model, dir_, name):
    path = join(dir_, name)
    torch.save(model.net_t.state_dict(), path) # TODO
    print(f'save {path}')


def save_pickle(obj, dir_, name):
    path = join(dir_, name)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'save {path}')


if __name__ == '__main__':
    main()
