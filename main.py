import time
import signal
import random
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
from model import DQN, DDQN, DuelingDQN, DuelingDDQN, PPO, DDPG, TD3


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
    parser.add_argument('--env', type=str, default='academy_empty_goal',
                        help='env')

    # Hyperparams
    parser.add_argument('--batch_size', type=int,
                        default=2048, help='batch size')
    parser.add_argument('--hidden_size', type=int, nargs='+',
                        default=[1024], help='hidden size')
    parser.add_argument('--memory_size', type=int,
                        default=1000000, help='memory size')
    parser.add_argument('--update_freq', type=int,
                        default=10000, help='update freq')
    parser.add_argument('--init_episode', type=int,
                        default=0, help='init episode')
    parser.add_argument('--lr', type=float, default=0.0001475, help='lr')
    parser.add_argument('--gamma', type=float,
                        default=0.999, help='gamma for q_value')
    parser.add_argument('--max_eps', type=float, default=1.0, help='max eps')
    parser.add_argument('--min_eps', type=float, default=0.01, help='min eps')
    parser.add_argument('--eps_decay', type=int,
                        default=500, help='eps decay per episode')
    parser.add_argument('--k_epoch', type=int,
                        default=0, help='update k_epoch after sample')

    # Show
    parser.add_argument('--min_show', type=int, default=100, help='')
    parser.add_argument('--test', action='store_true', help='')

    args = parser.parse_args()

    # Checkpoint directory
    args.dir = 'checkpoints/{}_{}_{}'.format(
        args.arch, args.env, int(time.time()))
    if args.test:
        args.dir = join('/tmp', args.dir)

    # if args.arch == 'PPO' or args.arch == 'DDPG':
    if args.arch == 'PPO':
        args.cumulated_reward = True
    else:
        args.cumulated_reward = False  # TODO clean

    assert args.arch != 'PPO' or args.k_epoch > 0, f'Using PPO, k_epoch must > 0'

    args.update_per_step = args.k_epoch == 0

    args.eps_greedy = args.arch in [
        'DQN', 'DDQN', 'DuelingDQN', 'DuelingDDQN', 'DDPG', 'TD3']
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


def choose_act(model, s, n_a, eps=0.1):  # TODO eps case in main function
    if random.random() < eps:
        a_random = random.randrange(n_a)
        return torch.tensor(a_random).to(s.device), None
    else:
        return model.choose_act(s.unsqueeze(0), add_noise=False)


def main():
    args = parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.dir)

    env, n_s, n_a = get_env(args)

    models = {
        'DQN': DQN,
        'DDQN': DDQN,
        'DuelingDQN': DuelingDQN,
        'DuelingDDQN': DuelingDDQN,
        'PPO': PPO,
        'DDPG': DDPG,
        'TD3': TD3,
    }
    model = models[args.arch](in_dim=n_s, hid_dims=args.hidden_size, out_dim=n_a, lr=args.lr).to(device)
    # optimizer = torch.optim.Adam(model.net.parameters(), lr=args.lr)

    # Signal handling (soft terminate)
    keep_running = [True]

    def signal_handler(sig, frame):
        print("Last training...")
        keep_running[0] = False
    signal.signal(signal.SIGINT, signal_handler)

    episodes = int(3e9)
    eps_ = args.max_eps

    if args.arch == 'PPO':
        pool = ReplayPool(namedtuple('Data', ['s', 'a', 'r', 's_', 'logprobs']),
                          capacity=args.memory_size,
                          cumulated_reward=args.cumulated_reward)  # TODO
    else:
        pool = ReplayPool(namedtuple('Data', ['s', 'a', 'r', 's_']),
                          capacity=args.memory_size,
                          cumulated_reward=args.cumulated_reward)  # TODO

    best_result, cur_result = 0, 0

    start = time.time()
    steps = []
    losss = []
    losss2 = []
    explore_rewards = []
    exploit_rewards = []
    rewards = exploit_rewards
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
        s = np2flattensor(s).to(device)

        losses = AverageMeter()
        losses2 = AverageMeter()

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
                a, _ = choose_act(model, s, n_a, eps)
            else:
                a, logprobs = model.choose_act(s.unsqueeze(0))

            a = a.detach()
            s_, r, done, _ = env.step(a.item())

            if args.env == 'CartPole-v0' and done:
                r = -1.0

            s_ = np2flattensor(s_).to(device)
            r = torch.tensor(r).to(device)

            if args.arch == 'PPO':
                pool.record([s, a, r, s_, logprobs], done)
            else:
                pool.record([s, a, r, s_], done)

            s = s_

            if len(pool) < args.batch_size:  # TODO change to more that init steps
                continue

            if args.update_per_step:
                batch = pool.sample(args.batch_size)
                if args.arch in ['DDPG', 'TD3']:
                    loss1, loss2 = model.step(batch, gamma=args.gamma)
                    losses.update(loss1, args.batch_size)
                    losses2.update(loss2, args.batch_size)
                else:
                    loss = model.step(batch, gamma=args.gamma)
                    losses.update(loss, args.batch_size)

            if (sum(steps) + step) % args.update_freq == 0:
                pass
                # model.update_net_t()  # TODO

        if not args.update_per_step and len(pool) >= args.batch_size:
            def _list2tensor(data):
                data = list(zip(*data))
                data = [torch.stack(d).view(len(d), -1) for d in data]
                return data

            batch = namedtuple('Data', ['s', 'a', 'r', 's_', 'logprobs'])(
                *_list2tensor(pool._memory))

            for _ in range(args.k_epoch):
                loss = model.step(batch)
                losses.update(loss, len(pool))
            pool.clean()

        rewards.append(int(r))
        steps.append(step)
        losss.append(losses.avg)
        losss2.append(losses2.avg)

        # print_info(step, None, losses, explore_rewards, exploit_rewards, args)
        # print_info(step, eps, losses, explore_rewards, exploit_rewards, args)
        print_info(step, eps, losses, losses2,
                   explore_rewards, exploit_rewards, args)

        if episode % 100 == 0:
            end = time.time()
            print(
                f'\nTime elapsed for {sum(steps[-100:])} steps: {end-start:.2f} s')
            print('10M step left {:.2f} hours'.format(
                (end-start)*(1e7-sum(steps))/sum(steps[-100:])/3600))
            start = end

        if len(exploit_rewards) >= args.min_show:
            cur_result = avg(exploit_rewards[-args.min_show:])

        if cur_result > best_result:
            best_result = cur_result
            # save_model(model, args.dir, 'model_best.pt')

    # save_model(model, args.dir, 'model_last.pt')
    save_pickle(explore_rewards, args.dir, 'explore.pkl')
    save_pickle(exploit_rewards, args.dir, 'exploit.pkl')
    save_pickle(steps, args.dir, 'steps.pkl')
    save_pickle(losss, args.dir, 'losss.pkl')
    save_pickle(losss2, args.dir, 'losss2.pkl')
    save_pickle(args, args.dir, 'args.pkl')
    rename_dir = '_'.join(args.dir.split('_')[:-1])
    rename(args.dir, f'{rename_dir}_{sum(steps)}steps_{cur_result:.2f}goal')


def rename(src, dst):
    os.rename(src, dst)
    print(f'rename {src} -> {dst}')


# def print_info(step, eps, losses, explore_rewards, exploit_rewards, args):
def print_info(step, eps, losses, losses2, explore_rewards, exploit_rewards, args):
    # info = f', step={step}, eps={eps:.2f}, loss={losses.avg:.2e}'
    info = f', step={step}, eps={eps:.2f}, loss critic={losses.avg:.2e}, loss actor={losses2.avg:.2e}'
    info += last_info(explore_rewards, 'explore', args.min_show)
    info += last_info(exploit_rewards, 'exploit', args.min_show)
    print(info)


def save_model(model, dir_, name):
    path = join(dir_, name)
    # torch.save(model.net.state_dict(), path) # TODO
    print(f'save {path}')


def save_pickle(obj, dir_, name):
    path = join(dir_, name)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'save {path}')


if __name__ == '__main__':
    main()
