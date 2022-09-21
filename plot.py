import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", dest='file', default=os.getcwd(),type=str)
parser.add_argument("--show_all", action='store_true')
parser.add_argument("--max_time", dest='max_time', default=-1,type=int)
parser.add_argument("--loss_interpolation", action='store_true')
parser.add_argument("--title", dest='title', default="",type=str)
parser.add_argument("--highlight", dest='highlight', default=None,type=int)
args = parser.parse_args()

if os.path.isfile(args.file) and args.file.split('.')[-1] == 'log':
    lst = [args.file]
else:
    lst = [os.path.join(args.file, filename) for filename in os.listdir(args.file) if os.path.isfile(os.path.join(args.file, filename)) and filename.split('.')[-1] == 'log']

fig, axes = plt.subplots(nrows=int(np.ceil(len(lst)/2)), ncols=int(np.ceil(len(lst)/2)),figsize=(200, 20),sharex=True,sharey=True)
if args.title != "":
    fig.suptitle(args.title.replace('_',' '))
previous_loss_axis = None
for f, idx in zip(lst,range(len(lst))):
    with open(f) as f:
        lines = f.readlines()

        rewards = []
        loss = []
        x_loss = []
        x_reward = []
        count = -1
        loss_appeared = False
        loss_started = False
        for line in lines:
            if 'buffer size:' in line: #beginning of one block
                if not args.show_all and not loss_appeared and loss_started:
                    break
                count += 1
                loss_appeared = False
                continue

            if 'average episode return:' in line:
                rewards.append(float(line.split(' ')[-1]))
                x_reward.append(count*20/60)
                reward_started = True
            elif 'loss:' in line:
                loss.append(float(line.split(' ')[-1]))
                x_loss.append(count*20/60)
                loss_appeared = True
                loss_started = True

        #x = np.arange(0,len(rewards)*20/60,20/60)

        if args.max_time > 0:
            max_iter = min(int(args.max_time*60/20),len(x_reward))
            x_reward = x_reward[:max_iter]
            rewards = rewards[:max_iter]
            x_loss = x_loss[:max_iter]
            loss = loss[:max_iter]

        interpolation = make_interp_spline(x_reward, rewards)
        x_inter = np.linspace(min(x_reward), max(x_reward), 120)
        rewards_interpolated = interpolation(x_inter)

        if args.loss_interpolation:
            interpolation_loss = make_interp_spline(x_loss, loss)
            x_inter_loss = np.linspace(min(x_loss), max(x_loss), 120)
            loss_interpolated = interpolation_loss(x_inter_loss)


        #fig, ax1 = plt.subplots(1, 1, figsize=(8,4))
        if type(axes) is np.ndarray:
            ax1 = axes[idx//2][idx%2]
            ax1.set_title('Agent {}'.format(idx + 1))
            if args.highlight == idx:
                ax1.set_facecolor('lightyellow')
        else:
            ax1 = axes
        ax1.plot(x_reward,rewards,color='tab:blue',alpha=0.3)
        ax1.plot(x_inter,rewards_interpolated,color='tab:blue')
        #ax1.set_xlim(0,max(x_loss)) #only show training reward
        ax1.tick_params('y',colors='tab:blue')
        if idx % 2 == 0:
            ax1.set_ylabel('avg reward',color='tab:blue')
        if idx//2 == len(lst)//2 - 1:
            ax1.set_xlabel('time in min')
        #ax1.set_xticks(list(np.arange(0,len(rewards)*20/60,60))+[np.ceil(len(rewards)*20/60)])
        ax2 = ax1.twinx()
        if previous_loss_axis is None:
            previous_loss_axis = ax2
        ax2.get_shared_y_axes().join(ax2, previous_loss_axis)
        if args.loss_interpolation:
            ax2.plot(x_loss, loss, color='tab:red', alpha=0.3)
            ax2.plot(x_inter_loss, loss_interpolated, color='tab:red', alpha=0.5)
        else:
            ax2.plot(x_loss,loss,color='tab:red',alpha=0.5)
        ax2.tick_params('y',colors='tab:red')
        if len(lst) > 1:
            if idx%2 == len(lst)//2 - 1:
                ax2.set_ylabel('loss',color='tab:red')
            if idx%2 == 0:
                ax2.yaxis.set_ticklabels([])
        else:
            ax2.set_ylabel('loss',color='tab:red')

plt.show()
