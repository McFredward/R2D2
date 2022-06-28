import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np

with open('models/Basic_with_attack1/train.log') as f:
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
        if not loss_appeared and loss_started:
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

interpolation = make_interp_spline(x_reward, rewards)
x_inter = np.linspace(min(x_reward), max(x_reward), 120)
rewards_interpolated = interpolation(x_inter)

fig, ax1 = plt.subplots(1, 1, figsize=(8,4))
ax1.plot(x_reward,rewards,color='tab:blue',alpha=0.3)
ax1.plot(x_inter,rewards_interpolated,color='tab:blue')
#ax1.set_xlim(0,max(x_loss)) #only show training reward
ax1.tick_params('y',colors='tab:blue')
ax1.set_ylabel('avg reward',color='tab:blue')
ax1.set_xlabel('time in min')
ax1.set_xticks(list(np.arange(0,len(rewards)*20/60,60))+[np.ceil(len(rewards)*20/60)])
ax2 = ax1.twinx()
ax2.plot(x_loss,loss,color='tab:red',alpha=0.5)
ax2.tick_params('y',colors='tab:red')
ax2.set_ylabel('loss',color='tab:red')
plt.show()