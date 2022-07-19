
game_name = 'Vizdoom'
env_type = 'BasicDeathmatch-v0'
#env_type = 'BasicWithAttack-v0'
pretrain = "" #name of the pretrain file in the root. No pretrain if empty string
#save_dir = '/data/lissek/R2D2/models'
save_dir = 'models'
frame_stack = 4
obs_shape = (frame_stack, 84, 84) #same as VizDOOM
frame_skip = 1


lr = 1e-4
eps = 1e-3 #Adam optimzer epsilon
grad_norm = 40 #maximum value of the total gradient norm, otherwise gradients will be clipped
batch_size = 128
learning_starts = 100#10000#50000
save_interval = 1000
target_net_update_interval = 2000
gamma = 0.997
prio_exponent = 0#0.9 #How much Prioritization should be used | 0 for no Prioritized replay
importance_sampling_exponent = 0#0.6

training_steps = 500000
buffer_capacity = 500000
max_episode_steps = 27000
actor_update_interval = 400
block_length = 400  # cut one episode to numbers of blocks to improve the buffer space utilization

amp = False # mixed precision training

num_actors = 2#10
base_eps = 0.4 #epsilon-greedy-strategy
alpha = 7 #for calculating a starting epsilon for each actor
log_interval = 20

#Multiplayer related
multiplayer = True
num_players = 2 # [Multiplayer ONLY] how many players are fighting inside one game
portlist = [5060 + i for i in range(num_actors)] #One port for each actor inside one player!

# sequence setting
burn_in_steps = 40
learning_steps = 10
forward_steps = 5
seq_len = burn_in_steps + learning_steps + forward_steps

# network setting
use_dueling = True
hidden_dim = 512
cnn_out_dim = 1024

render = False
save_plot = True
test_epsilon = 0.01
