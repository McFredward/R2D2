
game_name = 'CartPole'
env_type = '-v1'
pretrain = "" #name of the pretrain file in the root. No pretrain if empty string
save_dir = 'models/cartpole/'
frame_stack = 4
obs_shape = (frame_stack, 84, 84) #same as VizDOOM <-- GEN
frame_skip = 1 #TODO: Angucken | doch ganz interessant?


lr = 1e-4 # <-- GEN
eps = 1e-3 #Adam optimzer epsilon #TODO: Warum so gross | Adam angucken?
grad_norm = 40 #maximum value of the total gradient norm, otherwise gradients will be clipped #TODO: Angucken
batch_size = 128 # <-- GEN
learning_starts = 100#50000
save_interval = 100
target_net_update_interval = 200 # <--GEN
gamma = 0.997 #Gamme in goal Gleichung | #TODO: Einlesen | GEN?!

#Prioritized Replaybuffer
prio_exponent = 0#0.9 #How much Prioritization should be used (alpha) | 0 for no Prioritized replay <-- GEN
importance_sampling_exponent = 0#Bias regularization because of Prioritization | 0.6 <--GEN

training_steps = 5000
buffer_capacity = 50000 #<-- GEN
max_episode_steps = 18000
actor_update_interval = 40 #<-- Vielleicht GEN
block_length = 400  # cut one episode to numbers of blocks to improve the buffer space utilization

amp = False # mixed precision training

num_actors = 2#10 # <-- NOT GEN
base_eps = 0.4 #epsilon-greedy-strategy | TODO: Mal angucken ob sinnvoll als GEN| Fix oder angepasst?
alpha = 7 #for calculating a starting epsilon for each actor
log_interval = 20

#Multiplayer related
multiplayer = False
num_players = 2 # [Multiplayer ONLY] how many players are fighting inside one game | How many R2D2's
portlist = [5060 + i for i in range(num_actors)] #One port for each actor inside one player!

# sequence setting
burn_in_steps = 40 # <-- GEN
learning_steps = 10 # <-- GEN
forward_steps = 5 # TODO: Angucken
seq_len = burn_in_steps + learning_steps + forward_steps

# network setting
use_dueling = True #<-- GEN
hidden_dim = 512 #<-- GEN
cnn_out_dim = 1024 #<-- GEN

render = False
save_plot = True
test_epsilon = 0.01

