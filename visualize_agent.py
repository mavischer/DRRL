# Script to load a trained policy, play an example game and visualize attentional weights
import argparse
import os
import yaml
import torch
import gym
import matplotlib.pyplot as plt
import numpy as np

from helpers.a2c_ppo_acktr import algo, utils
from helpers.a2c_ppo_acktr.envs import make_vec_envs
from helpers.a2c_ppo_acktr.model import Policy, DRRLBase
from helpers.a2c_ppo_acktr.storage import RolloutStorage

plt.ioff() #only plot when asked explicitly

parser = argparse.ArgumentParser(description='PyTorch A2C BoxWorld Agent Visualization')
parser.add_argument("-s", "--savepath", type=str, required=True, help="path/to/savedirectory")
parser.add_argument("-i", "--imagepath", type=str, required=True, help="path/to/save/images")
args = parser.parse_args()

#create target directory
if not os.path.exists(args.imagepath):
    try:
        os.makedirs(args.imagepath)
    except OSError:
        print('Error: Creating images target directory. ')
#load config
with open(os.path.join(args.savepath, "config.yml"), 'r') as file:
    config = yaml.safe_load(file)
net_config = config["net_config"]
#load agent
[start_upd, actor_critic, agent] = torch.load(os.path.join(args.savepath, "saves", "ckpt.pt"))
actor_critic.eval()

#create environment
env = gym.make(config["env_name"], **config["env_config"])

#play environment
done = False
i_step = 0
img = env.reset()
xsize = img.shape[0]
ysize = img.shape[1]
while not done:
    obs = torch.tensor([np.moveaxis(img, -1, 0)], dtype=torch.uint8) #todo: seems like vecEnv permutes the image
    # correctly
    _, action, _, _ = actor_critic.act(obs, None, None)
    att_weights = actor_critic.base.get_attention_weights(obs) #att_weights is a list of lists, outer level contains
    # stacks of attention module, inner level heads

    fig = plt.figure(figsize=(18.8, 9), constrained_layout=False)
    # fig = plt.figure(figsize=(9, 9), constrained_layout=False)
    black_img = np.zeros(img.shape)

    # gridspec inside gridspec
    outer_grid = fig.add_gridspec(net_config["n_att_stack"], net_config["n_heads"], wspace=0.1, hspace=0.1)
    for i_head in range(net_config["n_heads"]):
        for i_stack in range(net_config["n_att_stack"]):
            #extract current weight map
            weightmap_curr = att_weights[i_stack][i_head].numpy().squeeze()  # map for specific stack and head

            #PLOTTING OVERHEAD
            #set up outer box corresponding to stack x head
            ax = fig.add_subplot(outer_grid[i_stack, i_head])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"head{ i_head}, pass {i_stack}")
            #all weights for specific stack number, head go inside as a subgridspec
            inner_grid = outer_grid[i_stack, i_head].subgridspec(xsize, ysize, wspace=0.0, hspace=0.0)

            #now loop over all source (query) entities:
            for ent_i in range(weightmap_curr.shape[0]): #index ent_i is single dim (e.g. entity 32 in a 7x7 grid)
                w_max = np.max(weightmap_curr[ent_i, :])  # memorize maximum weight. btw: this vector sums to 1
                x_s, y_s = np.unravel_index(ent_i, img.shape[:-1]) #translate to 2d idx for plotting

                # prepare grid
                ax = fig.add_subplot(inner_grid[x_s, y_s]) #inner_grid also has matrix-convention x,y-arrangement,
                # so surprisingly we don't need to revert the indices here!
                # ax.annotate(f"x:{x_s},y:{y_s}", (1,3)) #for bugfixing
                ax.set_xticks([])
                ax.set_yticks([])

                if (x_s==0 and y_s!=0) or (x_s!=0 and y_s==0) or x_s==img.shape[0]-1 or y_s==img.shape[1]-1:
                    #source entity is on black border
                    ax.imshow(black_img, vmin=0, vmax=255, interpolation="none")
                else:
                    ax.imshow(img, vmin=0, vmax=255, interpolation="none")

                    #loop over target (key) entitites:
                    for ent_j in range(weightmap_curr.shape[1]):
                        weight = weightmap_curr[ent_i, ent_j]
                        if weight > 0.5 * w_max:
                            x_t, y_t = np.unravel_index(ent_j, img.shape[:-1])
                            ax.scatter(y_t, x_t, s=3, c='red', marker='o')
                            # ax.scatter(y_s, x_s, s=5, c='blue', marker='o')


                            ax.arrow(y_s, x_s, y_t-y_s, x_t-x_s, #DUE TO IMSHOW AXES ARE INVERTED!!!
                                     length_includes_head=True,
                                     head_width=0.2,
                                     head_length=0.3,
                                     alpha=weight*5)# / w_max)
    #save figure
    plt.savefig(os.path.join(args.imagepath, 'frame_{}.png'.format(i_step)))

    #next step
    img, _, done, _ = env.step(action.item())
    obs = torch.tensor([np.moveaxis(img, -1, 0)], dtype=torch.uint8)
    i_step += 1

#
#
# #next step
# img, _, done, _ = env.step(action.item())
# obs = torch.tensor([np.moveaxis(img, -1, 0)], dtype=torch.uint8)
#
# _, action, _, _ = actor_critic.act(obs, None, None)
# att_weights = actor_critic.base.get_attention_weights(obs) #att_weights is a list of lists, outer level contains
#
# #example 1,3:
# xcoor = 5
# ycoor = 2
# # threshld = 5/(7*7)
# flat_idx = np.ravel_multi_index((xcoor, ycoor), [7,7])
# weights_1 = att_weights[0][0][0,flat_idx,:].numpy().squeeze()
# weights_2 = att_weights[0][1][0,flat_idx,:].numpy().squeeze()
# w_max = np.max(weights_1)
# target_idxs = [(idx, val) for (idx, val) in enumerate(weights_1) if val > w_max/3]
#
# plt.subplot(1,2,1)
# plt.imshow(img)
# for target_idx, val in target_idxs:
#     target_idx = np.unravel_index(target_idx, [7,7])
#     plt.arrow(xcoor, ycoor, target_idx[0]-xcoor, target_idx[1]-ycoor,
#               length_includes_head=True,
#               head_width=0.2,
#               head_length=0.3,
#               alpha=val/w_max)
# plt.subplot(1,2,2)
# plt.imshow(weights_1.reshape([7,7]).T)
#
#
#
#
# flat_idx = np.ravel_multi_index((xcoor, ycoor), [7,7])
# weights_1 = att_weights[0][0][0,flat_idx,:].reshape([7,7]).numpy()
# weights_2 = att_weights[0][1][0,flat_idx,:].reshape([7,7]).numpy()
# img = np.moveaxis(obs.numpy().squeeze(), 0,-1)
#
#
# target_idx = np.unravel_index(np.argmax(weights_1), [7,7])
# plt.subplot(1,2,1)
# plt.imshow(img)
#
# plt.arrow(xcoor, ycoor, target_idx[0]-xcoor, target_idx[1]-ycoor,
#           length_includes_head=True,
#           head_width=0.2,
#           head_length=0.3,
#           alpha=1)
# plt.subplot(1,2,2)
# plt.imshow(weights_1)
#
#
# # a = obs.numpy().squeeze().T
# plt.subplot(2,2,1)
# plt.imshow(a)
# plt.subplot(2,2,2)
# plt.imshow(weights_1)
# plt.subplot(2,2,4)
# plt.imshow(weights_2)





    # fig2 = plt.imshow(img, vmin=0, vmax=255, interpolation='none')
    # fig2.axes.get_xaxis().set_visible(False)
    # fig2.axes.get_yaxis().set_visible(False)
    # plt.savefig(os.path.join('images', 'observation_{}_{}.png'.format(i_episode, t)))
    #
    # #next step
    # img, _, done, _ = env.step(action.item())
    # obs = torch.tensor([img.T], dtype=torch.uint8)
    # 
    # i_episode += 1


