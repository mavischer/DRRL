import numpy as np
import torch
import os
import argparse
import yaml
import time
import csv
from collections import deque

from helpers.a2c_ppo_acktr import algo, utils
from helpers.a2c_ppo_acktr.envs import make_vec_envs
from helpers.a2c_ppo_acktr.model import Policy, DRRLBase
from helpers.a2c_ppo_acktr.storage import RolloutStorage
from helpers.lr_scheduling import Linear_decay
# from baselines.common import plot_util

# if "e_schedule" in config.keys(): #todo: implement entropy weight scheduling
#     e_schedule = config["e_schedule"]
# else:
#     e_schedule = False

def main():
    # parse yaml config file from cmdline
    parser = argparse.ArgumentParser(description='PyTorch A2C BoxWorld Experiment')
    parser.add_argument("-c", "--configpath", type=str, required=True, help="path/to/configfile.yml")
    parser.add_argument("-s", "--savepath", type=str, required=True, help="path/to/savedirectory")
    args = parser.parse_args()
    with open(os.path.abspath(args.configpath), 'r') as file:
        config = yaml.safe_load(file)

    SAVE_EVERY = 1000
    LOG_EVERY = 100

    # todo: implement adam optimizer?

    #set up torch
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    torch.set_num_threads(config["n_cpus"]) #intra-op parallelism
    device = torch.device("cuda:0" if config["cuda"] and torch.cuda.is_available() else "cpu")

    #set up logging
    log_dir = os.path.expanduser(os.path.join(args.savepath, "logs"))
    utils.cleanup_log_dir(log_dir)
    save_path = os.path.join(args.savepath, "saves")

    save_stats_path = os.path.join(log_dir, "training_losses.csv")

    #make environments in vectorizer wrapper sharing memory
    envs = make_vec_envs((config["env_name"], config["env_config"]), config["seed"], config["n_cpus"],
                         config["gamma"], log_dir, device, False, num_frame_stack=1)  # default is stacking 4 frames

    #load from startpoint
    modelckptpath = os.path.join(save_path, "ckpt.pt")
    if os.path.isfile(modelckptpath):
        #check whether configs are identical
        # load config and check whether identical
        with open(os.path.join(args.savepath, "config.yml"), 'r') as file:
            config_old = yaml.safe_load(file)
        if config_old != config:
            raise Exception("Existing config different from current config")
        #load iteration, algo and agent
        [start_upd, actor_critic, agent] = torch.load(modelckptpath)
        print(f"loaded from savepoint {start_upd} in folder {modelckptpath}")

    #or start fresh
    else:
        #start new entropy logging file
        with open(save_stats_path, "w") as f:
            f.write("value loss,action loss,entropy\n")

        # write config to new directory
        with open(os.path.join(args.savepath, "config.yml"), "w+") as f:
            f.write(yaml.dump(config))

        # start new training process
        stats = []
        i_start = 0
        print("starting new training process")
        start_upd = 0
        base_kwargs = config["net_config"]
        base_kwargs["w"] = base_kwargs["h"] = config["env_config"]["n"]+2 #+2 is for black edge

        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base=DRRLBase,
            base_kwargs=base_kwargs)
        actor_critic.to(device)

        #set up linear learning rate decay
        if config["lr_decay"]:
            ep_max = 3e8 / (config["n_cpus"] * config["update_every_n_steps"])
            # ep_max = config["n_env_steps"] / (config["n_cpus"] * config["update_every_n_steps"])
            if config["lr_term"]:
                lr_term = config["lr_term"]
            else:
                lr_term = 1e-5
            lr_sched_fn = Linear_decay(lr_init=config["lr"], lr_term=lr_term, ep_max=ep_max)
        else:
            lr_sched_fn = None

        agent = algo.A2C_ACKTR(
            actor_critic,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            lr=config["lr"],
            lr_decay=config["lr_decay"],
            lr_sched_fn=lr_sched_fn,
            eps=1e-5,
            alpha=0.99, #RMSProp optimizer alpha
            max_grad_norm=0.5) #max norm of grads

    rollouts = RolloutStorage(config["update_every_n_steps"], config["n_cpus"],
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    loss_stats = []

    start = time.time()
    num_updates = int(config["n_env_steps"]) // config["update_every_n_steps"] // config["n_cpus"]
    for j in range(start_upd, num_updates): #main training loop: global iteration counted in weight updates according
        # to num_steps number of environment steps used for each update

        for step in range(config["update_every_n_steps"]): #a batch update of num_steps for each num_process will be created in this
            # loop
            with torch.no_grad():             # Sample actions
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r']) #todo: is this needed?

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad(): #get value at end of look-ahead
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, use_gae=False, gamma=config["gamma"], gae_lambda=None)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        loss_stats.append([value_loss, action_loss, dist_entropy])

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % SAVE_EVERY == 0
                or j == num_updates - 1) and save_path != "":
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([j, actor_critic, agent], os.path.join(modelckptpath))

            #write and clear loss_stats
            with open(save_stats_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(loss_stats)
            loss_stats = []

        if j % LOG_EVERY == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * config["n_cpus"] * config["update_every_n_steps"]
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards)))

if __name__ == "__main__":
    main()
