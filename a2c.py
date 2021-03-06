"""Base version of advantage actor critic training of the DRRL architecture on a
BoxWorld task. Can be used with local actor instances on CPUS and global learner instance on GPU.
Make sure to have the gym-boxworld environment registered: https://github.com/mavischer/Box-World
Script made with, among others, inspiration from https://github.com/MorvanZhou/pytorch-A3C/ and
https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html.
"""
import torch
import torch.multiprocessing as mp
import gym
from attention_module import DRRLnet
from torch.distributions import Categorical
import os
import argparse
import yaml
import time

#parse yaml config file from cmdline
parser = argparse.ArgumentParser(description='PyTorch A2C BoxWorld Experiment')
parser.add_argument("-c", "--configpath", type=str, required=True, help="path/to/configfile.yml")
parser.add_argument("-s", "--savepath", type=str, required=True, help="path/to/savedirectory")
args = parser.parse_args()
with open(os.path.abspath(args.configpath), 'r') as file:
        config = yaml.safe_load(file)
SAVEPATH = args.savepath

#set stage
if not os.path.isdir(SAVEPATH):
    os.mkdir(SAVEPATH)
torch.manual_seed(config["seed"])
ENV_CONFIG = config["env_config"]
if config["n_cpus"] == -1:
    config["n_cpus"] = mp.cpu_count()
N_W = config["n_cpus"]
g_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running global net on {g_device}")
l_device = torch.device("cpu")
with open(os.path.join(SAVEPATH, "config.yml"), "w+") as f:
    f.write(yaml.dump(config))

#make environment
env = gym.make('gym_boxworld:boxworld-v0', **ENV_CONFIG)
N_ACT = env.action_space.n
INP_W = env.observation_space.shape[0]
INP_H = env.observation_space.shape[1]

#configure learning
N_STEP = config["n_step"]
GAMMA = config["gamma"]

# #todo: check whether we really need this?
# class SharedAdam(torch.optim.Adam):
#     """Shared optimizer, the parameters in the optimizer will be shared across multiprocessors."""
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.9), eps=1e-8,
#                  weight_decay=0):
#         super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
#         # State initialization
#         for group in self.param_groups:
#             for p in group['params']:
#                 state = self.state[p]
#                 state['step'] = 0
#                 state['exp_avg'] = torch.zeros_like(p.data)
#                 state['exp_avg_sq'] = torch.zeros_like(p.data)
#
#                 # share in memory
#                 state['exp_avg'].share_memory_()
#                 state['exp_avg_sq'].share_memory_()


class Worker(mp.Process):
    def __init__(self, gnet, w_idx, device=l_device, verbose=False):
        """
            Args:
                gnet:   global network that performs parameter updates
                w_idx:  integer index of worker process for identification
                device: assigned device
                verbose:whether to print results of current game (better disable for large number of workers)
        """
        super(Worker, self).__init__()
        self.name = f"w{w_idx:02}"
        self.g_net = gnet
        self.l_net = DRRLnet(INP_W, INP_H, N_ACT).to(device)  # local network
        self.l_net.eval() #otherwise gradients clutter memory
        print(f"{self.name}: running local net on {l_device}")
        self.env = gym.make('gym_boxworld:boxworld-v0', **ENV_CONFIG)
        self.device = device
        self.verbose = verbose

    def start(self):
        """Runs an entire episode, in the end return trajectory as list of s,a,r.

        Returns:
            Ready-to-digest trajectory as torch.tensors of state, action, reward+discounted future rewards
        """
        s = self.env.reset()
        s_, a_, r_ = [], [], [] #trajectory of episode goes here
        ep_r = 0. #total episode reward
        ep_t = 0  #episode step t, both just for oversight
        while True: #generate variable-length trajectory in this loop
            s = torch.tensor([s.T], dtype=torch.float, device=self.device)  # transpose for CWH-order, apparently
            # conv layer want floats
            p, _ = self.l_net(s)
            m = Categorical(p) # create a categorical distribution over the list of probabilities of actions
            a = m.sample().item() # and sample an action using the distribution
            s_new, r, done, _ = self.env.step(a)
            ep_r += r

            # append current step's elements to lists
            s_.append(s)
            a_.append(a)
            r_.append(r)

            if done:  # return trajectory as lists of elements
                if self.verbose:
                    print(f"{self.name}: episode ended after step {ep_t} with total reward {ep_r}")
                return(self.prettify_trajectory(s_, a_, r_), ep_r)
            s = s_new
            ep_t += 1

    def prettify_trajectory(self, s_, a_, r_):
        """Prepares trajectory to be sent to central learner in a more orderly fashion.

        Calculate temporally discounted future rewards for TD error and reverse lists so everything is in correct
        temporal order. Cast everything to appropriate tensorflow tensors. Return what's necessary to compute
        gradients upstream.

        Args:
            s_: List of states
            a_: List of actions
            r_: List of returns

        Returns:
            Ready-to-digest trajectory as torch.tensors of state, action, reward+discounted future rewards

        """
        # calculate temporal-discounted rewards
        r_acc = 0
        r_disc = []
        for r in r_[::-1]:  # reverse buffer r, discount accordingly and add in value at for t=t_end
            r_acc = r + GAMMA * r_acc
            r_disc.append(r_acc)  # discounted trajectory in reverse order
        r_disc = r_disc[::-1]
        # every element in r_disc now contains reward at corresponding step plus future discounted rewards

        #cast everything to tensors (states are already cast)
        s_ = torch.cat(s_).to(device=g_device).detach()
        a_ = torch.tensor(a_, dtype=torch.uint8, device=g_device).detach() #torch can only compute gradients for float
        # tensors, but this shouldn't be a problem
        r_disc = torch.tensor(r_disc, dtype=torch.float16, device=g_device).detach()

        return(s_,a_,r_disc)

    def pull_params(self):
        """Update own params from global network."""
        self.l_net.load_state_dict(self.g_net.state_dict(), strict=True)

def update_step(net, trajectories, opt, opt_step):
    """Calculate advantage-actor-critic loss on batch with network, updates parameters
    Args:
        net: network to perform training on
        batch: list of trajectories as tuple of 1 tensor object for each
            b_s:    states
            b_a:    chosen actions
            b_r_disc: returns and discounted future rewards for TD error with value function
    Returns: Sum of loss of all trajectories
    """
    rezip = zip(*trajectories)
    b_s, b_a, b_r_disc = [torch.cat(elems) for elems in list(rezip)] #concatenate torch tensors of all trajectories
    try:
        b_p, b_v = net.forward(b_s)
    except RuntimeError:
        print(f"failed to handle batch of size {b_s.shape}")
        raise

    td = b_r_disc - b_v
    m = torch.distributions.Categorical(b_p)
    # e_w = min(1, 2*0.995**opt_step) #todo: try out entropy annealing!
    e_w = 0.005  # like in paper
    total_loss = (0.5 * td.pow(2) - m.log_prob(b_a) * td.detach().squeeze() + e_w * m.entropy()).mean()

    opt.zero_grad()
    total_loss.backward()
    opt.step()
    return total_loss.sum()

def save_step(i_step, g_net, steps, losses, rewards):
    """Saves statistics to global var SAVEPATH and cleans up outdated save files
    Args:
        i_step: iteration step
        g_net: global net's state dictionary containing all variables' values
        steps: global number of environment steps (all workers combined)
        losses: global loss of episodes
        rewards: average rewards of episodes
    """
    try:
        ending = f"{i_step:05}.pt"
        for name, var in zip(["g_net", "steps", "losses", "rewards"], [g_net.state_dict(), steps, losses, rewards]):
            torch.save(var, os.path.join(SAVEPATH, name+ending))
    except Exception as e:
        print(f"failed to write step {i_step} to disk:")
        print(e)
    try:
        # clean out old files
        oldfiles = [f for f in os.listdir(SAVEPATH)
                    if (f.startswith("g_net") or f.startswith("steps") or
                        f.startswith("losses") or f.startswith("rewards"))
                    and not f.endswith(ending)]
        for f in oldfiles:
            os.remove(os.path.join(SAVEPATH, f))
    except Exception as e:
        print("failed to erase old saves:")
        print(e)

def load_step():
    """Loads statistics from global var SAVEPATH and loads g_nets parameters from saved state_dict
    Returns:
        list of loaded variables
            g_net: global net's state dictionary containing all variables' values
            steps: global number of environment steps (all workers combined)
            losses: global loss of episodes
            rewards: average rewards of episodes
    """
    loaded_vars = []
    for name in ["g_net", "steps", "losses", "rewards"]:
        files = [file for file in os.listdir(SAVEPATH) if file.startswith(name)]
        if len(files) > 1:
            raise Exception(f"more than one savefile found for {name}")
        else:
            loaded_vars.append(torch.load(os.path.join(SAVEPATH, files[0])))
    return(loaded_vars)

if __name__ == "__main__":
    #create global network and pipeline
    g_net = DRRLnet(INP_W, INP_H, N_ACT).to(g_device) # global network
    #todo: only implicit init so far
    g_net.share_memory()  # share the global parameters in multiprocessing #todo: check whether this makes a difference
    # optimizer = SharedAdam(g_net.parameters(), lr=0.0001)  # global optimizer
    if config["optimizer"] == "RMSprop":
        #RMSprop optimizer was used for the large state space, not the small ones and impala instead of a3c.
        # "Learning rate was tuned between 1e-5 and 2e-4" probably means they did hyperparameter search.
        # scheduling is also possible conveniently using torch torch.optim.lr_scheduler
        # perhaps use smaller decay term 0.9
        optimizer = torch.optim.RMSprop(g_net.parameters(), eps=0.1, lr=config["lr"])
    else:
        #Adam optimizer was used for the starcraft games with learning rate decaying linearly over 1e10 steps from
        # 1e-4 to 1e-5. other params are torch defaults
        optimizer = torch.optim.Adam(g_net.parameters(),lr=config["lr"])
    g_step = 0

    #create workers
    losses = []
    steps = []
    rewards = []
    trajectories = []
    workers = [Worker(g_net, i) for i in range(N_W)]

    [w.pull_params() for w in workers] #make workers identical copies of global network before training begins
    for i_step in range(N_STEP): #performing one parallel update step
        #parallel trajectory sampling
        episodes = [w.start() for w in workers] #list comprehension automatically waits for workers to finish
        trajectories, cum_rewards = zip(*episodes)
        #concatenate and push tracjectories to global network for learning (synchronized update)
        loss = update_step(g_net, trajectories, optimizer, i_step)
        #pull new parameters
        [w.pull_params() for w in workers]
        #trying to free some gpu memory...
        if g_device.type == "cuda": #these only release memory to be visible, should not make a substantial difference
            torch.cuda.empty_cache()
#            torch.cuda.synchronize()
        #bookkeeping
        len_iter = sum([len(traj[0])for traj in trajectories])
        g_step += len_iter
        steps.append(g_step)
        losses.append(loss.item())
        rewards.append(sum(cum_rewards)/N_W)
        print(f"{time.strftime('%a %d %b %H:%M:%S', time.gmtime())}: it: {i_step}, steps:{len_iter}, "
              f"cum. steps:{g_step}, total loss:{loss.item():.2f}, avg. reward:{rewards[-1]:.2f}.")
        if i_step%1 == 0: #save global network
            save_step(i_step, g_net, steps, losses, rewards)

    if config["plot"]:
        import matplotlib.pyplot as plt
        plt.plot(steps, losses)

    if config["tensorboard"]:
        from torch.utils.tensorboard import SummaryWriter
        # create writers
        g_writer = SummaryWriter(os.path.join(SAVEPATH, "tb_g_net"))
        l_writer = SummaryWriter(os.path.join(SAVEPATH, "tb_l_net"))
        # write graph to file
        rezip = zip(*trajectories)
        b_s, _, _ = [torch.cat(elems) for elems in list(rezip)]  # concatenate torch tensors of all trajectories
        g_writer.add_graph(g_net,b_s)
        l_writer.add_graph(workers[0].l_net,b_s)
        g_writer.close()
        l_writer.close()

