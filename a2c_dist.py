"""Version of a2c.py to calculate gradients in local worker instances and send them to gobal optimizer.  This way,
no GPU is required at the cost of slightly longer steps."""
import torch
import torch.multiprocessing as mp
import gym
from attention_module import DRRLnet
from torch.distributions import Categorical
import os
import argparse
import yaml
from time import time

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
    def __init__(self, g_net, g_lock, w_idx, device=l_device, verbose=False):
        """
            Args:
                gnet:   global network that performs parameter updates
                w_idx:  integer index of worker process for identification
                device: assigned device
                verbose:whether to print results of current game (better disable for large number of workers)
        """
        super(Worker, self).__init__()
        self.name = f"w{w_idx:02}"
        self.g_net = g_net
        self.g_lock = g_lock
        self.l_net = DRRLnet(INP_W, INP_H, N_ACT).to(device)  # local network
        self.l_net.train()  # sets net in training mode so gradient's don't clutter memory
        print(f"{self.name}: running local net on {l_device}")
        self.env = gym.make('gym_boxworld:boxworld-v0', **ENV_CONFIG)
        self.device = device
        self.verbose = verbose

    def start(self):
        """Runs an entire episode, calculates gradients for all weights

        Returns:
            loss and accumulated returns (not discounted) of sampled episode. the gradients are written directly to
            the central learner's parameters grads
        """
        ### sampling trajectory
        t0 = time()
        self.l_net.eval()
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
                break
            s = s_new
            ep_t += 1

        print(f"{self.name:} trajectory complete")
        ### forward and backward pass of entire episode
        # preprocess trajectory
        self.l_net.zero_grad()
        self.l_net.train()

        s_,a_,r_disc = self.prettify_trajectory(s_,a_,r_)
        t1 = time()
        print(f"{self.name:} trajectory complete ({t1-t0:.2f}s)")
        t2 = time()
        p_, v_ = self.l_net.forward(s_)
        print(f"{self.name:} forward pass complete ({t2-t1:.2f}s)")

        #backward pass to calculate gradients
        t3 = time()
        loss = self.a2c_loss(s_,a_,r_disc,p_, v_)
        print(f"{self.name:} loss computed ({t3-t2:.2f}s)")
        t4 = time()
        loss.backward()
        print(f"{self.name:} backward pass complete ({t4-t3:.2f}s)")

        ### shipping out gradients to centralized learner
        #acquire lock on global net, needed because += is not atomic
        self.g_lock.acquire()

        t5 = time()
        print(f"{self.name:} lock acquired ({t5-t4:.2f}s)")
        for lp, gp in zip(self.l_net.parameters(), self.g_net.parameters()):
            try:
                gp.grad += lp.grad #apparently in older torch versions, .grad was read-only, so you see ._grad for write
            except TypeError as e: #the very first time these gradients are touched, they are None, afterwards zero
                gp.grad = lp.grad
        t6 = time()
        print(f"{self.name:} gradients written ({t6-t5:.2f}s)")
        t7 = time()
        self.g_lock.release()
        print(f"{self.name:} lock released ({t7-t6:.2f}s)")

        return(loss.item(), ep_r)

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

    def a2c_loss(self, s_,a_,r_disc,p_, v_):
        """Calculate advantage-actor-critic loss on entire episode
        Args:
            for the entire trajectory, one tensor each of
            s_: states
            a_: actions
            r_disc: temporally discounted future rewards
            p_: action probabilities
            v_: value estimates

        Returns: Summed of losses of trajectory
        """
        td = r_disc - v_
        m = torch.distributions.Categorical(p_)
        # e_w = min(1, 2*0.995**opt_step) #todo: try out entropy annealing!
        e_w = 0.005  # like in paper
        total_loss = (0.5 * td.pow(2) - m.log_prob(a_) * td.detach().squeeze() + e_w * m.entropy()).mean()

        return total_loss.sum()

    def pull_params(self):
        """Update own params from global network."""
        self.l_net.load_state_dict(self.g_net.state_dict(), strict=True)


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
    g_net.zero_grad()

    g_net.share_memory()  # share the global parameters in multiprocessing #todo: check whether this makes a difference
    g_lock = mp.Lock()
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
    workers = [Worker(g_net, g_lock, i) for i in range(N_W)]

    [w.pull_params() for w in workers] #make workers identical copies of global network before training begins
    for i_step in range(N_STEP): #performing one parallel update step
        # optimizer.zero_grad()
        #parallel trajectory sampling and gradient computation
        episodes = [w.start() for w in workers] #workers will write the gradients to the parameters
        ep_losses, cum_rewards = zip(*episodes)
        #perform optimizer step on global network
        optimizer.step()  # centralized optimizer updates global network
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
        losses.append(sum(ep_losses))
        rewards.append(sum(cum_rewards)/N_W)
        # print(f"{time.strftime('%a %d %b %H:%M:%S', time.gmtime())}: it: {i_step}, steps:{len_iter}, "
        #       f"cum. steps:{g_step}, total loss:{losses[-1]:.2f}, avg. reward:{rewards[-1]:.2f}.")
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

