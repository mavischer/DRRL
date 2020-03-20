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
import time
import pandas as pd

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
    config["n_cpus"] = mp.cpu_count() -1
N_W = config["n_cpus"]
# g_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
g_device = torch.device("cpu") #gpu doesn't make sense here because all it does is basically run optimizer on
# received gradients
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
    def __init__(self, g_net, stats_q, grads_q, w_idx, device=l_device, verbose=False):
        """
            Args:
                gnet:   global network that performs parameter updates
                stats_q:queue to put statistics of sampled trajectory
                grads_q:queue to put gradients
                w_idx:  integer index of worker process for identification
                device: assigned device
                verbose:whether to print results of current game (better disable for large number of workers)
        """
        super(Worker, self).__init__()
        self.name = f"w{w_idx:02}" #overwrites Processes' name
        self.g_net = g_net
        self.stats_q = stats_q
        self.grads_q = grads_q
        self.l_net = DRRLnet(INP_W, INP_H, N_ACT).to(device)  # local network
        self.l_net.train()  # sets net in training mode so gradient's don't clutter memory
        print(f"{self.name}: running local net on {l_device}")
        self.env = gym.make('gym_boxworld:boxworld-v0', **ENV_CONFIG)
        self.device = device
        self.verbose = verbose

    def run(self):
        """Runs an entire episode, calculates gradients for all weights

        Writes to stats_queue the accumulated returns (not discounted), number of environment steps and loss of sampled
        episode.
        Write to grads_queue the gradients are written directly to the central learner's parameters grads.
        """
        ### sampling trajectory
        while start_cond.wait(1000): #wait for background process to signal start of an episode (if timeout reached
            # wait returns false and run is aborted
            # print(f"{self.name}: starting iteration")
            t_start = time.time()
            self.pull_params()
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
            t_sample = time.time()
            print(f"{self.name}: sampling took {t_sample-t_start:.2f}s")
            ### forward and backward pass of entire episode
            # preprocess trajectory
            self.l_net.zero_grad()
            self.l_net.train()

            s_,a_,r_disc = self.prettify_trajectory(s_,a_,r_)
            p_, v_ = self.l_net.forward(s_)

            #backward pass to calculate gradients
            loss = self.a2c_loss(s_,a_,r_disc,p_, v_)
            loss.backward()
            t_grads = time.time()
            print(f"{self.name}: calculating gradients took {t_grads-t_sample:.2f}s")

            ### shipping out gradients to centralized learner as named dict
            grads = []
            for name, param in self.l_net.named_parameters():
                grads.append((name, param.grad))
            grad_dict = dict(grads)
            t_end = time.time()

            self.stats_q.put({"cummulative reward": ep_r,
                              "loss": loss.item(),
                              "success": (r==self.env.reward_gem),
                              "steps": ep_t + 1,
                              "walltime": t_end-t_start})
            self.grads_q.put(grad_dict)
            print(f"{self.name}: distributing gradients took {t_end-t_grads:.2f}s")
            print(f"{self.name}: episode took {t_end-t_start}s")

    def prettify_trajectory(self, s_, a_, r_):
        """Prepares trajectory to compute loss on, just to make the code clearer

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

        Returns: Summed losses of trajectory
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


def save_step(i_step, g_net, stats):
    """Saves statistics to global var SAVEPATH and cleans up outdated save files
    Args:
        i_step: iteration step
        g_net: global net's state dictionary containing all variables' values
        stats: list of dicts to be saved to disc as pandas dataframe
    """
    try:
        ending = f"{i_step:05}"
        torch.save(g_net.state_dict(), os.path.join(SAVEPATH, "net"+ending))
        pd.DataFrame(stats).to_csv(os.path.join(SAVEPATH, "stats"+ending))
        #remove old files
        oldfiles = [f for f in os.listdir(SAVEPATH)
                    if (f.startswith("g_net") or f.startswith("stats"))
                    and not f.endswith(ending)]
        for f in oldfiles:
            os.remove(os.path.join(SAVEPATH, f))
    except Exception as e:
        print(f"failed to write step {i_step} to disk:")
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
    # mp.set_start_method("fork") #fork is unix default and means child process inherits all resources from parent
    # process. in case problems occur, might use "forkserver"
    #create global network and pipeline
    g_net = DRRLnet(INP_W, INP_H, N_ACT).to(g_device) # global network
    g_net.zero_grad()
    g_net.share_memory()  # share the global parameters in multiprocessing #todo: check whether this makes a difference
    stats_queue = mp.SimpleQueue() #statistics about the episodes will be returned in this queue
    grads_queue = mp.SimpleQueue() #the calculated gradients will be returned as dicts in this queue
    start_cond = mp.Event() #condition object to signal processes to perform another iteration    # iteration
    # so worker process needs to be still alive when queue is accessed)
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

    #create workers
    stats = []
    trajectories = []
    workers = [Worker(g_net, stats_queue, grads_queue, i) for i in range(N_W)]
    [w.start() for w in workers]  # workers will write the gradients to the parameters directly
    # [w.pull_params() for w in workers] #make workers identical copies of global network before training begins
    for i_step in range(N_STEP): #performing one parallel update step
        t0 = time.time()
        ###parallel trajectory sampling and gradient computation
        start_cond.set() # all processes start an iteration
        time.sleep(0.1)
        start_cond.clear() # this will halt processes' run method at the end of the current episode

        ###copying gradients to global net (also saving statistics)
        g_net.zero_grad()
        for i_w in range(N_W):
            grad_dict = grads_queue.get()
            for name, param in g_net.named_parameters():
                try:
                    param.grad += grad_dict[name]
                except TypeError:
                    param.grad = grad_dict[name] #at the very beginning, gradients are initialized to None
            stats_curr = stats_queue.get()
            stats_curr["global ep"] = i_step #append current global step to dictionary
            stats.append(stats_curr)

        # ### copying gradients and perform optimizer step on global network
        # while not grads_queue.empty():
        #     grad_dict = grads_queue.get()
        #     for name, param in g_net.named_parameters():
        #         try:
        #             param.grad += grad_dict[name]
        #         except TypeError:
        #             param.grad = grad_dict[name] #at the very beginning, gradients are initialized to None
        ### centralized optimizer step
        optimizer.step()  # centralized optimizer updates global network
        #bookkeeping
        if i_step%1 == 0: #save global network
            save_step(i_step, g_net, stats)
        t1 = time.time()
        print(f"{time.strftime('%a %d %b %H:%M:%S', time.gmtime())}: iteration {i_step}: {t1-t0:.1f}s")

    [w.terminate() for w in workers]

    if config["plot"]:
        import matplotlib.pyplot as plt
        import seaborn as sns
        data = pd.DataFrame(stats)
        for i,measure in enumerate(["cummulative reward", "loss", "steps"]):
            plt.figure()
            sns.lineplot(x="global ep",y=measure,data=data)

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
