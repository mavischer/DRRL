import gym
import numpy as np
import matplotlib.pyplot as plt

N_EP = 3000

env2 = gym.make("gym_boxworld:boxworld-v0", n= 5, goal_length= 2, num_distractor= 1,
               distractor_length= 1, num_colors= 8, max_steps= 1000)
env3 = gym.make("gym_boxworld:boxworld-v0", n= 5, goal_length= 3, num_distractor= 1,
               distractor_length= 1, num_colors= 8, max_steps= 1000)

cum_rewards = []
steps = []
success = []
for env in [env2, env3]:
    for i_ep in range(N_EP):
        #roll out trajectory
        s = env.reset()
        ep_r = 0.  # total episode reward
        ep_t = 0  # episode step t, both just for oversight
        while True:  # generate variable-length trajectory in this loop
            _, r, done, _ = env.step(np.random.choice(4))
            ep_r += r
            ep_t += 1
            if done:  # return trajectory as lists of elements
                cum_rewards.append(ep_r)
                steps.append(ep_t)
                success.append(r==env.step_cost + env.reward_gem)
                break
        if i_ep%100 == 0:
            print(f"completed trajectory {i_ep}")

print("mean cum. reward:", np.mean(cum_rewards))
print("std cum. reward:", np.std(cum_rewards))
print("success ratio:", np.mean(success))
print("avg steps:", np.mean(steps))

plt.hist(cum_rewards, bins=np.arange(-0.5, 13.5, 1))
plt.scatter(np.mean(cum_rewards), 0)



