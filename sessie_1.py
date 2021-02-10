# ontzettend mooi gedaan heren! De list van lists is bij jullie mooi statisch
# (heeft vanaf het begin al afmetingen gelijk aan nr_episodes). Dit helpt ontzettend bij debugging omdat het sneller
# problemen oplevert als je code niet klopt. Ik heb niks op deze code aan te merken ;)




import gym
import numpy as np

def decay(rewards, decay_factor):
    """
    e.g.:
    rewards [1, 1, 1]
    decayed_rewards = [0, 0, 0]
    decayed_rewards = [0, 0, 1]
    decayed_rewards = [0, 1 + 0.9 * 1, 1]
    decayed_rewards = [1 + 0.9 * 1.9, 1 + 0.9 * 1, 1]
    """
    decayed_rewards = np.zeros(len(rewards))
    decayed_rewards[-1] = rewards[-1]
    for i in reversed(range(len(rewards) - 1)):
        decayed_rewards[i] = rewards[i] + decay_factor * decayed_rewards[i + 1]
    return decayed_rewards

def decay_and_normalize(total_rewards, decay_factor):
    for i, rewards in enumerate(total_rewards):
        total_rewards[i] = decay(rewards, decay_factor)

    total_rewards = np.concatenate(total_rewards)

    return (total_rewards - np.mean(total_rewards)) / np.std(total_rewards)

env = gym.make("CartPole-v0")
env.reset()
done = False
nr_episodes = 2
observations = [[] for _ in range(nr_episodes)]
rewards = [[] for _ in range(nr_episodes)]
dones = [[] for _ in range(nr_episodes)]
infos = [[] for _ in range(nr_episodes)]

for i_episode in range(nr_episodes):
    while 1:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        observations[i_episode].append(observation)
        rewards[i_episode].append(reward)
        dones[i_episode].append(done)
        infos[i_episode].append(info)
        if done:
            env.reset()
            break

env.close()

print(observations)
print(rewards)
print(dones)

result = decay_and_normalize(rewards, 0.9)

print(result)