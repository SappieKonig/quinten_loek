# ontzettend mooi gedaan heren! De list van lists is bij jullie mooi statisch
# (heeft vanaf het begin al afmetingen gelijk aan nr_episodes). Dit helpt ontzettend bij debugging omdat het sneller
# problemen oplevert als je code niet klopt. Ik heb niks op deze code aan te merken ;)




import gym
import numpy as np
import tensorflow as tf
import random

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

neurons = 32
input1 = tf.keras.layers.Input(4)
X = tf.keras.layers.BatchNormalization()(input1)
X = tf.keras.layers.Dense(neurons, "relu")(X)
X = tf.keras.layers.Dense(neurons, "relu")(X)
output = tf.keras.layers.Dense(1, "sigmoid")(X)  # Sigmoid ipv softmax aangezien we maar 1 output hebben.

agent = tf.keras.models.Model(inputs=[input1], outputs=[output])

for i_episode in range(nr_episodes):
    while 1:
        env.render()
        # TF predict verwacht een lijst van lijsten: dimensies x by 4.
        # inputs = np.expand_dims(inputs, axis=0)
        inputs = [[0, 0, 0, 0]]
        print('inputs', inputs)
        prediction = agent.predict(np.array(inputs))[0][0]
        print('prediction', prediction)
        action = int(random.random() < prediction)
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
