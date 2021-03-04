# ontzettend mooi gedaan heren! De list van lists is bij jullie mooi statisch
# (heeft vanaf het begin al afmetingen gelijk aan nr_episodes). Dit helpt ontzettend bij debugging omdat het sneller
# problemen oplevert als je code niet klopt. Ik heb niks op deze code aan te merken ;)

# Prachtig om die documentatie te zien! Dan vergeet je iig niet tussendoor waar je nou mee bezig was.
# We gebruiken helaas de langere methode voor het trainen, dus het laatste stukje code moet nog een beetje veranderen.
# Verder heb ik nog 3 enters toegevoegd in de hoop dat het overzichtelijker wordt, kijk zelf maar wat julle fijner vinden.


import gym
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt


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

observation = env.reset()
done = False
reward = 1
info = {}

scores = []

nr_epochs = 99
nr_episodes = 10
learning_rate = 3e-3
batch_size = 16

neurons = 32
input1 = tf.keras.layers.Input(4)
X = tf.keras.layers.BatchNormalization()(input1)
X = tf.keras.layers.Dense(neurons, "relu")(X)
X = tf.keras.layers.Dense(neurons, "relu")(X)
output = tf.keras.layers.Dense(1, "sigmoid")(X)  # Sigmoid ipv softmax aangezien we maar 1 output hebben.

agent = tf.keras.models.Model(inputs=[input1], outputs=[output])
optimizer = tf.keras.optimizers.Adam(learning_rate)  # slaat parameters op tijdens het leren die je wilt bewaren

for i_epoch in range (nr_epochs):
    observations = [[] for _ in range(nr_episodes)]
    rewards = [[] for _ in range(nr_episodes)]
    dones = [[] for _ in range(nr_episodes)]
    infos = [[] for _ in range(nr_episodes)]
    actions = [[] for _ in range(nr_episodes)]
    for i_episode in range(nr_episodes):
        while 1:
            # env.render()
            # TF predict verwacht een lijst van lijsten: dimensies x by 4. Daarom extra haken toevoegen.
            prediction = agent.predict(np.array([observation]))[0][0]
            action = int(random.random() < prediction)
            # Voeg juiste reward etc bij actie toe
            observations[i_episode].append(observation)
            rewards[i_episode].append(reward)
            dones[i_episode].append(done)
            infos[i_episode].append(info)
            actions[i_episode].append(action)
            # update observation etc.
            observation, reward, done, info = env.step(action)
            if done:
                env.reset()
                break

    env.close()

    decayed_rewards = decay_and_normalize(rewards, 0.9)

    rewards = np.concatenate(rewards)
    rewards = np.expand_dims(rewards, axis=1)
    observations = np.concatenate(observations)
    dones = np.concatenate(dones)
    dones = np.expand_dims(dones, axis=1)
    actions = np.concatenate(actions)
    actions = np.expand_dims(actions, axis=1)

    aantal_stappen = len(actions)
    score = aantal_stappen // nr_episodes
    print('Gemiddeld aantal stappen is:', score)
    scores += [score]

    # for idx in range(aantal_stappen // batch_size):
    #     batch_actions = actions[0+1*idx:batch_size * idx]
    aantal_batches = np.ceil(aantal_stappen / batch_size)
    batch_actions = np.array_split(actions, aantal_batches)
    batch_observations = np.array_split(observations, aantal_batches)
    batch_decayed_rewards = np.array_split(decayed_rewards, aantal_batches)

    for batch_action, batch_observation, batch_decayed_reward in zip(batch_actions, batch_observations, batch_decayed_rewards):

        with tf.GradientTape() as tape:
            predictions = agent(batch_observation)
            loss = tf.keras.losses.mse(batch_action, predictions) * batch_decayed_reward
        train_vars = agent.trainable_variables
        grads = tape.gradient(loss, train_vars)
        optimizer.apply_gradients(zip(grads, train_vars))

    if min(scores[-5:]) > 195:
        break

plt.plot(scores)
plt.show()





