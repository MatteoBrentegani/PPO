import sys
import matplotlib.pyplot as plt
import numpy as np
import yaml
import random

from collections import deque
from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K
from keras.optimizers import Adam
# from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import numba as nb

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EPISODES = 5000


EPOCHS = 10
NOISE = 1.0 # Exploration noise
ENTROPY_LOSS = 1e-3
EPSILON = 0.2


@nb.jit
def exponential_average(old, new, b1):
    return old * b1 + (1-b1) * new


def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        r = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1-EPSILON, max_value=1+EPSILON) * advantage) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
    return loss

# losses ={
#     "output_layer1":proximal_policy_optimization_loss(advantage=advantage, old_prediction= old_prediction_p),
#     "output_layer2":proximal_policy_optimization_loss(advantage=advantage, old_prediction= old_prediction_q)
# }
lossWeights={
    "output1":0.625,
    "output2":0.375
}

target1 =  tf.placeholder(tf.float32, shape=(64,5)) 
target2 = tf.placeholder(tf.float32, shape=(64,3))


class PPO:
    def __init__(self, env, batch_size, target_update, episodes, state_size, action_size, action_size2, buffer_size = 256, gamma = 0.99, lr = 1e-4, tau = 0.001):

        self.env = env
        print(self.env.action_space, 'action_space', self.env.observation_space, 'observation_space')
        self.episode = 0
        self.observation = self.env.reset()
        self.reward = []
        self.reward_over_time = []
        self.gradient_steps = 0
        self.action_size = action_size
        self.action_size2 = action_size2
        self.state_size = state_size
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr


        self.reward_list = []
        self.success_list = []
        self.success = 0
        self.success_queue = deque(maxlen = 100)

        self.dummy_action1, self.dummy_action2, self.dummy_value1, self.dummy_value2 = np.zeros((1, self.action_size )), np.zeros((1, self.action_size2 )), np.zeros((1, 1)), np.zeros((1, 1))
        self.critic = self.build_critic()
        self.model = self.build_model()

        self.run()


    def build_model(self): 
        state_input = Input(shape=(self.state_size,))
        advantage1 = Input(shape=(1,))
        advantage2 = Input(shape=(1,))
        old_prediction_p = Input(shape=(self.action_size,))
        old_prediction_q = Input(shape=(self.action_size2,))

        x = Dense(64, activation='relu')(state_input)
        x = Dense(64, activation='relu')(x)

        out_actions1 = Dense(self.action_size , activation='softmax', name='output1')(x)
        out_actions2 = Dense(self.action_size2, activation='softmax', name='output2')(x)

        model = Model(inputs=[state_input, advantage1, advantage2, old_prediction_p, old_prediction_q], outputs=[out_actions1, out_actions2])



        model.compile(optimizer=Adam(lr=self.lr),
                      loss=[proximal_policy_optimization_loss(
                          advantage=advantage1,
                          old_prediction=old_prediction_p),
                          proximal_policy_optimization_loss(
                              advantage=advantage2,
                              old_prediction=old_prediction_q)], loss_weights=lossWeights, target_tensors=[target1,target2])
        model.summary()

        return model

    def build_critic(self):

        state_input = Input(shape=(self.state_size,))
        x = Dense(64, activation='relu')(state_input)
        x = Dense(64, activation='relu')(x)

        out_value1 = Dense(1)(x)
        out_value2 = Dense(1)(x)

        model = Model(inputs=[state_input], outputs=[out_value1, out_value2])
        model.compile(optimizer=Adam(lr=self.lr), loss='mse')

        return model

    def reset_env(self):
        self.episode += 1
        self.observation = self.env.reset()
        self.reward = []


    def sumprob(self, prob, lenght):
        sum = 0
        for value in prob[0]:
            sum+=value
        return sum / lenght

    def get_action(self):
        p, q = self.model.predict([self.observation.reshape(1, self.state_size), self.dummy_value1, self.dummy_value2, self.dummy_action1, self.dummy_action2])

        action = np.random.choice(self.action_size , p=np.nan_to_num(p[0]))
        action2 = np.random.choice(self.action_size2, p=np.nan_to_num(q[0]))


        action_matrix1 = np.zeros(self.action_size)
        action_matrix2 = np.zeros(self.action_size2)
        action_matrix1[action] = 1
        action_matrix2[action2] = 1 #check
        return action, action2, action_matrix1, action_matrix2, p, q

    def transform_reward(self):
        self.reward_over_time.append(np.array(self.reward).sum())
        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * self.gamma

    def get_batch(self):
        batch = [[], [], [], [], [], []]

        tmp_batch = [[], [], [], [], []]
        while len(batch[0]) < self.buffer_size:
            action1, action2, action_matrix1, action_matrix2, predicted_action_p, predicted_action_q = self.get_action()

            observation, reward, done, _ = self.env.step([action1, action2])
            self.reward.append(reward)


            tmp_batch[0].append(self.observation)
            tmp_batch[1].append(action_matrix1)
            tmp_batch[2].append(action_matrix2)
            tmp_batch[3].append(predicted_action_p)
            tmp_batch[4].append(predicted_action_q)
            self.observation = observation

            if done:
                self.success_queue.append(reward)
                self.success = int(self.success_queue.count(1)/(len(self.success_queue)+0.0)*100)

                self.success_list.append(self.success)
                self.reward_list.append(reward)

                if self.episode % 500:
                    np.savetxt("results/reward_list.txt" , self.reward_list, fmt='%3i')
                    np.savetxt("results/success_list.txt", self.success_list, fmt='%3i')
                    self.model.save("models/trainedPPO.h5")


                self.transform_reward()
                for i in range(len(tmp_batch[0])):
                    obs, action1, action2, pred_p, pred_q = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i], tmp_batch[3][i], tmp_batch[4][i]
                    r = self.reward[i]
                    batch[0].append(obs)
                    batch[1].append(action1)
                    batch[2].append(action2)
                    batch[3].append(pred_p)
                    batch[4].append(pred_q)
                    batch[5].append(r)
                tmp_batch = [[], [], [], [], []]
                self.reset_env()
        obs, action1, action2, pred_p,pred_q, reward = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.array(batch[3]), np.array(batch[4]), np.reshape(np.array(batch[5]), (len(batch[5]), 1))
        pred_p = np.reshape(pred_p, (pred_p.shape[0], pred_p.shape[2]))
        pred_q = np.reshape(pred_q, (pred_q.shape[0], pred_q.shape[2]))
        return obs, action1, action2, pred_p, pred_q, reward

    def run(self):
        print("> Start")
        while self.episode < EPISODES:
            print("EPIDODE: ")
            print(self.episode)
            obs, action1, action2, pred_p, pred_q, reward = self.get_batch()

            obs, action1, action2, pred_p, pred_q, reward = obs[:self.buffer_size], action1[:self.buffer_size], action2[:self.buffer_size], pred_p[:self.buffer_size], pred_q[:self.buffer_size], reward[:self.buffer_size]
            old_prediction_p = pred_p
            old_prediction_q = pred_q
            pred_values1, pred_values2 = self.critic.predict(obs)

            advantage1 = reward - pred_values1
            advantage2 = reward - pred_values2
            model_loss = self.model.fit([obs, advantage1, advantage2, old_prediction_p, old_prediction_q], [action1, action2], batch_size=self.batch_size, shuffle=True, epochs=EPOCHS, verbose=False)
            critic_loss = self.critic.fit([obs], [reward, reward], batch_size=self.batch_size, shuffle=True, epochs=EPOCHS, verbose=False)

            # print("model_loss: ",model_loss.history['loss'])
            # print("critic_loss: ",critic_loss.history['loss'])
            self.gradient_steps += 1

        self.model.save("models/trainedPPO.h5")
        print("> Stop")

        old, ma_list, ma_success = 0, [], []
        for value in self.reward_over_time:
            old = exponential_average(old, value, .99)
            ma_list.append(old)
        for value in self.success_list:
            ma_success.append(value)

        x1=np.linspace(0.0, EPISODES)
        x2=np.linspace(0.0, EPISODES)

        y1=np.linspace(0.0, 100)
        y2=np.linspace(0.0, 100)

        plt.subplot(2, 1, 1)
        plt.plot(ma_list)
        plt.grid(True)
        plt.title("PPO - Linear and Angular Velocity")
        plt.ylabel("reward")

        plt.subplot(2, 1, 2)
        plt.plot(ma_success, 'r')
        plt.grid(True)
        plt.xlabel('episodes')
        plt.ylabel('success')

        plt.show()
