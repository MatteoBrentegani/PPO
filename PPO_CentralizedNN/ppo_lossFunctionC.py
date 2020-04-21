import sys
import matplotlib.pyplot as plt
import numpy as np
import yaml
import random

from array import *
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
NOISE = 1.0
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
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value= 0.8 , max_value= 1.2) * advantage) + ENTROPY_LOSS * - (prob * K.log(prob + 1e-10)))
    return loss


lossWeights={
    "output11":0.3125,
    "output12":0.1875,
    "output21":0.3125,
    "output22":0.1875
}

target1 = tf.placeholder(tf.float32, shape=(64,5))
target2 = tf.placeholder(tf.float32, shape=(64,3))
target3 = tf.placeholder(tf.float32, shape=(64,5))
target4 = tf.placeholder(tf.float32, shape=(64,3))


class PPO:
    def __init__(self, env, eps, eps_min, eps_decay, batch_size, target_update, episodes, state_size, action_size, action_size2, buffer_size = 256, gamma = 0.99, lr = 1e-4, tau = 0.001):

        self.env = env
        print(self.env.action_space, 'action_space', self.env.observation_space, 'observation_space')
        self.episode = 0
        self.observation = self.env.reset()
        self.reward = []
        self.reward2 = []
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
        self.critic1 = self.build_critic()
        self.critic2 = self.build_critic()
        self.model = self.build_model()

        self.run()


    def build_model(self):
        state_input = Input(shape=(self.state_size,))
        state_input2 = Input(shape=(self.state_size,))

        advantage1 = Input(shape=(1,))
        advantage2 = Input(shape=(1,))
        advantage3 = Input(shape=(1,))
        advantage4 = Input(shape=(1,))
        old_prediction_p1 = Input(shape=(self.action_size,))
        old_prediction_q1 = Input(shape=(self.action_size2,))
        old_prediction_p2 = Input(shape=(self.action_size,))
        old_prediction_q2 = Input(shape=(self.action_size2,))


        x = Dense(64, activation='relu')(state_input)
        x = Dense(64, activation='relu')(x)
        y = Dense(64, activation='relu')(state_input2)
        y = Dense(64, activation='relu')(y)


        out_actions1 = Dense(self.action_size , activation='softmax', name='output11')(x)
        out_actions2 = Dense(self.action_size2, activation='softmax', name='output12')(x)
        out_actions3 = Dense(self.action_size , activation='softmax', name='output21')(y)
        out_actions4 = Dense(self.action_size2, activation='softmax', name='output22')(y)

        model = Model(inputs=[state_input, state_input2, advantage1, advantage2, advantage3, advantage4, old_prediction_p1, old_prediction_q1, old_prediction_p2, old_prediction_q2], outputs=[out_actions1, out_actions2, out_actions3, out_actions4])



        model.compile(optimizer=Adam(lr=self.lr),
                      loss=[proximal_policy_optimization_loss(
                          advantage=advantage1,
                          old_prediction=old_prediction_p1),
                          proximal_policy_optimization_loss(
                              advantage=advantage2,
                              old_prediction=old_prediction_q1),
                          proximal_policy_optimization_loss(
                              advantage=advantage3,
                              old_prediction=old_prediction_p2),
                          proximal_policy_optimization_loss(
                              advantage=advantage4,
                              old_prediction=old_prediction_q2)], loss_weights=lossWeights, target_tensors=[target1, target2, target3, target4])
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
        self.reward2 = []


    def sumprob(self, prob, lenght):
        sum = 0
        for value in prob[0]:
            sum+=value
        return sum / lenght

    def get_action(self):

        arr1 = self.observation[0]
        arr2 = self.observation[1]

        tmp1 = arr1.tolist()
        tmp2 = arr2.tolist()


        obs1 = np.array(tmp1[:])
        obs2 = np.array(tmp2[:])

        p1, q1, p2, q2 = self.model.predict([
        obs1.reshape(1, self.state_size),
        obs2.reshape(1, self.state_size),
        self.dummy_value1, self.dummy_value2,
        self.dummy_value1, self.dummy_value2,
        self.dummy_action1, self.dummy_action2,
        self.dummy_action1, self.dummy_action2])


        action = (np.random.choice(self.action_size , p=np.nan_to_num(p1[0])))
        action2 =(np.random.choice(self.action_size2, p=np.nan_to_num(q1[0])))


        action3=(np.random.choice(self.action_size , p=np.nan_to_num(p2[0])))
        action4=(np.random.choice(self.action_size2, p=np.nan_to_num(q2[0])))

        action_matrix1 = np.zeros(self.action_size)
        action_matrix2 = np.zeros(self.action_size2)
        action_matrix3 = np.zeros(self.action_size)
        action_matrix4 = np.zeros(self.action_size2)

        action_matrix1[action] = 1
        action_matrix2[action2] = 1
        action_matrix3[action3] = 1
        action_matrix4[action4] = 1

        return action, action2, action3, action4, action_matrix1, action_matrix2, action_matrix3, action_matrix4, p1, q1, p2, q2

    def transform_reward(self, index):
        if index == 0:
            self.reward_over_time.append(np.array(self.reward).sum())
        if index == 1:
            self.reward_over_time.append(np.array(self.reward2).sum())
        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * self.gamma
        # self.reward_over_time2.append(np.array(self.reward2).sum())
        for j in range(len(self.reward2) - 2, -1, -1):
            self.reward2[j] += self.reward2[j + 1] * self.gamma

    def get_batch(self):
        batch = [[], [], [], [], [], [], [], [], [], [], [], []]

        tmp_batch = [[], [], [], [], [], [], [], [], [], []]
        while len(batch[0]) < self.buffer_size:
            action1, action2, action3, action4, action_matrix1, action_matrix2, action_matrix3, action_matrix4, predicted_action_p, predicted_action_q, predicted_action_p2, predicted_action_q2 = self.get_action()
            #print(action1)
            #print(action2)
            observation, reward, done, _ = self.env.step([[action1, action2], [action3, action4]])

            self.reward.append(reward[0])
            self.reward2.append(reward[1])
            tmp_batch[0].append(self.observation[0][:])
            tmp_batch[1].append(action_matrix1)
            tmp_batch[2].append(action_matrix2)
            tmp_batch[3].append(predicted_action_p)
            tmp_batch[4].append(predicted_action_q)
            tmp_batch[5].append(action_matrix3)
            tmp_batch[6].append(action_matrix4)
            tmp_batch[7].append(predicted_action_p2)
            tmp_batch[8].append(predicted_action_q2)
            tmp_batch[9].append(self.observation[1][:])
            self.observation = observation

            try:
                index = done.index(True)
                index = index
            except:
                index = -1

            if index >= 0:

                self.success_queue.append(reward[:][index])
                self.success = int(self.success_queue.count(1)/(len(self.success_queue)+0.0)*100)

                self.success_list.append(self.success)
                self.reward_list.append(reward[:][index])

                if self.episode % 500:
                    # print("> Saving...")
                    np.savetxt("results/reward_list.txt" , self.reward_list, fmt='%3i')
                    np.savetxt("results/success_list.txt", self.success_list, fmt='%3i')
                    self.model.save("models/trainedPPOC.h5")


                self.transform_reward(index)
                for i in range(len(tmp_batch[0])):
                    obs, obs2, action1, action2, pred_p, pred_q, action3, action4, pred_p2, pred_q2 = tmp_batch[0][i], tmp_batch[9][i], tmp_batch[1][i], tmp_batch[2][i], tmp_batch[3][i], tmp_batch[4][i], tmp_batch[5][i], tmp_batch[6][i], tmp_batch[7][i], tmp_batch[8][i]

                    r = self.reward[i]
                    r2= self.reward2[i]
                    if done[0]:
                        r2 = r
                    else:
                        r = r2


                    batch[0].append(obs)
                    batch[1].append(action1)
                    batch[2].append(action2)
                    batch[3].append(pred_p)
                    batch[4].append(pred_q)
                    batch[5].append(action3)
                    batch[6].append(action4)
                    batch[7].append(pred_p2)
                    batch[8].append(pred_q2)
                    batch[9].append(r)
                    batch[10].append(r2)
                    batch[11].append(obs2)
                tmp_batch = [[], [], [], [], [], [], [], [], [], []]
                self.reset_env()

        obs, action1, action2, pred_p, pred_q, action3, action4, pred_p2, pred_q2, reward1, reward2, obs2 = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.array(batch[3]), np.array(batch[4]), np.array(batch[5]), np.array(batch[6]), np.array(batch[7]), np.array(batch[8]), batch[9], batch[10], np.array(batch[11])
        pred_p = np.reshape(pred_p, (pred_p.shape[0], pred_p.shape[2]))
        pred_q = np.reshape(pred_q, (pred_q.shape[0], pred_q.shape[2]))#controllare
        pred_p2 = np.reshape(pred_p2, (pred_p2.shape[0], pred_p2.shape[2]))
        pred_q2 = np.reshape(pred_q2, (pred_q2.shape[0], pred_q2.shape[2]))

        return obs, obs2, action1, action2, pred_p, pred_q, action3, action4, pred_p2, pred_q2, reward1, reward2, index #reward array 2 dimensional

    def run(self):
        print("> Start")
        while self.episode < EPISODES:
            obs, obs2, action1, action2, pred_p, pred_q, action3, action4, pred_p2, pred_q2, reward1, reward2, index = self.get_batch()

            obs, obs2, action1, action2, pred_p, pred_q, action3, action4, pred_p2, pred_q2, reward1, reward2 = obs[:self.buffer_size], obs2[:self.buffer_size], action1[:self.buffer_size], action2[:self.buffer_size], pred_p[:self.buffer_size], pred_q[:self.buffer_size], action3[:self.buffer_size], action4[:self.buffer_size], pred_p2[:self.buffer_size], pred_q2[:self.buffer_size], reward1[:self.buffer_size], reward2[:self.buffer_size]
            old_prediction_p = pred_p
            old_prediction_q = pred_q
            old_prediction_p2 = pred_p2
            old_prediction_q2 = pred_q2
            #print(obs)
            #qui faccio un reshape però devo controllare se lo faccio correttamente per dare l'input corretto nel fit
            obsC1 = obs.reshape(-1, obs.shape[-1])
            obsC2 = obs2.reshape(-1, obs2.shape[-1])
            pred_values1, pred_values2 = self.critic1.predict(obsC1)
            pred_values3, pred_values4 = self.critic2.predict(obsC2)

            advantage1 = reward1 - pred_values1[:256]
            advantage2 = reward1 - pred_values2[:256]
            advantage3 = reward2 - pred_values3[:256]
            advantage4 = reward2 - pred_values4[:256]


            model_loss = self.model.fit([obsC1[:self.buffer_size], obsC2[:self.buffer_size], advantage1[:self.buffer_size, 1], advantage2[:self.buffer_size, 1], advantage3[:self.buffer_size, 1], advantage4[:self.buffer_size, 1], old_prediction_p, old_prediction_q, old_prediction_p2, old_prediction_q2], [action1, action2, action3, action4], batch_size=self.batch_size, shuffle=True, epochs=EPOCHS, verbose=False)

            critic_loss = self.critic1.fit([obsC1[:self.buffer_size]], [reward1, reward1], batch_size=self.batch_size, shuffle=True, epochs=EPOCHS, verbose=False)
            critic_loss2 = self.critic2.fit([obsC2[:self.buffer_size]], [reward2, reward2], batch_size=self.batch_size, shuffle=True, epochs=EPOCHS, verbose=False)

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
        #plt.plot(x1, y1, 'o-')
        plt.plot(ma_list)
        plt.grid(True)
        plt.title("PPO centralized")
        plt.ylabel("reward")

        plt.subplot(2, 1, 2)
        #plt.plot(x2, y2, '.-')
        plt.plot(ma_success, 'r')
        plt.grid(True)
        plt.xlabel('episodes')
        plt.ylabel('success')


        plt.show()
