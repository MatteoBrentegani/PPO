import sys
import matplotlib.pyplot as plt
import numpy as np
import yaml
import random

from array import *
from collections import deque
from keras.models import Model, load_model
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
AGENT = 7


@nb.jit
def exponential_average(old, new, b1):
    return old * b1 + (1-b1) * new


def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        r = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=0.8, max_value=1.2) * advantage) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
    return loss


lossWeights={
    "output1":0.5,
    "output2":0.5
}

target1 =  tf.placeholder(tf.float32, shape=(64,5)) # shapes of output1 your target1 has
target2 = tf.placeholder(tf.float32, shape=(64,3)) # shapes of output2 your target1 has



class PPO:
    def __init__(self, env, eps, eps_min, eps_decay, batch_size, target_update, episodes, state_size, action_size, action_size2, buffer_size = 256, gamma = 0.99, lr = 1e-4, tau = 0.001):

        self.env = env
        print(self.env.action_space, 'action_space', self.env.observation_space, 'observation_space')
        self.episode = 0
        self.observation = self.env.reset()
        self.reward = []
        self.reward2 = []
        self.reward3 = []
        self.reward4 = []
        self.reward5 = []
        self.reward6 = []
        self.reward7 = []
        self.reward_over_time = []
        self.reward_over_time2 = []
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
        out_value3 = Dense(1)(x)
        out_value4 = Dense(1)(x)

        model = Model(inputs=[state_input], outputs=[out_value1, out_value2])
        model.compile(optimizer=Adam(lr=self.lr), loss='mse')

        return model

    def reset_env(self):
        self.episode += 1
        self.observation = self.env.reset()
        self.reward = []
        self.reward2 = []
        self.reward3 = []
        self.reward4 = []
        self.reward5 = []
        self.reward6 = []
        self.reward7 = []


    def sumprob(self, prob, lenght):
        sum = 0
        for value in prob[0]:
            sum+=value
        return sum / lenght

    def get_action(self, index):

        arr1 = self.observation[index]

        tmp1 = arr1.tolist()

        obs = np.array(tmp1[:])

        p, q = self.model.predict([obs.reshape(1, self.state_size), self.dummy_value1, self.dummy_value2, self.dummy_action1, self.dummy_action2])

        action = np.random.choice(self.action_size , p=np.nan_to_num(p[0]))
        action2 = np.random.choice(self.action_size2, p=np.nan_to_num(q[0]))


        action_matrix1 = np.zeros(self.action_size)
        action_matrix2 = np.zeros(self.action_size2)
        action_matrix1[action] = 1
        action_matrix2[action2] = 1 #check

        return [action, action2, action_matrix1, action_matrix2, p, q]

    def transform_reward(self, index):
        if index == 1:
            self.reward_over_time.append(np.array(self.reward).sum())
        if index == 2:
            self.reward_over_time.append(np.array(self.reward2).sum())
        if index == 3:
            self.reward_over_time.append(np.array(self.reward3).sum())
        if index == 4:
            self.reward_over_time.append(np.array(self.reward4).sum())
        if index == 5:
            self.reward_over_time.append(np.array(self.reward5).sum())
        if index == 6:
            self.reward_over_time.append(np.array(self.reward6).sum())
        if index == 7:
            self.reward_over_time.append(np.array(self.reward7).sum())

        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * self.gamma

        for j in range(len(self.reward2) - 2, -1, -1):
            self.reward2[j] += self.reward2[j + 1] * self.gamma

        for j in range(len(self.reward3) - 2, -1, -1):
            self.reward3[j] += self.reward3[j + 1] * self.gamma

        for j in range(len(self.reward4) - 2, -1, -1):
            self.reward4[j] += self.reward4[j + 1] * self.gamma

        for j in range(len(self.reward5) - 2, -1, -1):
            self.reward5[j] += self.reward5[j + 1] * self.gamma

        for j in range(len(self.reward6) - 2, -1, -1):
            self.reward6[j] += self.reward6[j + 1] * self.gamma

        for j in range(len(self.reward7) - 2, -1, -1):
            self.reward7[j] += self.reward7[j + 1] * self.gamma

    def return_reward(self, i):
        if i == 1:
            return self.reward[i]
        if i == 2:
            return self.reward2[i]
        if i == 3:
            return self.reward3[i]
        if i == 4:
            return self.reward4[i]
        if i == 5:
            return self.reward5[i]
        if i == 6:
            return self.reward6[i]
        if i == 7:
            return self.reward7[i]

    def get_batch(self):
        batch = [[], [], [], [], [], []]

        tmp_batch = [[], [], [], [], [], [], [], []]

        while len(batch[0]) < self.buffer_size:
            agent1 = self.get_action(0)
            agent2 = self.get_action(1)
            agent3 = self.get_action(2)
            agent4 = self.get_action(3)
            agent5 = self.get_action(4)
            agent6 = self.get_action(5)
            agent7 = self.get_action(6)
            

            observation, reward, done, _ = self.env.step([[agent1[0],agent1[1]],[agent2[0],agent2[1]],[agent3[0],agent3[1]],[agent4[0],agent4[1]],[agent5[0],agent5[1]],[agent6[0],agent6[1]],[agent7[0],agent7[1]]])

            self.reward.append(reward[0])
            self.reward2.append(reward[1])
            self.reward3.append(reward[2])
            self.reward4.append(reward[3])
            self.reward5.append(reward[4])
            self.reward6.append(reward[5])
            self.reward7.append(reward[6])
            
            #[action, action2, action_matrix1, action_matrix2, p, q]
            # 0         1       2               3              4  5

            
            tmp_batch[0].append(self.observation)
            tmp_batch[1].append(agent1)
            tmp_batch[2].append(agent2)
            tmp_batch[3].append(agent3)
            tmp_batch[4].append(agent4)
            tmp_batch[5].append(agent5)
            tmp_batch[6].append(agent6)
            tmp_batch[7].append(agent7)
           

            self.observation = observation


            try:
                index = done.index(True)
                index = index +1
            except:
                index = -1
                

           
            if index >= 0:

                self.success_queue.append(reward[:][index-1])
                self.success = int(self.success_queue.count(1)/(len(self.success_queue)+0.0)*100)

                self.success_list.append(self.success)
                self.reward_list.append(reward[:][index-1])

                if self.episode % 500:
                    # print("> Saving...")
                    np.savetxt("results/reward_list.txt" , self.reward_list, fmt='%3i')
                    np.savetxt("results/success_list.txt", self.success_list, fmt='%3i')
                    self.model.save("models/trainedPPO.h5")

               
                self.transform_reward(index)

                for i in range(len(tmp_batch[index])): #j == index into the normal execution
                    obs, action1, action2, pred_p, pred_q = tmp_batch[0], tmp_batch[index][i][2], tmp_batch[index][i][3], tmp_batch[index][i][4], tmp_batch[index][i][5]
                    #print(pred_p)
                    if index == 1:
                        r = self.reward[i]
                    if index == 2:
                        r = self.reward2[i]
                    if index == 3:
                        r = self.reward3[i]
                    if index == 4:
                        r = self.reward4[i]
                    if index == 5:
                        r = self.reward5[i]
                    if index == 6:
                        r = self.reward6[i]
                    if index == 7:
                        r = self.reward7[i]

                    batch[0].append(obs[i][index-1])
                    batch[1].append(action1)
                    batch[2].append(action2)
                    batch[3].append(pred_p)
                    batch[4].append(pred_q)
                    batch[5].append(r)

                tmp_batch = [[], [], [], [], [], [], [], []]
                self.reset_env()

        obs, action1, action2, pred_p, pred_q, reward = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.array(batch[3]), np.array(batch[4]), np.reshape(np.array(batch[5]), (len(batch[5]), 1))
        pred_p = np.reshape(pred_p, (pred_p.shape[0], pred_p.shape[2]))


        #print(pred_p)
        pred_q = np.reshape(pred_q, (pred_q.shape[0], pred_q.shape[2]))

        return obs, action1, action2, pred_p, pred_q, reward

    def run(self):
        print("> Start")
        #self.model = load_model("models/trainedPPO.h5")

        while self.episode < EPISODES:
            obs, action1, action2, pred_p, pred_q, reward = self.get_batch()

           

            obs, action1, action2, pred_p, pred_q, reward= obs[:self.buffer_size], action1[:self.buffer_size], action2[:self.buffer_size], pred_p[:self.buffer_size], pred_q[:self.buffer_size], reward[:self.buffer_size]
           
            pred_values1, pred_values2= self.critic1.predict(obs)

            advantage1 = reward - pred_values1
            advantage2 = reward - pred_values2
            old_prediction_p = pred_p
            old_prediction_q = pred_q

            model_loss = self.model.fit([obs, advantage1, advantage2, old_prediction_p, old_prediction_q], [action1, action2], batch_size=self.batch_size, shuffle=True, epochs=EPOCHS, verbose=False)


            critic_loss = self.critic1.fit([obs], [reward, reward], batch_size=self.batch_size, shuffle=True, epochs=EPOCHS, verbose=False)

         
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
        plt.title("PPO MultiAgent - no obstacle")
        plt.ylabel("reward")

        plt.subplot(2, 1, 2)
        #plt.plot(x2, y2, '.-')
        plt.plot(ma_success, 'r')
        plt.grid(True)
        plt.xlabel('episodes')
        plt.ylabel('success')

        plt.show()
