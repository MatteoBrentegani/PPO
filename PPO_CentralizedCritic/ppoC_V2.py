import sys
import matplotlib.pyplot as plt
import numpy as np
import yaml
import random

from array import *
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense, Add
from keras import backend as K
from keras.optimizers import Adam
# from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import pandas as pd
import numba as nb

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EPISODES = 5000

# LOSS_CLIPPING = 0.2 # Only implemented clipping for the surrogate loss, paper said it was best
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

target1 =  tf.placeholder(tf.float32, shape=(64,5))
target2 = tf.placeholder(tf.float32, shape=(64,3))


# old buffer size 20000
class PPO:
    #256 buffer_size
    def __init__(self, env, eps, batch_size, target_update, episodes, state_size, action_size, action_size2, buffer_size = 256, gamma = 0.99, lr = 1e-4, tau = 0.001):

        self.env = env
        print(self.env.action_space, 'action_space', self.env.observation_space, 'observation_space')
        self.episode = 0
        self.observation = self.env.reset()
        self.reward = []
        self.reward2 = []
        self.reward3 = []
        self.reward4 = []
        self.reward_over_time1 = []
        self.reward_over_time2 = []
        self.reward_over_time3 = []
        self.reward_over_time4 = []
        self.gradient_steps = 0
        self.action_size = action_size
        self.action_size2 = action_size2
        self.state_size = state_size
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr


        self.reward_list = []
        self.success_list1 = []
        self.success_list2 = []
        self.success_list3 = []
        self.success_list4 = []
        self.success1 = 0
        self.success_queue1 = deque(maxlen = 100)
        self.success2 = 0
        self.success_queue2 = deque(maxlen = 100)
        self.success3 = 0
        self.success_queue3 = deque(maxlen = 100)
        self.success4 = 0
        self.success_queue4 = deque(maxlen = 100)

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

        state_input1 = Input(shape=(self.state_size,))
        state_input2 = Input(shape=(self.state_size,))
        state_input3 = Input(shape=(self.state_size,))
        state_input4 = Input(shape=(self.state_size,))

        x = Add()([state_input1, state_input2, state_input3, state_input4])
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)

        #agent1
        out_value11 = Dense(1)(x)
        out_value12 = Dense(1)(x)
        # #agent2
        # out_value21 = Dense(1)(x)
        # out_value22 = Dense(1)(x)
        # #agent3
        # out_value31 = Dense(1)(x)
        # out_value32 = Dense(1)(x)
        # #agent4
        # out_value41 = Dense(1)(x)
        # out_value42 = Dense(1)(x)

        model = Model(inputs=[state_input1, state_input2, state_input3, state_input4], outputs=[out_value11, out_value12])#, out_value21, out_value22, out_value31, out_value32, out_value41, out_value42])
        model.compile(optimizer=Adam(lr=self.lr), loss='mse')

        return model

    def reset_env(self):
        self.episode += 1
        self.observation = self.env.reset()
        self.reward = []
        self.reward2 = []
        self.reward3 = []
        self.reward4 = []


    def sumprob(self, prob, lenght):
        sum = 0
        for value in prob[0]:
            sum+=value
        return sum / lenght

    def get_action(self, index):

        arr1 = self.observation[index]

        tmp1 = arr1.tolist()

        obs = np.array(tmp1[:])

        #print(self.observation)
        p, q = self.model.predict([obs.reshape(1, self.state_size), self.dummy_value1, self.dummy_value2, self.dummy_action1, self.dummy_action2])

        action = np.random.choice(self.action_size , p=np.nan_to_num(p[0]))
        action2 = np.random.choice(self.action_size2, p=np.nan_to_num(q[0]))


        action_matrix1 = np.zeros(self.action_size)
        action_matrix2 = np.zeros(self.action_size2)
        action_matrix1[action] = 1
        action_matrix2[action2] = 1 #check

        return [action, action2, action_matrix1, action_matrix2, p, q]

    def transform_reward(self, index):
        #print('Episode reward', np.array(self.reward).sum(), self.episode)
        if index == 1:
            self.reward_over_time1.append(np.array(self.reward).sum())
            for j in range(len(self.reward) - 2, -1, -1):
                self.reward[j] += self.reward[j + 1] * self.gamma
        if index == 2:
            self.reward_over_time2.append(np.array(self.reward2).sum())
            for j in range(len(self.reward2) - 2, -1, -1):
                self.reward2[j] += self.reward2[j + 1] * self.gamma
        if index == 3:
            self.reward_over_time3.append(np.array(self.reward3).sum())
            for j in range(len(self.reward3) - 2, -1, -1):
                self.reward3[j] += self.reward3[j + 1] * self.gamma
        if index == 4:
            self.reward_over_time4.append(np.array(self.reward4).sum())
            for j in range(len(self.reward4) - 2, -1, -1):
                self.reward4[j] += self.reward4[j + 1] * self.gamma


    def return_reward(self, i):
        if i == 1:
            return self.reward[i]
        if i == 2:
            return self.reward2[i]
        if i == 3:
            return self.reward3[i]
        if i == 4:
            return self.reward4[i]

    def get_batch(self):
        batch = [[], [], [], [], [], []]
        obs_batch = [[], [], [], []]

        tmp_batch = [[], [], [], [], [], [], [], []]
        oldReward1, oldReward2, oldReward3, oldReward4 = -1, -1, -1, -1

        while len(batch[0]) < self.buffer_size:
            agent1 = self.get_action(0)
            agent2 = self.get_action(1)
            agent3 = self.get_action(2)
            agent4 = self.get_action(3)
            observation, reward, done, _ = self.env.step([[agent1[0],agent1[1]],[agent2[0],agent2[1]],[agent3[0],agent3[1]],[agent4[0],agent4[1]]])



            self.reward.append(reward[0])
            self.reward2.append(reward[1])
            self.reward3.append(reward[2])
            self.reward4.append(reward[3])
            #[action, action2, action_matrix1, action_matrix2, p, q]
            # 0         1       2               3              4  5

            #print(np.shape(self.observation)) #(7 ,27)
            tmp_batch[0].append(self.observation)
            tmp_batch[1].append(agent1)
            tmp_batch[2].append(agent2)
            tmp_batch[3].append(agent3)
            tmp_batch[4].append(agent4)

            

            self.observation = observation

            if (done[0] == True and done[1] == True and done[2] == True and done[3] == True):
                # if done[0] == True and (reward[0] == 1 or reward[0] == -1):
                    #print("done 1", reward[0])
                index = 1;
                self.batchElaboration(tmp_batch, reward[0], index, batch, obs_batch)
                # ack = True;
                # if done[1] == True and (reward[1] == 1 or reward[1] == -1):
                    #print("done 2", reward[1])
                index = 2;
                self.batchElaboration(tmp_batch, reward[1], index, batch, obs_batch)
                # ack = True;
                # if done[2] == True and (reward[2] == 1 or reward[2] == -1):
                    #print("done 3", reward[2])
                index = 3;
                self.batchElaboration(tmp_batch, reward[2], index, batch, obs_batch)
                # ack = True;
                # if done[3] == True and (reward[3] == 1 or reward[3] == -1):
                    #print("done 4", reward[3])
                index = 4;
                self.batchElaboration(tmp_batch, reward[3], index, batch, obs_batch)

                tmp_batch = [[], [], [], [], [], [], [], []]
                self.reset_env()


        obs, action1, action2, pred_p, pred_q, reward = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.array(batch[3]), np.array(batch[4]), np.reshape(np.array(batch[5]), (len(batch[5]), 1))

        pred_p = np.reshape(pred_p, (pred_p.shape[0], pred_p.shape[2]))
        pred_q = np.reshape(pred_q, (pred_q.shape[0], pred_q.shape[2]))
        return obs, action1, action2, pred_p, pred_q, reward, np.array(obs_batch[0]), np.array(obs_batch[1]), np.array(obs_batch[2]), np.array(obs_batch[3])

    def batchElaboration(self, tmp_batch, reward, index, batch, obs_batch):


        if index == 1:
            self.success_queue1.append(reward)
            self.success1 = int(self.success_queue1.count(1)/(len(self.success_queue1)+0.0)*100)
            self.success_list1.append(self.success1)
        if index == 2:
            self.success_queue2.append(reward)
            self.success2 = int(self.success_queue2.count(1)/(len(self.success_queue2)+0.0)*100)
            self.success_list2.append(self.success2)
        if index == 3:
            self.success_queue3.append(reward)
            self.success3 = int(self.success_queue3.count(1)/(len(self.success_queue3)+0.0)*100)
            self.success_list3.append(self.success3)
        if index == 4:
            self.success_queue4.append(reward)
            self.success4 = int(self.success_queue4.count(1)/(len(self.success_queue4)+0.0)*100)
            self.success_list4.append(self.success4)

        self.reward_list.append(reward)

        if self.episode % 500 == 0:
            # print("> Saving...")
            np.savetxt("results/reward_list.txt" , self.reward_list, fmt='%3i')
            np.savetxt("results/success_list1.txt", self.success_list1, fmt='%3i')
            np.savetxt("results/success_list2.txt", self.success_list2, fmt='%3i')
            np.savetxt("results/success_list3.txt", self.success_list3, fmt='%3i')
            np.savetxt("results/success_list4.txt", self.success_list4, fmt='%3i')
            self.model.save("models/trainedPPO.h5")
            print("> EPISODE:", self.episode )

        self.transform_reward(index)
        for i in range(len(tmp_batch[index])):
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

            batch[0].append(obs[i][index-1])
            batch[1].append(action1)
            batch[2].append(action2)
            batch[3].append(pred_p)
            batch[4].append(pred_q)
            batch[5].append(r)
            obs_batch[index-1].append(obs[i][index-1])


        #return obs, action1, action2, pred_p, pred_q, reward

    def run(self):
        print("> Start ppo_lossFunction")
        #self.model = load_model("models/trainedPPO.h5")

        while self.episode < EPISODES:
            obs, action1, action2, pred_p, pred_q, reward, obs_batch1, obs_batch2, obs_batch3, obs_batch4 = self.get_batch()


            obs, action1, action2, pred_p, pred_q, reward= obs[:self.buffer_size], action1[:self.buffer_size], action2[:self.buffer_size], pred_p[:self.buffer_size], pred_q[:self.buffer_size], reward[:self.buffer_size]

            obs_batch1, obs_batch2, obs_batch3, obs_batch4 = obs_batch1[:self.buffer_size], obs_batch2[:self.buffer_size], obs_batch3[:self.buffer_size], obs_batch4[:self.buffer_size]

            pred_values1, pred_values2 = self.critic1.predict([obs_batch1, obs_batch2, obs_batch3, obs_batch4])

            advantage1 = reward - pred_values1
            advantage2 = reward - pred_values2
            old_prediction_p = pred_p
            old_prediction_q = pred_q

            model_loss = self.model.fit([obs, advantage1, advantage2, old_prediction_p, old_prediction_q], [action1, action2], batch_size=self.batch_size, shuffle=True, epochs=EPOCHS, verbose=False)


            critic_loss = self.critic1.fit([obs_batch1, obs_batch2, obs_batch3, obs_batch4], [reward, reward], batch_size=self.batch_size, shuffle=True, epochs=EPOCHS, verbose=False)

            #CONTROLALRE IL REWARD DEL CRITI NON CORRETTO SECONDO ME, PERCHE ANCHE AL'AGENTE 2 DO REWARD DELL'AGENTE 1
            self.gradient_steps += 1

        self.model.save("models/trainedPPO.h5")
        print("> Stop")


        old1, ma_list1, ma_success1 = 0, [], []
        old2, ma_list2, ma_success2 = 0, [], []
        old3, ma_list3, ma_success3 = 0, [], []
        old4, ma_list4, ma_success4 = 0, [], []

        for value in self.reward_over_time1:
            old1 = exponential_average(old1, value, .99)
            ma_list1.append(old1)
        for value in self.success_list1:
            ma_success1.append(value)

        for value in self.reward_over_time2:
            old2 = exponential_average(old2, value, .99)
            ma_list2.append(old2)
        for value in self.success_list2:
            ma_success2.append(value)

        for value in self.reward_over_time3:
            old3 = exponential_average(old3, value, .99)
            ma_list3.append(old3)
        for value in self.success_list3:
            ma_success3.append(value)

        for value in self.reward_over_time4:
            old4 = exponential_average(old4, value, .99)
            ma_list4.append(old4)
        for value in self.success_list4:
            ma_success4.append(value)

        df1 = pd.DataFrame({'x': range(EPISODES), 'a1_S': ma_success1, 'a1_R': ma_list1})
        df2 = pd.DataFrame({'x': range(EPISODES), 'a2_S': ma_success2, 'a2_R': ma_list2})
        df3 = pd.DataFrame({'x': range(EPISODES), 'a3_S': ma_success3, 'a3_R': ma_list3})
        df4 = pd.DataFrame({'x': range(EPISODES), 'a4_S': ma_success4, 'a4_R': ma_list4})
        plt.figure(100)
        # Initialize the figure
        plt.style.use('seaborn-darkgrid')
        palette = plt.get_cmap('Set1')
        num=0
        for column in df1.drop('x', axis=1):
            num+=1

            # Find the right spot on the plot
            plt.subplot(2,1, num)
            # plot every groups, but discreet
            plt.plot(df1['x'], df1[column], marker='', color=palette(num), linewidth=1.9, alpha=0.9, label=column)
            plt.xlim(0,EPISODES)
            plt.ylim(-100,100)
            if (num -1) % 2 == 0:
                plt.ylabel("Success")
            else:
                plt.ylabel("Reward")
            plt.xlabel("Episodes")
            # Not ticks everywhere
            if num in range(7) :
                plt.tick_params(labelbottom='off')
            if num not in [1,4,7] :
                plt.tick_params(labelleft='off')

            # Add title
            plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num) )


        plt.suptitle("Agent 1 SuccessRate & Reward", fontsize=11, fontweight=0, color='black', style='italic')

        # # Axis title
        # plt.text(0.5, 0.02, 'Eisodes', ha='center', va='center')
        # plt.text(0.06, 0.5, 'Success & Reward', ha='center', va='center', rotation='vertical')


        plt.figure(200)

        num=0
        for column in df2.drop('x', axis=1):
            num+=1

            # Find the right spot on the plot
            plt.subplot(2,1, num)
            # plot every groups, but discreet
            plt.plot(df2['x'], df2[column], marker='', color=palette(num), linewidth=1.9, alpha=0.9, label=column)
            plt.xlim(0,EPISODES)
            plt.ylim(-100,100)
            if (num -1) % 2 == 0:
                plt.ylabel("Success")
            else:
                plt.ylabel("Reward")
            plt.xlabel("Episodes")
            # Not ticks everywhere
            if num in range(7) :
                plt.tick_params(labelbottom='off')
            if num not in [1,4,7] :
                plt.tick_params(labelleft='off')

            # Add title
            plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num) )


        plt.suptitle("Agent 2 SuccessRate & Reward", fontsize=11, fontweight=0, color='black', style='italic')

        # Axis title
        # plt.text(0.5, 0.02, 'Episodes', ha='center', va='center')
        # plt.text(0.06, 0.5, 'Success & Reward', ha='center', va='center', rotation='vertical')

        plt.figure(300)

        num=0
        for column in df3.drop('x', axis=1):
            num+=1

            # Find the right spot on the plot
            plt.subplot(2,1, num)
            # plot every groups, but discreet
            plt.plot(df3['x'], df3[column], marker='', color=palette(num), linewidth=1.9, alpha=0.9, label=column)
            plt.xlim(0,EPISODES)
            plt.ylim(-100,100)
            if (num -1) % 2 == 0:
                plt.ylabel("Success")
            else:
                plt.ylabel("Reward")
            plt.xlabel("Episodes")
            # Not ticks everywhere
            if num in range(7) :
                plt.tick_params(labelbottom='off')
            if num not in [1,4,7] :
                plt.tick_params(labelleft='off')

            # Add title
            plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num) )


        plt.suptitle("Agent 3 SuccessRate & Reward", fontsize=11, fontweight=0, color='black', style='italic')

        # Axis title
        # plt.text(0.5, 0.02, 'Episodes', ha='center', va='center')
        # plt.text(0.06, 0.5, 'Success & Reward', ha='center', va='center', rotation='vertical')

        plt.figure(400)

        num=0
        for column in df4.drop('x', axis=1):
            num+=1

            # Find the right spot on the plot
            plt.subplot(2,1, num)
            # plot every groups, but discreet
            plt.plot(df4['x'], df4[column], marker='', color=palette(num), linewidth=1.9, alpha=0.9, label=column)
            plt.xlim(0,EPISODES)
            plt.ylim(-100,100)
            if (num -1) % 2 == 0:
                plt.ylabel("Success")
            else:
                plt.ylabel("Reward")
            plt.xlabel("Episodes")
            # Not ticks everywhere
            if num in range(7) :
                plt.tick_params(labelbottom='off')
            if num not in [1,4,7] :
                plt.tick_params(labelleft='off')

            # Add title
            plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num) )


        plt.suptitle("Agent 4 SuccessRate & Reward", fontsize=11, fontweight=0, color='black', style='italic')

        # Axis title
        # plt.text(0.5, 0.02, 'Episodes', ha='center', va='center')
        # plt.text(0.06, 0.5, 'Success & Reward', ha='center', va='center', rotation='vertical')

        plt.show()
