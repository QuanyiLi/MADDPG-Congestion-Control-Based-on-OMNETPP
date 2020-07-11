"""
The double DQN based on this paper: https://arxiv.org/abs/1509.06461

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""
# code is from Moufan Zhou. Searching for more algorithms on his website https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import maddpg.common.tf_util as U
import tensorflow.contrib.layers as layers
import maddpg.common.environment as environment
import json
import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


class DoubleDQN:
    def __init__(
            self,
            n_actions,
            n_features,
            idx,
            learning_rate=0.005,
            reward_decay=0.95,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=3000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            double_q=True,
            sess=None,
    ):
        self.idx = idx
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.double_q = double_q    # decide to use double q or not

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features*2+2))
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1'+str(self.idx)):
                w1 = tf.get_variable('w1'+str(self.idx), [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1'+str(self.idx), [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            with tf.variable_scope('l2'+str(self.idx)):
                w2 = tf.get_variable('w2'+str(self.idx), [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2'+str(self.idx), [1, self.n_actions], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l1, w2) + b2
            return out
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)

        if not hasattr(self, 'q'):  # record action value it gets
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q*0.99 + 0.01 * np.max(actions_value)
        self.q.append(self.running_q)

        if np.random.uniform() > self.epsilon:  # choosing action
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            #print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],    # next observation
                       self.s: batch_memory[:, -self.n_features:]})    # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            max_act4next = np.argmax(q_eval4next, axis=1)        # the action that brings the highest value is evaluated by q_eval
            selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
        else:
            selected_q_next = np.max(q_next, axis=1)    # the natural DQN

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


def parse_args():  #
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")  # ？
    parser.add_argument("--num-episodes", type=int, default=20000, help="number of episodes")
    parser.add_argument("--good-policy", type=str, default="mdddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=250, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--exp-name", type=str, default="test1", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="../policy/policy",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=10,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")
    return parser.parse_args()

def get_trainers(env, obs_shape_n, arglist):
    trainers = []  # agent集合
    trainer = DoubleDQN  # agent
    for i in range(env.n):  # env.n表示环境中的总机器人个数
        trainers.append(trainer(10,3,i))
    return trainers  # 返回实验中的训练机器人集合，好的和坏的agent集合，使用maddpg训练


def train(arglist,obs_shape,agent_num,act_space_dim):
    if(arglist.restore == False):
        inp = input("Do you want to restore model?\n\r[y/n]")
    if(inp == 'y'):
        arglist.restore = True
    if arglist.display == False:
        print("Begin to Train")
    else:
        print("Display")

    with U.single_threaded_session():
        # Create environment Important to config its here
        env = environment.NetworkEnviroment(obs_shape,agent_num,act_space_dim)
        # Create agent trainers
        obs_shape_n = [env.obs_shape for i in range(env.n)]  # 元组集合，每个元组是一个agent的obs
        #  n个agents观测空间的集合,tuple 不允许add/del以及修改
        trainers = get_trainers(env, obs_shape_n, arglist)
        print('Using good policy REINFORCE ')

        # Initialize global variables
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)  # 从给定的dir加载参数表


        episode_rewards = [0.0]  # sum of rewards for all agents
        steps = [0]
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        obs_n = env.start()[1]
        saver = tf.train.Saver()
        episode_step = 0  # 第几个episode
        train_step = 0  # 该episode进行到第几步
        t_start = time.time()
        # training process
        process_json_restore = False
        training_process = "../../trainingprocess/process.json"

        best_policy = "../../trainingprocess/BestAction.json"
        colletct_best_policy = False
        best_reward = 1000
        '''Use temporally'''


        if process_json_restore != True:
            trainingdata = {"episodes":[],"mean_steps":[],"mean_rewards":[]}
            with open(training_process, 'w') as file_obj:
                json.dump(trainingdata, file_obj)
            print("Create a new process.json")
        else:
            print("Use the process.json mathing the model")

        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.choose_action(obs) for agent, obs in zip(trainers, obs_n)]
            # 所有机器人的动作集合
            # environment step
            new_obs_n, rew_n, done_n = env.step(action_n,train_step)  # delete info_n
            for i,RL in enumerate(trainers):
                RL.store_transition(obs_n[i], action_n[i], rew_n[i],new_obs_n[i])
            episode_step += 1
            # print(episode_step)
            done = False
            for i in done_n:
                if i == True:
                    done = i
                    break
            terminal = (episode_step >= arglist.max_episode_len)  # 第几个episode
            # collect experience
            obs_n = new_obs_n  # 获得新的观察

            for i, rew in enumerate(rew_n):  # 对奖励集合中的奖励
                episode_rewards[-1] += rew  # 所有episode获得的奖励集合，每个元素为其中一个episode获得的奖励和
                agent_rewards[i][-1] += rew  # i个agent，i个元素，每个元素是i号agent的n个episode的reward构成的一维向量
            # print(episode_step)
            if done or terminal:  # 防止episode不结束
                if  env.trainstep > best_reward and colletct_best_policy:
                    best_reward = env.trainstep
                    U.save_state("../bestpolicy/bestpolicy", saver=saver)
                    print("Save as Best Policy with steps = ", env.trainstep)
                    with open(best_policy, 'w') as file:
                        json.dump(env.collect_netstatus, file)
                for i, RL in enumerate(trainers):
                    RL.learn()
                steps[-1] = episode_step
                # print("Reset")
                obs_n = env.reset()
                episode_step = 0
                steps.append(0)
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])



            # increment global step counter
            train_step += 1

            '''for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue
            '''
            # save model, display training output
            if (done or terminal) and (len(episode_rewards) % arglist.save_rate == 0):
                #U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                meanstep = np.mean(steps[-arglist.save_rate:])
                print("steps: {}, episodes: {}, mean episode reward: {}, mean step: {}, time: {}".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                    np.mean(steps[-arglist.save_rate:]),
                    round(time.time() - t_start, 3)))

                f = open(training_process, "r", encoding='utf-8')
                jsonf = json.load(f)
                f.close()
                with open(training_process, 'w') as file_obj:
                    jsonf["episodes"].append(len(episode_rewards))
                    jsonf["mean_steps"].append(np.mean(steps[-arglist.save_rate:]))
                    jsonf["mean_rewards"].append(np.mean(episode_rewards[-arglist.save_rate:]))
                    json.dump(jsonf,file_obj)


                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist,(3,), 4, 5) # obs_shape,agent_num,act_space

