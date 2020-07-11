"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

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

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),

        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),

        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs


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
    parser.add_argument("--save-rate", type=int, default=50,
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
    trainer = PolicyGradient  # agent
    for i in range(env.n):  # env.n表示环境中的总机器人个数
        trainers.append(trainer(10,3))
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
        env = environment.NetworkEnviroment(obs_shape,agent_num,act_space_dim,True)
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
                RL.store_transition(new_obs_n[i], action_n[i], rew_n[i])
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
                U.save_state(arglist.save_dir, saver=saver)
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
