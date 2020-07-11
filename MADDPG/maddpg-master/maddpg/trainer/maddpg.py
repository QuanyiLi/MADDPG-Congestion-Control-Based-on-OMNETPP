import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U  # tensorflow常用函数

from maddpg.common.distributions import make_pdtype  # 常用的概率分布
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer  # 经验回放池


def discount_with_dones(rewards, dones, gamma):  # 计算总的折扣回报,疑似未使用
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r
        r = r * (1. - done)  # 1.0-done
        discounted.append(r)
    return discounted[::-1]  # 倒序返回


def make_update_exp(vals, target_vals):  # softupdate两个神经网络的变量
    polyak = 1.0 - 1e-2  # 0.99
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0 - polyak) * var))  # target网络和当前网络
    expression = tf.group(*expression)  # expression被会话调用，其中所有变量均会被调用生效
    return U.function([], [], updates=[expression])  # 更新target_vals网络节点的值
    # expression一组对变量的操作的集合

    # 疑似缺少每个agent维护了多个policy部分
    # 策略网络，双网络，同时拟合了对自己输出行为的改善与预测其他agent的行为，仅可以访问到自己的观测值


def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False,
            num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):  # 重用变量
        # create distribtuions初始动作概率分布列表
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]  # 为所有agent的动作空间都创造一个动作概率分布类
        # 类的集合
        # set up placeholders
        obs_ph_n = make_obs_ph_n  # 所有的agent观察到的环境信息
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        # 返回用于存放每个agent的动作的占位符集合，用于填充所有agent选择的动作[none]代表可以填入无数组数据
        p_input = obs_ph_n[p_index]  # 仅观察到自身周围环境

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        # 建立神经网络，输出单元数为动作个数...这代码写的太呆了 输出每一个动作的值
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))
        # 获取该神经网络全部变量
        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)
        act_sample = act_pd.sample()  # 确定性动作叠加噪声进行探索，成为随机策略，得到一组act，作用未知
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))  # flatparam是所有动作的actor网络输出值的集合
        # 猜测引入p_reg是因为预测其agent动作的需要
        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()  # 仅替换自己的动作输入，自己的动作来自于自己的policy网络输出
        # 所以通过这一步将两个网络连接，通过q网络优化自己的policy网络
        q_input = tf.concat(obs_ph_n + act_input_n, 1)  # q输入所有的环境观察值与所有的agents采取的动作
        # q的输入
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:, 0]
        # 这里是用的q_func由于reuse所以使用已经创建好的变量，即自己的q网络而不是再创建一个
        # q_train,p_train属于同一个scope！
        # 策略优化目标
        pg_loss = -tf.reduce_mean(q)  # loss与p_reg均需要加-号进行优化
        # 目标使q的均值最大，等于采样后的-reduce_mean最小
        loss = pg_loss + p_reg * 1e-3  # 引入熵？
        # 梯度下降优化器节点表达式
        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions可调用函数，批量使用session训练
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)  # 依据自身观察给出确定性动作
        p_values = U.function([obs_ph_n[p_index]], p)  # 输出的是动作值集合

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func",
                          num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}

    # q值预测网络,双网络，包括更新，可以访问全部agent的观测值与采取的动作，上帝视角评分


def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False,
            scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        # make_ob_ph_n是输入的placeholder，与obs_n同shape
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]  # 获取概率类型，传入动作维度（5）
        # act_space来自于env.act_space,由实验环境决定
        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")  # 一维输入占位符
        # 以上为三个placeholder, [None]增加维度，不知道喂进去多少数据时使用, 即None是batchsize大小

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)  # q函数输入网络为动作加上环境，在1维上，即q网络输入是所有agent观察和动作
        if local_q_func:  # 用ddpg时即只用自己的行为训练
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:, 0]  # 取所有行的第0个数据
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))
        # q网络变量集合
        q_loss = tf.reduce_mean(tf.square(q - target_ph))  # target_ph 会被什么占据呢？ 会被喂进去的td target占据
        # q网络的损失函数，均方差，target_ph来自于target网络的预测
        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss  # + 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)
        # 优化器表达式，以及是否梯度clip
        # Create callable functions
        # theano function
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:, 0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)
        # 以下返回值均为theano function可以直接填入传入placeholder的参数
        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


class MADDPGAgentTrainer(AgentTrainer):  # model是采用的神经网络模型的输出，即神经网络模型
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):  # 是否用ddpg训练
        self.name = name
        self.n = len(obs_shape_n)  # 总的agent个数
        self.agent_index = agent_index  # 当前是几号agent
        self.args = args  # cmd传入的训练参数，交互用
        obs_ph_n = []
        for i in range(self.n):  # 用于一批环境数据放入的占位符集合，收集所有agent的observations，
            # 依据他们observation的shape创造不同大小的批量占位符集合
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation" + str(i)).get())

        # Create all the functions necessary to train the model
        # 训练节点，更新target网络，字典得到对应输出的q值与target-q值(已经被session激活)
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )  # 得到act，训练策略网络，策略网络的target网络更新，字典给出p值和target策略网络的输出动作值
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def action(self, obs):  # 选择动作
        return self.act(obs[None])[0]  # 返回的是一组act中的第一个,[None]表示数组的一维

    def experience(self, obs, act, rew, new_obs, done, terminal):  # 为该agent收集经验
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None  # batch大小数组置空

    def update(self, agents, t):  # 经验回放训练该agent,100步才训练，train step为100倍数
        if len(self.replay_buffer) < self.max_replay_buffer_len:  # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)  # 一个batch大小的数组，存放随机生成的index
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index  # index数组
        for i in range(self.n):  # 从每个agent的经验池采样
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)  # 采样一个agent的一批经验
            obs_n.append(obs)  # n批经验构成的二维数组，每个元素为一个agent的一次对环境的观察，下同理
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)  # 采样自己的经验，和上面的采样相同
        # 最后obs_n中有n个元素，每个元素表示一个agent的一个batch_size大小的经验集合
        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            # debug是一个字典，值是函数
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
            # target网络预测的所有agent的下一个行为！！！！！！！！！每个agent维护一个用于预测其余agent动作的神经网络
            # 这里用于选择自己的policy的神经网络与预测别人行为的网络是同一个
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            # target网络预测的下一个状态的所有agent的q值集合！！！！！！！！！！
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
            # 上式为target-q值，

        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))  # 仅用当前经验完成对q值的训练

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))  # 填入所有观测值是因为用于q网络对p网络的改善，选择行为时实际上只用p网络

        self.p_update()  # 每一次完成神经网络训练后softupdate target网络的值,其实是100步才会执行一次update
        self.q_update()  # 同上
        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
