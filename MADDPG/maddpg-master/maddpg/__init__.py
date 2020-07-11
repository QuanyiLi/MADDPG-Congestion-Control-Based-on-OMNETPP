class AgentTrainer(object):#作为MADDPGtrainer的父类，以下函数在子类中被重构
    def __init__(self, name, model, obs_shape, act_space, args):
        raise NotImplemented()

    def action(self, obs):
        raise NotImplemented()#如果这个方法行不通就找别的方法来完成https://www.jianshu.com/p/a8613baefa30

    def process_experience(self, obs, act, rew, new_obs, done, terminal):
        raise NotImplemented()

    def preupdate(self):
        raise NotImplemented()

    def update(self, agents):
        raise NotImplemented()