import json
import os
import numpy as np
import time
import gym
import io


class NetworkEnviroment(object):
    def __init__(self, obs_shape, num, act_space_n,baseline):
        self.obs_space = obs_shape[0]
        self.obs_shape = obs_shape
        self.n = num
        # self.action_space = [gym.spaces.Discrete(act_space_n) for i in range(num)]
        self.queue_length = 1000
        self.min_service_rate = 0.015
        self.old_obs = 0
        self.workdirector = "../../dataexchange"
        self.obs_name = "NetStatus.json"
        self.act_name = "Actions.json"
        self.reset_name = "../../dataexchange/Reset.json"
        self.collect_path = "../../trainingprocess/CollectInfo.json"
        self.collectdata = False
        self.What2Collect = "NetStatus" # or "NetStatus"
        self.collect_netstatus = {}
        self.trainstep = 0
        self.readAction = False
        self.ActionJson = "../../trainingprocess/BestAction.json"
        self.jsonf =0
        self.PGorDQN = baseline
        self.act_space = act_space_n if self.PGorDQN else [gym.spaces.Discrete(act_space_n)
        for i in range(num)]
        # self.max_done_count = 5
        # self.done_count = [0 for i in range(num)]

    def start(self):
        if self.readAction:
            f = open(self.ActionJson, "r", encoding='utf-8')
            self.jsonf = json.load(f)
            f.close()
        self.trainstep = 0
        self.collect_netstatus = {"Episode":[]}
        for root, dirs, files in os.walk(self.workdirector, topdown=True):
            if self.act_name in files:
                os.remove(os.path.join(self.workdirector,self.act_name))
        obs = self.LoadStatus(self.workdirector, self.obs_name, self.obs_space)
        self.old_obs = obs[0]
        return obs

    def reset(self):
        self.trainstep = 0
        with open(self.collect_path,'w') as file:
            json.dump(self.collect_netstatus,file)
        self.collect_netstatus = {"Episode":[]}
        # reset need to do two times
        dict = {"reset": 1}
        with open(self.reset_name, 'w') as file_obj:
            json.dump(dict, file_obj)
        dict = {"reset": 1}
        with open(self.reset_name, 'w') as file_obj:
            json.dump(dict, file_obj)
        json_obj, new_obs = self.LoadStatus(self.workdirector, self.obs_name, self.obs_space)
        # print(new_obs)
        self.old_obs = json_obj
        return new_obs

    def get_reward(self, jsonf, act):
        reward_n = []
        netstatus = jsonf["NetStatus"]
        old = self.old_obs["NetStatus"]
        for k, switch in enumerate(netstatus):
            reward = 0
            result_factor = 0 # queue diff
            for i, peer in enumerate(switch["peers"]):  # i,peer gate_router_pair
                idx = self.get_switch_idx(peer) - 1  # peer's idx
                # 2号是给服务器的端口速率，也是正态分布流量的均值
                # obs的idx1 为去往 gate0的pkt数目
                reward += (act[k][i] - act[k][2]) * (netstatus[idx]["QueueLength"] - switch["QueueLength"]) * \
                          switch["OBS"][i + 1]
            reward = reward / self.queue_length

            '''
            for i, peer in enumerate(switch["peers"]):  # i,peer gate_router_pair
                idx = self.get_switch_idx(peer) - 1  # peer's idx
                # act[2] = baseline
                # obs的idx1 为去往 gate0的pkt数目
                if switch["QueueLength"]!= 0:
                    result_factor += 1-(abs(netstatus[idx]["QueueLength"] - switch["QueueLength"])/switch["QueueLength"])
                else:
                    result_factor  = 1

                p = (old[idx]["QueueLength"] - old[k]["QueueLength"])*(act[k][i] - act[k][2])

                if p < 0:
                    right_factor = -1
                else:
                    right_factor = 1


                if old[k]["QueueLength"] == 0:
                    relate_factor = 1
                else:
                    relate_factor = (old[k]["OBS"][i + 1] / old[k]["QueueLength"])
                reward +=  right_factor * relate_factor * result_factor
            '''
            reward_n.append(reward)
        return reward_n

    def step(self, action_n, t):
        self.trainstep += 1
        act = []

        if self.readAction==False:
            for action in action_n:
                if self.PGorDQN:
                    rateall = (float(action)/10,float(action)/10,float(self.min_service_rate))
                else:
                    rate1 = abs(action[0] - action[1])
                    rate2 = abs(action[2] - action[3])
                    rateall = [float(rate1), float(rate2), float(self.min_service_rate)]
                act.append(rateall)
            self.DistributeAction(act, os.path.join(self.workdirector, self.act_name), t)
        # print(act)
        else:
            self.RdAction(os.path.join(self.workdirector, self.act_name))
        json_obj, new_obs = self.LoadStatus(self.workdirector, self.obs_name, self.obs_space)
        done_n = self.get_done(json_obj)
        if self.readAction:
            reward_n = [ i*0 for i in range(0,self.n)]
        else:
            reward_n = self.get_reward(json_obj, act)
        self.old_obs = json_obj
        return new_obs, reward_n, done_n

    def RdAction(self,targetpath):
        dict = self.jsonf["Episode"][self.trainstep]
        with open(targetpath, 'w') as file_obj:
            json.dump(dict, file_obj)


    def get_done(self, jsonobj):
        done = []
        netstatus = jsonobj["NetStatus"]
        for i, switch in enumerate(netstatus):
            if switch["Overflow"] == 1:
                done.append(True)
            else:
                done.append(False)
        return done

    def get_switch_idx(self, name):
        return int(name[len("Switch"):])

    def LoadStatus(self, file_path, name, clip):
        find = False
        while not find:
            for root, dirs, files in os.walk(file_path, topdown=True):
                if name in files:
                    find = True
                    time.sleep(0.01)
                    break
                # else:
                # time.sleep(0.005)
        obs_n = []
        jf = []
        f = open(os.path.join(file_path, name), "r", encoding='utf-8')
        jsonf = json.load(f)
        f.close()
        try:
            os.remove(os.path.join(file_path, name))
        except PermissionError:
            print("Permission Error！")
        jf.append(jsonf)
        netstatus = jsonf["NetStatus"]
        for i in netstatus:
            obs_n.append(np.asarray(i["OBS"][0:clip]))
        # debug data exchange
        #print(jsonf["SimTime"])
        #print(netstatus[1]["OBS"][0])
        if self.collectdata == True and self.What2Collect == "NetStatus":
            self.collect_netstatus["Episode"].append(jsonf)
        return jf[0], obs_n

    def DistributeAction(self, action_n, filepath, t):
        dict = {}
        for i in range(0, len(action_n)):
            target_module = "Switch" + str(i + 1)
            dict[target_module] = action_n[i]
        dict["Done"] = True
        dict["step"] = t
        with open(filepath, 'w') as file_obj:
            json.dump(dict, file_obj)
        if self.collectdata == True and self.What2Collect == "Actions":
            self.collect_netstatus["Episode"].append(dict)

