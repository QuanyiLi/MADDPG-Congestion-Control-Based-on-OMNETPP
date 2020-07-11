//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
// 

#include <stdlib.h>
#include "TrainingCenter.h"
#include <string>
#include <iostream>
#include <fstream>
#include "json/json.h"
#include "ForwardNode.h"
#include <stdio.h>
#include <string.h>
#include <queue>
#include "Server.h"
#include <unistd.h>
Define_Module(TrainingCenter);

int file_exist(const char* file_path){
    if(access(file_path,F_OK)==0){
        return 0;
    }
    return -1;
}
Define_Module(TrainingCenter);
void TrainingCenter::initialize(){
    allrecv = registerSignal("recv");
    allsend = registerSignal("send");
    collectdata = par("collectdata");
    testdata = par("testdata");
    base_test = par("basetest");
    NetStatus = new std::map<std::string,Status>;
    SwitchNum = par("SwitchNum");
    para_name = par("CollectModuleName");
    CollectModuleName = std::string(para_name);
    collect_interval = par("collect_interval");
    char msgname[20] = "collect_data_timer";
    cMessage* msg = new cMessage(msgname);
    scheduleAt(omnetpp::SimTime::ZERO+collect_interval, msg);
    AgentNum = SwitchNum;
    if(testdata){
        jsonos.open("testact.json");
    }
    if(base_test){
        to_search="./dataexchange/Actions_base.json";
    }
    else{
        to_search="./dataexchange/Actions.json";
    }
    long ok;
    ok = file_exist(obs_ok);
    // clear
    if(ok != -1){
        std::remove(obs_ok);
    }
}

void TrainingCenter::handleMessage(cMessage *msg)
{
    long all_recv = 0;
    long all_send = 0;
    for(int i = 0;i<Servers.size();i++){
        all_recv += Servers[i]->recvnum;
        all_send += Servers[i]->sendnum;
    }
    emit(allrecv,all_recv);
    emit(allsend,all_send);
    // ^ test use
    if(testdata){
        if(simTime() == 500){
            jsonos<<jsonwriter.write(testroot);
            jsonos.close();
        }
    }
    // ^ test data
    if(msg->isSelfMessage()){
        if(count == 0){
            for(int i=1;i<=SwitchNum;i++){
                std::string tmpName = std::string("Network.")+CollectModuleName+std::to_string(i)+std::string(".ForwardNode");
                cModule* Module = this->getModuleByPath(tmpName.c_str());
                FWN* fn = check_and_cast<FWN*>(Module);
                Switchs[std::string(para_name)+std::to_string(i)]=fn;
                int gatenum = fn->gateSize("gate")-1; // this connection should be ignored, thus -1
                Space[i].ActionNum = fn->getpeers();
                Space[i].StateNum = 2+fn->getpeers();
                Space[i].GateNum = gatenum;
                EV << "Switch" <<i<<" 's (S,A):"<<"("<< Space[i].StateNum <<","<<Space[i].ActionNum<<")"<<std::endl;

                tmpName = std::string("Network.")+"Server"+std::to_string(i);
                Module = this->getModuleByPath(tmpName.c_str());
                Server* sv = check_and_cast<Server*>(Module);
                Servers.push_back(sv);
            }
        }
        count++;
        scheduleAt(simTime()+collect_interval, msg);
        CollectDateNow();
        CommunicateByJson();
    }
}

void TrainingCenter::CollectDateNow(){
    for(std::map<std::string,FWN*>::iterator i = Switchs.begin();i != Switchs.end();i++){
        i->second->Collect_data();
        (*NetStatus)[i->first]=*(i->second->stats);
    }
    EV << "Collect AT:" << simTime() << std::endl;
}

void TrainingCenter::CommunicateByJson(){
        Json::Value root;
        root["TimeInterval"] = Json::Value(collect_interval);
        root["SimTime"] = Json::Value(count);
        for(std::map<std::string,Status>::iterator i = NetStatus->begin();i != NetStatus->end();i++){
            Json::Value partner;
            partner["Name"] = Json::Value(i->first);
            for(int k = 0;k<i->second.peers.size();k++){
                partner["peers"].append(i->second.peers[k]);
            }
            partner["QueueLength"] = Json::Value(i->second.queuelength);
            partner["Overflow"] = Json::Value(i->second.overflow);
            partner["OBS"].append(i->second.queuelength);
            partner["DropProb"]=Json::Value(i->second.pkt_drop_prob);
            int ccc = 0;
            while(ccc < i->second.peers.size()){
                for(std::map<int,int>::iterator iter = i->second.dest_num_pair.begin();iter!=i->second.dest_num_pair.end();iter++)
                {
                    if(ccc == iter->first){
                        partner[std::string("gate")+std::to_string(iter->first)] = Json::Value(iter->second);
                        partner["OBS"].append(iter->second);
                        ccc++;
                    }
                }
            }
            root["NetStatus"].append(partner);
        }
        if(!base_test || collectdata){ // if training, do this
            long ok;
            while(true){
                ok = file_exist(obs_ok);
                if(-1 == ok){
                    Json::StyledWriter sw;
                    std::ofstream os;
                    os.open(obs_ok);
                    os << sw.write(root);
                    os.close();
                    break;
                }
                else{
                    usleep(5000);
                }
            }
        }

        // chock
        long handle;
        while(true){
            handle = file_exist(to_search);
            if(-1 == handle){
            }
            else{
                usleep(100000);
                if (ActionParser(std::string(to_search)) == true){
                    EV << "Get Action"<< std::endl;
                    if(!base_test){
                        std::remove(to_search);
                        break;
                    }
                }
            }
            handle = file_exist(reset);
            if(-1 == handle){
                if(base_test){
                    break;
                }
                continue;
            }
            else{
                usleep(10000);
                // ^ test exchange
                if(testdata){
                    testroot.append(Json::Value(true));
                }
                EV << "Reset!!" <<std::endl;
                std::remove(reset);
                for(std::map<std::string,FWN*>::iterator i = Switchs.begin();i != Switchs.end();i++){
                    i->second->reset();
                }
                for(int i = 0;i<Servers.size();i++){
                    Servers[i]->reset();
                }
                break;
            }
        }
}


bool TrainingCenter::ActionParser(std::string Jsonpath){
    Json::Reader reader;
    Json::Value root;
    std::ifstream in(Jsonpath, std::ios::binary);
    if (reader.parse(in, root)){
        if(root["Done"].isNull()){ // lock data
                return false;
        }

        for(std::map<std::string,FWN*>::iterator i = Switchs.begin();i != Switchs.end();i++){
            // continuous actions
            EV << "Module:" << i->first<<std::endl;
            i->second->ChangeST(root[i->first]);
            // discrete actions

            /*
            std::string action = Ten2Three(STimeinTernary,i->first);
            i->second->ChangeServiceTime(action);
            */
        }
        // ^ test data
        if(testdata){
            root["count"]=std::to_string(count);
            testroot.append(root);
        }
    }
    /*
    Json::StyledWriter sw;
    std::ofstream os;
    os.open("test.json");
    os << sw.write(root);
    os.close();
    */
    return true;
}

std::string TrainingCenter::Ten2Three(int num,std::string id){ // take care !! it doesn't reverse
    std::string res="";
    EV << "Action:" << num;
    std::stack<int> tmp;
    int count = Switchs[id]->getpeers();
    if(Switchs[id]->ConnectServer){
        tmp.push(0);
    }
    while(count>0){
        if(num != 0){
            tmp.push(num%3);
            num = num/3;
        }
        else{
            tmp.push(0);
        }
        count--;
    }
    while(!tmp.empty()){
        res+=std::to_string(tmp.top());
        tmp.pop();
    }
    EV << " res:" << res<<std::endl;
    return res;
}

void TrainingCenter::finish(){
    // _CrtDumpMemoryLeaks();
}




