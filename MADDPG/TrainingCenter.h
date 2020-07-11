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

#ifndef TRAININGCENTERR_H_
#define TRAININGCENTERR_H_
#include <omnetpp.h>
#include <vector>
#include "ForwardNode.h"
#include <iostream>
#include <fstream>
#include "json/json.h"
using namespace omnetpp;

/***

 Actually this is an interface with MADDPG Algorithm, some usage are informal but convenient

 ***/
struct statistic{
    std::vector<std::string> peers;
    double pkt_drop_prob;
    std::map<int,int> dest_num_pair;
    int queuelength;
    int overflow; // 1 overflow
};
typedef statistic Status;
typedef class ForwardNode FWN;

// Useless now, config it in python source code
typedef struct SpaceType{
    int ActionNum;
    int StateNum;
    int GateNum;
} space;
typedef class Server SV;
class TrainingCenter:public cSimpleModule {
public:
    Json::Value testroot;
    Json::StyledWriter jsonwriter;
    std::ofstream jsonos;
    std::vector<SV*> Servers;
    double collect_interval;
    int SwitchNum;
    int AgentNum;
    std::map<std::string,Status>* NetStatus;
    std::string CollectModuleName;
    std::map<std::string,FWN*> Switchs;
    std::map<int,space> Space;
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
    // void CollectAfter(omnetpp::SimTime starttime);
    void CollectDateNow();
    void CommunicateByJson();
    bool ActionParser(std::string Jsonpath);
    std::string Ten2Three(int num,std::string id);
    int count = 0;
    const char* para_name;
    bool testdata; // test the data exchange with python
    bool base_test; // if true, this program can run without python when Actions_base.json exists
    bool collectdata; // collect data used to plot when display results
    const char* to_search;
    const char* obs_ok = "./dataexchange/NetStatus.json";
    const char* reset = "./dataexchange/Reset.json";
    simsignal_t allrecv;
    simsignal_t allsend;
protected:
    virtual void finish() override;
};

#endif /* TRAININGCENTERR_H_ */
