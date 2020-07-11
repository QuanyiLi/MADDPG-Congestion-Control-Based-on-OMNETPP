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

#ifndef FORWARDNODE_H_
#define FORWARDNODE_H_

#include <stdio.h>
#include <string.h>
#include <omnetpp.h>
#include <map>
#include <string>
#include "message.h"
#include "DropTailqueue.h"
#include "TrainingCenter.h"
#include "json/json.h"
using namespace omnetpp;

class ForwardNode: public cSimpleModule {
protected:
    bool complete_map = false;
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
    std::map<std::string,int>* router_gate_map = nullptr;
    int count;
    void forward_msg(message* msg);
    void get_router_gate_pair(message* Mymsg);
    int packet_len;
    DropTailqueue* QueueModule = nullptr;
    const char* GetName(){return moduleName;}
    const char* moduleName;
    std::map<int,float>* gate_service_time = nullptr;
    double ini_service_time;
    double service_time_step; // control service rate
    double min_service_time; // message processing time
    int peers;
public:
    void reset();
    struct statistic* stats;
    bool ConnectServer = false;
    void ChangeServiceTime(std::string servicerates);
    int getpeers(){return peers;}
    void Collect_data();
    void ChangeST(Json::Value servicerates);
};

#endif /* FORWARDNODE_H_ */
