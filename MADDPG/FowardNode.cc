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
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <omnetpp.h>
#include "message.h"
#include <map>
#include <string>
#include "ForwardNode.h"
#include "TrainingCenter.h"
#include "json/json.h"

Define_Module(ForwardNode);

void ForwardNode::initialize(){
    peers = 0;
    moduleName = par("module_name");
    min_service_time = par("min_service_time");
    service_time_step = par("service_time_step");
    ini_service_time = par("ini_service_time");
    router_gate_map = new std::map<std::string,int>;
    gate_service_time = new std::map<int,float>;
    count = 0;
    packet_len = par("packet_length");
    cModule* Module = this->getModuleByPath("^.queue");
    QueueModule = check_and_cast<DropTailqueue*>(Module);
    stats = new statistic;
    for(int i=0 ; i<gateSize("gate") ; i++){
        char msgname[20] = "control_msg_ini";
        message* msg = new message(msgname, 0, packet_len);
        msg->setType(2);
        msg->setSenderGate(i);
        msg->setFromModule(GetName());
        scheduleAt(0.0, msg);
    }
    // char msgname[20] = "collect_data_timer";
    // message* msg = new message(msgname, 0, packet_len);
    // msg->setType(4);
    // scheduleAt(0.0, msg);
}

void ForwardNode::handleMessage(cMessage *msg){
    message* Mymsg = check_and_cast<message *>(msg);
    if (!Mymsg->isSelfMessage() && !complete_map && Mymsg->getType()!=1){
        get_router_gate_pair(Mymsg);
    }
    else if(Mymsg->isSelfMessage()){
        if(strcmp("Ask_for_new_packet", Mymsg->getName()) == 0){
            QueueModule->requestPacket();
            delete Mymsg;
        }
        else if(Mymsg->getType() == 1){
            send(Mymsg,"gate",Mymsg->getSenderGate());
        }
        else if(Mymsg->getType() == 2||Mymsg->getType() == 3){
            send(Mymsg,"gate",Mymsg->getSenderGate());
            QueueModule->requestPacket();
        }
    }
    else if (!Mymsg->isSelfMessage() && Mymsg->getType()==1 && complete_map){
        forward_msg(Mymsg);
    }
}

void ForwardNode::Collect_data(){
    std::map<std::string,int> temp = QueueModule->GatePktNumStat();
    // clear data of last time interval
    for (std::map<std::string,int>::iterator i = temp.begin();i!=temp.end();i++){
            stats->dest_num_pair[(*router_gate_map)[i->first]] = 0;
    }
    // collect new data
    for (std::map<std::string,int>::iterator i = temp.begin();i!=temp.end();i++){
        stats->dest_num_pair[(*router_gate_map)[i->first]] += i->second;
        // EV <<i->first<<","<< (*router_gate_map)[i->first]<<":" << i->second <<std::endl;
    }
    stats->pkt_drop_prob = QueueModule->getdropprob();
    stats->queuelength = QueueModule->length();
    stats->overflow = QueueModule->overflow();

    EV << "QueueModule->length()" << stats->queuelength << std::endl;
    // EV << "QueueModule->overflow()" << stats->overflow <<std::endl;

}

void ForwardNode::get_router_gate_pair(message* Mymsg){
    if(Mymsg->getType()==2){
        int sender_gate_ID=Mymsg->getSenderGate();
        std::string target_router = Mymsg->getFromModule();
        delete Mymsg;
        for(int i=0;i<gateSize("gate");i++){
            char msgname[20] = "control_msg_back";
            message* msg = new message(msgname, 0, packet_len);
            msg->setType(3);
            msg->setFromModule(GetName());
            msg->own_send_gateID = sender_gate_ID;
            msg->setSenderGate(i);
            msg->target_router = target_router;
            scheduleAt(simTime(), msg);
        }
    }
    else if(Mymsg->getType()==3){
        if(Mymsg->target_router == GetName()){
            std::string sender = Mymsg->getFromModule();
            int send_gate = Mymsg->own_send_gateID;
            EV <<"INFO:"<<GetName()<<"'s gate(ID:"<<send_gate<<")<----->"<<sender<<std::endl;
            (*router_gate_map)[sender] = send_gate;
            if (sender.find("Switch")!=sender.npos){
                peers+=1;
                stats->peers.push_back(sender);
            }
            if (sender.find("Server")!=sender.npos){
                ConnectServer = true;
            }
             // initialize service_rate
             (*gate_service_time)[send_gate] = ini_service_time;

             // initialize gate_num_pair
             stats->dest_num_pair[send_gate] = 0;
        }
        if(router_gate_map->size() == gateSize("gate")-1){ // -1 because of the training center
                complete_map = true;
                EV << "Routing Table is established"<<std::endl;
                for(std::map<std::string,int>::iterator i=router_gate_map->begin();i!=router_gate_map->end();i++){
                    EV<< i->first<<":"<< i->second<<std::endl;
                }
                // it's hardcode!!!!!!!!!!!!!!!!!!!!!! I am too lazy to design an algorithm
                int i =0;
                if((*router_gate_map)[stats->peers[i]] != 0){
                    std::string c = stats->peers[i];
                    stats->peers[i] = stats->peers[i+1];
                    stats->peers[i+1] = c;
                }

                QueueModule->clear();
        }
        QueueModule->requestPacket();
        delete Mymsg;
    }
}

void ForwardNode::forward_msg(message* Mymsg){
    std::string next_router = Mymsg->RoutersOnPath.front();
    Mymsg->RoutersOnPath.pop();
    int send_gate;
    if(router_gate_map->find(next_router)!=router_gate_map->end()){
        send_gate = (*router_gate_map)[next_router];
        Mymsg->setSenderGate(send_gate);
        scheduleAt(simTime()+(*gate_service_time)[send_gate],Mymsg);
        EV << "info:" << (*gate_service_time)[send_gate] << " by gate:"<< send_gate;
    }
    else{
        std::string error = "Can't find "+next_router+" in Routing Table";
        bubble(error.c_str());
    }
    char msgname[20] = "Ask_for_new_packet";
    message* msg = new message(msgname, 0, packet_len);
    scheduleAt(simTime()+(*gate_service_time)[send_gate]+min_service_time, msg);
}

void ForwardNode::ChangeServiceTime(std::string servicerates){
    // example: "2021" : port0 = 2*service_t_interval+base, port1 = 0+base ...
    for(std::map<int,float>::iterator iter = gate_service_time->begin();iter!=gate_service_time->end();iter++){
        iter->second = (servicerates[iter->first] - '0'+1)*service_time_step;
        EV << "gate" <<iter->first << ":" << iter->second << std::endl;
    }
}

void ForwardNode::ChangeST(Json::Value servicerates){
    for (std::map<std::string,int>::iterator i = router_gate_map->begin();i!=router_gate_map->end();i++){
        if(i->first.find("Switch")!=i->first.npos){
            (*gate_service_time)[i->second] = servicerates[i->second].asFloat()*service_time_step;
        }
        else if(i->first.find("Server")!=i->first.npos){
            (*gate_service_time)[i->second] = servicerates[2].asFloat()*service_time_step;
        }

    }
    for(std::map<int,float>::iterator iter = gate_service_time->begin();iter!=gate_service_time->end();iter++){
        EV << "gate" <<iter->first << ":" << iter->second << std::endl;

    }
}

void ForwardNode::reset(){
    QueueModule->clear();
}

