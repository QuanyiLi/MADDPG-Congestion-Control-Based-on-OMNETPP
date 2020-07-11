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

#include "Server.h"
#include "message.h"
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <random>
#include <time.h>
#include <stdlib.h>
#include <math.h>
double gaussrand()
{
    static double V1, V2, S;
    static int phase = 0;
    double X;

    if (phase == 0) {
        do {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;

            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while (S >= 1 || S == 0);

        X = V1 * sqrt(-2 * log(S) / S);
    }
 else
     X = V2 * sqrt(-2 * log(S) / S);

     phase = 1 - phase;

     return X;
}
Define_Module(Server);

void Server::initialize(){
    timeuse = registerSignal("Time");
    mean_time = 0;
    avilable = true;
    start_time = par("start_time");
    end_time = par("end_time");
    // send_time_interval_max = par("send_time_interval_max");
    // send_time_interval_min = par("send_time_interval_min");
    packet_len = par("packet_len");
    routername = par("SwitchName");
    servername = par("ServerName");
    sendpkt(1);
    count = 0;
    recvnum = 0;
    stddev_norm = par("stddev");
    sendtimeinterval1 = par("send_time_interval_1");
    sendtimeinterval2 = par("send_time_interval_2");
    sendnum = 0;
    episode = 1;
    EV_INFO << "1:" <<sendtimeinterval1<<std::endl;
    EV_INFO << "2:" <<sendtimeinterval2<<std::endl;
    // normal_pb = new std::normal_distribution<double>(mean,stddev);
}

void Server::reset(){
    episode += 1;
    recvnum = 0;
    sendnum = 0;
}
/*
double Server::truncnormal(double low){
    double x = (*normal_pb)(generator);
    while(x < low){
        x = (*normal_pb)(generator);
    }
    return x;
}
*/
void Server::sendpkt(double start){
    std::string msgname1 = "pkt";
    const char* pkt1 = msgname1.c_str();
    message* msg1 = new message(pkt1, 0, packet_len);
    msg1->setType(1);

    double z1 = gaussrand();
    double z2 = gaussrand();

    // EV << "Z1:"<< z1 << std::endl;
    // EV << "Z2:"<< z2 << std::endl;

    double timeinterval1 = z1*stddev_norm+sendtimeinterval1>0?(z1*stddev_norm+sendtimeinterval1>0.1?0.1:z1*stddev_norm+sendtimeinterval1):0.001;
    double timeinterval2 = z2*stddev_norm+sendtimeinterval2>0?(z2*stddev_norm+sendtimeinterval2>0.1?0.1:z2*stddev_norm+sendtimeinterval2):0.001;

    std::string msgname2 = "pkt";
    const char* pkt2 = msgname2.c_str();
    message* msg2 = new message(pkt2, 0, packet_len);
    msg2->setType(1);

    // hardcode it's bad to do like this, will refactor
    //TODO refactor
    if(strcmp("Server1",getName())==0){
        // EV <<"Server1:2578"<<std::endl;
        SetRouteforPkt(msg1,"234","2");
        scheduleAt(start+simTime()+timeinterval1, msg1);
        SetRouteforPkt(msg2,"214","2");
        scheduleAt(start+simTime()+timeinterval2, msg2);
    }
    else if(strcmp("Server2",getName())==0){
        // EV <<"Server2:4865"<<std::endl;
        SetRouteforPkt(msg1,"412","1");
        scheduleAt(start+simTime()+timeinterval1, msg1);
        SetRouteforPkt(msg2,"432","1");
        scheduleAt(start+simTime()+timeinterval2, msg2);
    }
    else if(strcmp("Server3",getName())==0){
            // EV <<"Server2:4865"<<std::endl;
        SetRouteforPkt(msg1,"143","4");
        scheduleAt(start+simTime()+timeinterval1, msg1);
        SetRouteforPkt(msg2,"123","4");
        scheduleAt(start+simTime()+timeinterval2, msg2);
    }
    else if(strcmp("Server4",getName())==0){
            // EV <<"Server2:4865"<<std::endl;
        SetRouteforPkt(msg1,"341","3");
        scheduleAt(start+simTime()+timeinterval1, msg1);
        SetRouteforPkt(msg2,"321","3");
        scheduleAt(start+simTime()+timeinterval2, msg2);
    }
}


void Server::handleMessage(cMessage *msg){
    message* Mymsg = check_and_cast<message*>(msg);
    if(!Mymsg->isSelfMessage()){
        if(Mymsg->getType()==2){
            int sender_gate_ID=Mymsg->getSenderGate();
            std::string sender = Mymsg->getFromModule();
            delete Mymsg;
            char msgname[20] = "control_msg_back";
            message* Newmsg = new message(msgname, 0, packet_len);
            Newmsg->setType(3);
            Newmsg->setFromModule(getName());
            Newmsg->target_router = sender;
            Newmsg->own_send_gateID = sender_gate_ID;
            Newmsg->setSenderGate(0);
            scheduleAt(simTime()+ctl_msg_reply_delay, Newmsg);
        }
        else if(Mymsg->getType()==1){
            double time = simTime().dbl()-Mymsg->sendtime;
            emit(timeuse, time);
            recvnum++;
            bubble("Message is received!");
            delete Mymsg;
        }
    }
    else if(Mymsg->isSelfMessage()){
        if(Mymsg->getType()==3){
            send(msg, "gate$o",Mymsg->getSenderGate());
        }
        if(Mymsg->getType()==1){
            count += 1;
                if(count == 2){
                    sendpkt(0);
                    count = 0;
                }
            Mymsg->RoutersOnPath.pop();
            sendnum++;
            Mymsg->sendtime = simTime().dbl();
            send(msg, "gate$o", 0);
        }
    }
}

void Server::SetRouteforPkt(message* pkt,std::string router_seq, std::string dst){
    for(int i=0;i<router_seq.length();i++){
        pkt->RoutersOnPath.push(std::string(routername)+router_seq[i]);
        // EV<<"Debug routepath:"<< routername+router_seq[i] <<std::endl;
    }
    pkt->RoutersOnPath.push(std::string(servername)+dst);
}

void Server::refreshDisplay() const
{
    std::string status= "false";
    if(avilable){
        status = "true";
    }

    char buf[100];
    sprintf(buf, "send: %d\nrecv:%d\n",sendnum,recvnum);
    getDisplayString().setTagArg("t", 0, buf);
}


