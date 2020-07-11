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

#include "DropTailqueue.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <map>

Define_Module(DropTailqueue);
void DropTailqueue::initialize()
{
    numQueueDropped = 0;
    QueueCapacity = par("QueueCapacity");
    queue = new std::queue<message*>;
    requestPacket();
    avilable = true;
}

void DropTailqueue::refreshDisplay() const
{
    std::string status= "false";
    if(avilable){
        status = "true";
    }

    char buf[100];
    // sprintf(buf, "queue length: %d\nqueue dropped: %d\nforward avilable: %s\ndrop in one step:%d\nrecv in one step:%d\n", queue->size(), numQueueDropped, status.c_str(),
    //        droppkt,allpkt);
    sprintf(buf, "queue length: %d\nqueue dropped: %d\n", queue->size(), numQueueDropped);
    cDisplayString& parentDispStr = getParentModule()->getDisplayString();
    parentDispStr.setTagArg("t", 0, buf);
}

 void DropTailqueue::requestPacket()
{
    Enter_Method("requestPacket()");
    if(!queue->empty()){
        message* pkt = queue->front();
        queue->pop();
        send(pkt,gate("out"));
    }
    else{
        avilable = true;
    }
}

void DropTailqueue::clear(){
    Enter_Method("clear()");
    int round = queue->size();
    for(int i = 0;i<round;i++ ){
        delete queue->front();
        queue->pop();
    }
    numQueueDropped = 0;
}

void DropTailqueue::handleMessage(cMessage *msg)
{
    allpkt+=1;
    message* pkt = check_and_cast<message *>(msg);
    if(avilable){
        avilable = false;
        send(pkt,gate("out"));
    }
    else{
        enqueue(pkt);
    }
}

void DropTailqueue::enqueue(message* pkt)
{
    if (queue->size() < QueueCapacity){
        queue->push(pkt);
    }
    else{
        // TODO add statistics here
        droppkt+=1;
        bubble("Overflow,Drop!!");
        numQueueDropped++;
        delete pkt;
    }
}

std::map<std::string,int> DropTailqueue::GatePktNumStat(){
    std::map<std::string,int> temp;
    int round = queue->size();
    for(int i=0;i<round;i++){
        std::string nxt_router = queue->front()->RoutersOnPath.front();
        queue->push(queue->front());
        queue->pop();
        if(temp.find(nxt_router) != temp.end()){
            temp[nxt_router] += 1;
        }
        else{
            temp[nxt_router] = 1;
        }
    }
    // using for debug
    //for(std::map<std::string,int>::iterator i = temp.begin();i!=temp.end();i++){
    //    EV<<i->first<<":"<<i->second<<std::endl;
    //}
    return temp;
}

double DropTailqueue::getdropprob(){
    double res=0;
    if(allpkt!= 0){
        res = droppkt/allpkt;
    }
    droppkt = 0;
    allpkt = 0;
    return res;
}


