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

#ifndef DROPTAILQUEUE_H_
#define DROPTAILQUEUE_H_

#include <queue>
#include "message.h"
#include <map>
using namespace omnetpp;
class DropTailqueue: public cSimpleModule{
private:
    // configuration
    int QueueCapacity;
    // queue
    std::queue<message*>* queue = nullptr;
    // status
    bool avilable;
    int numQueueDropped;
    int allpkt = 0;
    int droppkt = 0;
protected:
    void initialize() override;
    void handleMessage(cMessage *msg) override;
    void enqueue(message* pkt);
public:
    std::map<std::string,int> GatePktNumStat();
    void requestPacket();
    bool UpperlayerStatus(){return avilable;}
    virtual void refreshDisplay() const override;
    void clear();
    int length(){return queue->size();};
    int overflow(){return queue->size() == QueueCapacity?1:0;}
    double getdropprob();
};

#endif /* DROPTAILQUEUE_H_ */
