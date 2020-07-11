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



#ifndef SERVER_H_
#define SERVER_H_
#include <random>
#include <stdio.h>
#include <string.h>
#include <omnetpp.h>
#include "message.h"

using namespace omnetpp;

class Server: public cSimpleModule {
protected:
    double base_send_interval = 0.02; // base service time, in case of a very small svc time
    // double mean;
    // double stddev;
    simsignal_t timeuse;
    double sendtimeinterval1;
    double sendtimeinterval2;
    int episode;
    bool avilable;
    int count;
    void sendpkt(double start);
    double start_time;
    double end_time;
    // double truncnormal(double low);
    // double send_time_interval_1;
    // double send_time_interval_;
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
    // if the num of router >= 10, revise the function below !!!!!!!!!!!!
    virtual void refreshDisplay() const override;
    void SetRouteforPkt(message* pkt,std::string router_seq, std::string dst);
    double ctl_msg_reply_delay=0.1;
    int packet_len;
    const char* routername;
    const char* servername;
    double stddev_norm;
    double mean_time;
public:
    void reset();
    int recvnum;
    int sendnum;
};

#endif /* SERVER_H_ */
