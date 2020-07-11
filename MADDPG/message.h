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

#ifndef MESSAGE_H_
#define MESSAGE_H_
#include <omnetpp.h>
#include <string>
#include <queue>

class message: public ::omnetpp::cPacket {
public:
    double sendtime;
    message(const char* name, short kind, int pkt_len);
    void setType(int ttype);
    void setFromModule(const char* name);
    int getType(){return type;}
    const char* getFromModule(){return router_name;}
    void setSenderGate(int gate){send_gate = gate;}
    int getSenderGate(){return send_gate;}
    int own_send_gateID;// only use in type 3, return the origin sender gate when receivers reply senders
    std::string target_router;// only use in type3, double check receiver of msg
    std::queue<std::string> RoutersOnPath;
private:
    int type; // 2,3 = initialize type, 1 = message type, 4 = collect data
    int send_gate;
    const char* router_name;
};

#endif /* MESSAGE_H_ */
