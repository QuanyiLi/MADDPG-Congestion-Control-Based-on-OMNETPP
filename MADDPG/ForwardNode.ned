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
simple ForwardNode
{
    parameters:
        @display("i=abstract/router");
        int gate_num;
        int packet_length @unit(byte);
        string queueModule = default("^.queue");
        string module_name;
        double ini_service_time @unit(s);
        double min_service_time @unit(s);
        double service_time_step @unit(s);
    gates:
        output gate[gate_num];
        input in;
}