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


simple Server
{
    parameters:
        @display("i=device/pc2_vl");
        double send_time_interval_1 @unit(s);
        double send_time_interval_2 @unit(s);
        double start_time @unit(s);
        double end_time @unit(s);
        int packet_len @unit(byte);
        string SwitchName;
        string ServerName;
        double stddev;
        @signal[Time](type="double");
        @statistic[transmit_time](title="transmit_time"; source="Time"; record=count,mean, max, min,vector);
        
    gates:
        inout gate[];
}



