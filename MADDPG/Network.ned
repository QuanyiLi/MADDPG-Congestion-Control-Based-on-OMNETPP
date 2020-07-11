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
network Network
{
    parameters:
        @display("bgb=3001.486,2053");
    types:
        channel Channel extends ned.DelayChannel
        {
            delay = 100ms;
        }
    submodules:
        TrainingCenter: TrainingCenter {
            @display("p=1404.25205,1005.97006");
        }
        Server1: Server {
            @display("p=505.03802,505.03802");
        }
        Server2: Server {
            @display("p=2303.466,1506.9021");
        }
        Server3: Server {
            @display("p=505.03802,1506.9021");
        }
        Server4: Server {
            @display("p=2303.466,505.03802");
        }
        Switch1: Switch {
            @display("p=903.32,1506.9021");
            name = "Switch1";
        }
        Switch2: Switch {
            @display("p=903.32,505.03802");
            name = "Switch2";
        }
        Switch3: Switch {
            @display("p=1905.1841,505.03802");
            name = "Switch3";
        }
        Switch4: Switch {
            @display("p=1905.1841,1506.9021");
            name = "Switch4";
        }
    connections:

        // config connection among Switchs at first
        Switch2.gate++ <--> Channel <--> Switch1.gate++;
        Switch2.gate++ <--> Channel <--> Switch3.gate++;
        Switch3.gate++ <--> Channel <--> Switch4.gate++;
        Switch1.gate++ <--> Channel <--> Switch4.gate++;

        // Connection of Servers
        Server1.gate++ <--> Channel <--> Switch2.gate++;
        Server2.gate++ <--> Channel <--> Switch4.gate++;
        Server3.gate++ <--> Channel <--> Switch1.gate++;
        Server4.gate++ <--> Channel <--> Switch3.gate++;

		// the config of TC must implement at last
        TrainingCenter.gate++ <--> Switch2.gate++;
        TrainingCenter.gate++ <--> Switch3.gate++;
        TrainingCenter.gate++ <--> Switch1.gate++;
        TrainingCenter.gate++ <--> Switch4.gate++;
}
