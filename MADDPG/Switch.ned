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
module Switch
{
    parameters:
        @display("i=device/switch_vl;bgb=86.664,256.224");
        int packet_len @unit(byte);
        string name;
    gates:
        inout gate[];
    submodules:
        ForwardNode: ForwardNode {
            parameters:
                @display("p=405.06,346.656");
                gate_num = sizeof(gate);
                packet_length = packet_len;
                module_name = name;
        }
        queue: DropTailqueue {
            parameters:
                @display("p=768.672,339.12;q=l2queue");
                ingate_num = sizeof(gate);
        }
    connections:
        queue.out --> { @display("m=s"); } --> ForwardNode.in;
        for i=0..sizeof(gate)-1 {
            gate$i[i] --> { @display("m=s"); } --> queue.in[i];          
            ForwardNode.gate[i] --> { @display("m=s"); } --> gate$o[i];
        }
}
