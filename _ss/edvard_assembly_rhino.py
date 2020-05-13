import rhinoscriptsyntax as rs
import os

import compas_assembly.datastructures
import compas_rbe.equilibrium

from random import choice
from compas.rpc import Proxy

from compas.geometry import Box
from compas.geometry import Translation
from compas.geometry import scale_vector

from compas_assembly.datastructures import Assembly
from compas_assembly.datastructures import Block

from compas_assembly.rhino import AssemblyArtist


def shift(block):
    scale = choice([+0.01, -0.01, +0.05, -0.05, +0.1, -0.1])
    axis = choice([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    vector = scale_vector(axis, scale)
    T = Translation(vector)
    block.transform(T)


#file_name = ghenv.Component.OnPingDocument().FilePath
filepath = rs.DocumentPath()
print(filepath)
#FILE_IN = os.path.join(filepath,"yolo.json")
#FILE_OUT = os.path.join(filepath,"yolo_done.json")
FILE_IN = os.path.join(filepath,"cantilever0.json")
FILE_OUT = os.path.join(filepath,"cantilever0_done.json")

print(FILE_OUT)

"""
# ==============================================================================
# Parameters
# ==============================================================================

# number of blocks

N = 10

# block dimensions

W = 8.0
H = 0.5
D = 1.0

# ==============================================================================
# Assembly
# ==============================================================================

assembly = Assembly()

# default block

box = Box.from_width_height_depth(W, H, D)
brick = Block.from_shape(box)

# make all blocks
# place each block on top of previous
# shift block randomly in XY plane

for i in range(N):
    block = brick.copy()
    block.transform(Translation([0, 0, 0.5 * H + i * H]))
    shift(block)
    assembly.add_block(block)

# mark the bottom block as support

print(assembly.blocks)
assembly.node_attribute(0, 'is_support', True)
assembly.to_json(FILE_OUT)

# ==============================================================================
# Proxy zone
# ==============================================================================
"""
assembly = Assembly.from_json(FILE_IN)

data = {}
assembly_data = assembly.to_data()
blocks_data = {key: block.to_data() for key, block in assembly.blocks.items()}
data['assembly'] = assembly_data
data['blocks'] = blocks_data


with Proxy('compas_assembly.datastructures') as dt:
    data = dt.assembly_interfaces_xfunc(data)
    
with Proxy('compas_rbe.equilibrium') as eq:
    data = eq.compute_interface_forces_xfunc(data)
     

assembly = Assembly.from_data(data['assembly'])
assembly.blocks = {int(key): Block.from_data(data['blocks'][key]) for key in data['blocks']}


# ==============================================================================
# Visualize
# ==============================================================================

assembly.to_json(FILE_OUT)

artist = AssemblyArtist(assembly, layer="Wall")
artist.clear_layer()
artist.draw_blocks()
artist.draw_interfaces()
artist.draw_resultants(scale=0.05)
#has an issue with division by zero
#artist.color_interfaces(mode=1)
artist.redraw()