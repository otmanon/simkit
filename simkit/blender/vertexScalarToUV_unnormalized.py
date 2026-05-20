# Copyright 2020 Hsueh-Ti Derek Liu
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import bpy
import numpy as np

def vertexScalarToUV_unnormalized(mesh_obj, vertex_scalars, name="funcUV"):
    """
    This function takes a vertex scalar data and set to vertex UV (useful for render isoline)

    Inputs
    mesh_obj: bpy.object of the mesh
    C: |V| numpy array of vertex scalars

    Outputs
    mesh_obj
    """
    mesh = mesh_obj.data
    nV = len(mesh.vertices)
    nC = len(vertex_scalars.flatten())

    # guess the type of colors
    if nC != nV:
        raise ValueError('Error in "vertexScalarToUV": input color format must be eithe |V| array of vertex colors')

    if name in mesh.uv_layers:
        uv_layer = mesh.uv_layers[name]
    else:
        uv_layer = mesh.uv_layers.new(name=name)

    C = np.copy(vertex_scalars.flatten())  # we don't do any normalization here. we assume C has already been preprocessed to lie between 0 and 1

    for face in mesh.polygons:
        for vIdx, loopIdx in zip(face.vertices, face.loop_indices):
            uv_layer.data[loopIdx].uv = (C[vIdx], 0)
    
    
    mesh_obj.data.uv_layers[name].active_render = True

    mesh.uv_layers.active = uv_layer
    mesh.update()
    mesh_obj.data.update()

    return mesh_obj



