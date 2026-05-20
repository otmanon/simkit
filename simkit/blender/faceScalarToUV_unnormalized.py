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

def faceScalarToUV_unnormalized(mesh_obj, face_scalars):
    """
    This function takes a vertex scalar data and set to vertex UV (useful for render isoline)

    Inputs
    mesh_obj: bpy.object of the mesh
    C: |V| numpy array of vertex scalars

    Outputs
    mesh_obj
    """
    mesh = mesh_obj.data
    nF = len(mesh.polygons)
    nC = len(face_scalars.flatten())


    C = np.copy(face_scalars.flatten())
    # C /= C.max() # we don't do any normalization here. we assume C has already been preprocessed to lie between 0 and 1
            
    uv_layer = mesh.uv_layers.new(name="funcUV")

    idx = 0
    for fIdx in range(nF):
        for vIdx in mesh.polygons[fIdx].vertices:
            uv_layer.data[idx].uv = (C[fIdx], 0)
            idx += 1
    # idx = 0
    # for fIdx in range(nF):
    #     for vIdx in mesh.polygons[fIdx].vertices:
    #         uv_layerdata[idx].color = (C_RGB[fIdx, 0], C_RGB[fIdx, 1], C_RGB[fIdx, 2], 1.0)
    #         idx += 1
    return mesh_obj



