import sys
 # change this to your path to “path/to/BlenderToolbox/
import blendertoolbox as bt
import os, bpy, bmesh
import numpy as np
from os import listdir

from .vertexScalarToUV_unnormalized import vertexScalarToUV_unnormalized
def SkinningWeightsViewerBlender( V, F, W,render_dir,
                           texture_path='RdBu_11.png',
                           imgRes_x=1200, imgRes_y=1200, numSamples=10, exposure=2,
                           location=[0, 0, 0], rotation=[90, 0, 0],
                           scale=[1.25, 1.25, 1.25], camLocation=[0.5, -1, 0.5], lookAtLocation=[0, 0, 0],
                           lightAngle=[-50, 5, -150], lightStrength=1, shadowSoftness=0.05, shadowThreshold=0.1,
                            lightAmbient=(0.1, 0.1, 0.1, 1)):
    cwd = os.getcwd()


    if isinstance(location, list):
        location = tuple(location)
    if isinstance(rotation, list):
        rotation = tuple(rotation)
    if isinstance(scale, list):
        scale = tuple(scale)
    if isinstance(camLocation, list):
        camLocation = tuple(camLocation)
    if isinstance(lookAtLocation, list):
        lookAtLocation = tuple(lookAtLocation)
    if isinstance(lightAngle, list):
        lightAngle = tuple(lightAngle)
    if isinstance(lightAmbient, list):
        lightAmbient = tuple(lightAmbient)
    focalLength = 45 # (UI: click camera > Object Data > Focal Length)


    for wi in range(W.shape[1]):

        os.makedirs(render_dir, exist_ok=True)
        bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)

        # mesh = bt.readOBJ(obj_file, location, rotation, scale);
        mesh = bt.readNumpyMesh(V, F, location, rotation, scale)
        Cdata = W[:, wi];
        # min
        print("min", Cdata.min())
        # max
        print("max", Cdata.max())
        maxim = np.abs(Cdata).max()
        Cdata = (Cdata) + maxim;
        Cdata = Cdata / (2.0 * maxim);
        Cdata += 1e-3
        Cdata = np.minimum(Cdata, 0.99)

        vertex_scalars = Cdata  # vertex color list
        mesh = vertexScalarToUV_unnormalized(mesh, vertex_scalars)
        # mesh = bt.setMeshScalars(mesh, vertex_scalars, color_map, color_type)

        useless = (0, 0, 0, 1)
        meshColor = bt.colorObj(useless, 0.5, 1, 1, 0.0, 0.0)
        bt.setMat_texture(mesh, texture_path, meshColor)
        # bt.setMat_VColor(mesh, meshColor)
        bpy.ops.object.shade_smooth()
        cam = bt.setCamera(camLocation, lookAtLocation, focalLength)

        ## set light
        sun = bt.setLight_sun(lightAngle, lightStrength, shadowSoftness)

        ## set ambient light
        bt.setLight_ambient(color=lightAmbient)
        ## set gray shadow to completely white with a threshold
        bt.shadowThreshold(alphaThreshold=0.02, interpolationMode='CARDINAL')

        ## save blender file so that you can adjust parameters in the UI
        # bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test.blend')
        outputPath = os.path.join(cwd, render_dir, str(wi).zfill(3) + '.png')

        # save rendering
        bt.renderImage(outputPath, cam)







#
#
#
# Ws = np.load(weight_path)
#
# for i in range(0, Ws.shape[1]):
#   bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)
#
#   # mesh = bt.readNumpyMesh(V,F,location,rotation,scale)
#   mesh = bt.readOBJ(obj_path, location, rotation, scale)
#
#   #bt smooth
#   bt.subdivision(mesh, level = 1)
#   bpy.ops.object.shade_smooth()
#   outputPath = os.path.join(cwd, './skinning_weight_' + str(i) + '.png')
#
#   Cdata = Ws[:, i];
#   #min
#   print("min", Cdata.min())
#   #max
#   print("max", Cdata.max())
#   #Cdata = (Ws - Ws.mean());
#   maxim = np.abs(Cdata).max()
#   #Cdata = Cdata/(maxim);
#   Cdata = (Cdata) +  maxim;
#   Cdata = Cdata / (2.0*maxim);
#   Cdata += 1e-3
#   Cdata  = np.minimum(Cdata, 0.99)
#   #min
#   print("min", Cdata.min())
#
#   #max
#   print("max", Cdata.max())
#
#
#   vertex_scalars = Cdata  # vertex color list
#   color_type = 'vertex'
#   color_map = 'default'
#   mesh = bt.vertexScalarToUV_unnormalized(mesh, vertex_scalars)
#
#  # mesh = bt.setMeshScalars(mesh, vertex_scalars, color_map, color_type)
#
#   useless = (0,0,0,1)
#   meshColor = bt.colorObj(useless, 0.5,1, 1, 0.0, 0.0)
#   bt.setMat_texture(mesh, texturePath, meshColor)
#  # bt.setMat_VColor(mesh, meshColor)
#
#   cam = bt.setCamera(camLocation, lookAtLocation, focalLength)
#
#   ## set light
#   sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)
#
#   ## set ambient light
#   bt.setLight_ambient(color=(0.1,0.1,0.1,1))
#
#   ## set gray shadow to completely white with a threshold
#   bt.shadowThreshold(alphaThreshold = 0.02, interpolationMode = 'CARDINAL')
#
#   ## save blender file so that you can adjust parameters in the UI
#   # bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test.blend')
#
#   # save rendering
#   bt.renderImage(outputPath, cam)
# c += 1
#
#
