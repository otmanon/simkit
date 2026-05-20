
import os
import numpy as np
import blendertoolbox as bt
import bpy

from .vertexScalarToUV_unnormalized import vertexScalarToUV_unnormalized
curr_dir = os.path.dirname(__file__)
def render_skinning_weights(V, F, W,render_dir,
                           texture_path= curr_dir + '/../../data/colormaps/RdBu_11.png',
                           imgRes_x=1200, imgRes_y=1200, numSamples=10, exposure=2,
                           location=[0, 0, 0], rotation=[90, 0, 0],
                           scale=[1.25, 1.25, 1.25], camLocation=[0.5, -1, 0.5], lookAtLocation=[0, 0, 0],
                           lightAngle=[-50, 5, -150], lightStrength=1, shadowSoftness=0.05, shadowThreshold=0.1,
                            lightAmbient=(0.1, 0.1, 0.1, 1), save_blend=False):
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
            bpy.data.objects['numpy mesh object'].select_set(True)
            bpy.ops.object.shade_smooth()
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
            if save_blend:
                bpy.ops.wm.save_mainfile(filepath=os.path.join(os.getcwd(), render_dir, 'test.blend'))
            # bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test.blend')
            outputPath = os.path.join(os.getcwd(), render_dir, str(wi).zfill(3) + '.png')

            # save rendering
            bt.renderImage(outputPath, cam)