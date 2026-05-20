import blendertoolbox as bt
import bpy
import os
import numpy as np

from .setMeshScalars import setMeshScalars

def render_anim_modes( X, F, B, result_dir,
                     period=30, a=1,
                     mesh_color= [201/255, 148/255, 199/255, 255/255],
                     imgRes_x=500, imgRes_y=500, numSamples=5, exposure=2,
                     location=[0, 0, 0],
                     rotation=[90, 0, 0],
                     scale=[1, 1, 1], camLocation=[0.75, -0.75, 0.75], lookAtLocation=[0, 0, 0],
                     lightAngle=[-50, 5, -150], lightStrength=1, shadowSoftness=0.05,
                     shadowThreshold=0.1,
                     lightAmbient=(0.01, 0.01, 0.01, 1), l=None, frames=None):

        dim = X.shape[1]
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

        if frames is None:
            frames = np.arange(period+1)
        lightStrength = lightStrength
        shadowSoftness = shadowSoftness
        shadowThreshold = shadowThreshold

        imgRes_x = imgRes_x
        imgRes_y = imgRes_y

        focalLength = 45 
        
         # (UI: click camera > Object Data > Focal Length)
        init = bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)
        camLocation = camLocation
        cam = bt.setCamera(camLocation, lookAtLocation, focalLength)

        sun = bt.setLight_sun(lightAngle, lightStrength, shadowSoftness)
        bt.setLight_ambient(color=lightAmbient)
        bpy.data.objects['Sun'].select_set(False)
        # bt.invisibleGround(location=(0, 0, -0.05), shadowBrightness=0.0)
        bt.shadowThreshold(shadowThreshold)

        os.makedirs(result_dir, exist_ok=True)

        l = l
        mesh_color = mesh_color
        # bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test.blend')
        # bpy.data.objects['Plane'].select_set(False) # deselect this so it doesn't get deleted


        for mode in range(B.shape[1]):
            mode_dir = result_dir + "/mode_" + str(mode) + "/"
            os.makedirs(mode_dir, exist_ok=True)

            for frame in frames:

                outputPath =mode_dir +"/" + str(frame).zfill(4) + ".png"

                z = a * np.sin(2 * np.pi * frame / period)
                u = B[:, mode] * z
                U = u.reshape((-1, dim))
                mesh = bt.readNumpyMesh(X + U, F, location, rotation, scale)
                bevel_mod = mesh.modifiers.new(name="MY-Bevel2", type='BEVEL')
                bevel_mod.width = 0.0001
                bpy.data.objects['numpy mesh object'].select_set(True)
                bt.subdivision(mesh, level = 1)

                bpy.ops.object.shade_smooth()
                # bpy.ops.object.modifier_add(type='bevel')
                # bpy.context.object.modifiers["bevel"].width = 0.001
                # bpy.context.space_data.context = 'MODIFIER'
                

                meshColor = bt.colorObj(mesh_color, 0.5, 1.3, 1.0, 0.4, 2.0)
                AOStrength = 2
                bt.setMat_balloon(mesh, meshColor, AOStrength)


                # render the image
                bt.renderImage(outputPath, cam)
                # bpy.ops.wm.save_mainfile(filepath=os.getcwd() +  s.result_dir  + '/' + str(frame_num).zfill(4) + '.blend')

                # delte new mesh right after so we don't end up creating them forever
                mesh.select_set(True)
                bpy.ops.object.delete()
        # outputPath = os.getcwd() + "/" + result_dir + "/" + str(frame_num).zfill(4) + ".png"
        # mesh = bt.readNumpyMesh(X, F, location, rotation, scale)

   
        # meshColor = bt.colorObj(mesh_color, 0.5, 1.3, 1.0, 0.4, 2.0)
        # AOStrength = 2
        # bt.setMat_balloon(mesh, meshColor, AOStrength)
    
        # # s.l  = bt.colorObj(s.mesh_color, 0.5, 1.3, 1.0, 0.4, 2.0)

        # bpy.ops.object.shade_smooth()

        # # render the image
        # bt.renderImage(outputPath, cam)
        # # bpy.ops.wm.save_mainfile(filepath=os.getcwd() +  s.result_dir  + '/' + str(frame_num).zfill(4) + '.blend')

        # # delte new mesh right after so we don't end up creating them forever
        # mesh.select_set(True)
        # bpy.ops.object.delete()







    # def __init__(self, name, X, T, B, screenshot_dir=None, eye_pos=[0.4, 0.4, 0.4],
    #              eye_target=[0, 0, 0], scale=[1, 1, 1], numSamples=100, imgRes_x=800, imgRes_y = 800 , color=derekBlue):
    #     self.dim = X.shape[1]
    #     self.name = name
    #     self.X = X

    #     if self.dim == 2:
    #         # concatenate column of zeros
    #         self.X = np.concatenate((self.X, np.zeros((self.X.shape[0], 1))), axis=1)
    #     self.T = T
    #     self.B = B

    #     if T.shape[1] == 4:
    #         F = igl.boundary_facets(T)
    #     elif T.shape[1] == 3:
    #         F = T

    #     self.F = F
    #     self.screenshot_dir = screenshot_dir
    #     self.renderer = RenderAnimModes(self.X, F, self.screenshot_dir, scale=scale, camLocation=eye_pos,
    #                                     lookAtLocation=eye_target,  numSamples=numSamples, imgRes_x=imgRes_x, imgRes_y=imgRes_y, mesh_color=color)

    # pass
    # def render_callback(self, z, z_vel, step, info=None):
    #     a = info['environment_state'].sim_state_hist[
    #         step].a  # hack, need a better way to get each timestep's actuation...
    #     U = (self.B @ a).reshape((-1, self.dim), order='F')

    #     if self.dim == 2:
    #         # concatenate column of zeros
    #         U = np.concatenate((U, np.zeros((U.shape[0], 1))), axis=1)
    #     P = U + self.X

    #     # these are cage displacements
    #     self.renderer.render_frame(P,  step)
    # pass
