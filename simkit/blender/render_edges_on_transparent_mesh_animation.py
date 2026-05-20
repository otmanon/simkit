import blendertoolbox as bt
import bpy
import os
import numpy as np
from pathlib import Path

from .drawEdgeSubset_colored import drawEdgeSubset_colored
from .readNumpyPoints import readNumpyPoints
from ..filesystem.mp4_to_gif import mp4_to_gif
from ..filesystem.video_from_image_dir import video_from_image_dir
from .setMeshScalars import setMeshScalars

def render_edges_on_transparent_mesh_animation( X, F, E,path, P_source=None, P_target=None,
                     mesh_color= [255/255, 255/255, 255/255, 255/255],
                     edge_color= [49/255, 130/255, 189/255, 255/255],
                     edge_scalars = None,
                     edge_colormap_path = None,
                     imgRes_x=500, imgRes_y=500, numSamples=5, exposure=2, edge_radius=0.01,
                     location=[0, 0, 0],
                     rotation=[90, 0, 0],
                     scale=[1, 1, 1], camLocation=[0.75, -0.75, 0.75], lookAtLocation=[0, 0, 0],
                     lightAngle=[-50, 5, -150], lightStrength=1, shadowSoftness=0.05,
                     lightAngle2=[45, 45, 45], lightStrength2=0.5,
                     shadowThreshold=0.1,
                     lightAmbient=(0.01, 0.01, 0.01, 1), save_blend_file=False, fps=30, shade_smooth=False, alpha=0.2, transmission=0.5,
                     source_color=[100/255, 255/255, 100/255, 255/255], source_radius=0.01, target_radius=0.01,
                     target_color=[255/255, 100/255, 100/255, 255/255], frames=None, **kwargs): 

    dirstem = None
    if path is not None:
        stem = Path(path).stem
        dir = Path(path).parent
        dirstem = os.path.join(dir, stem)
        os.makedirs(dirstem, exist_ok=True)
    
    
    num_frames = X.shape[-1]
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
    if isinstance(lightAngle2, list):
        lightAngle2 = tuple(lightAngle2)
    

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
    # bpy.data.objects['Sun'].select_set(False)
    # sun.select_set(False)
    sun2 = bt.setLight_sun(lightAngle2, lightStrength2, shadowSoftness)
    # sun2.select_set(False)
    bt.setLight_ambient(color=lightAmbient)
    
    # bpy.data.objects['Sun'].select_set(False)
    # bt.invisibleGround(location=(0, 0, -0.05), shadowBrightness=0.0)
    bt.shadowThreshold(shadowThreshold)

    bpy.ops.object.select_all(action='DESELECT')
    mesh_color = mesh_color
    save_yet = False
    # bpy.data.objects['Plane'].select_set(False) # deselect this so it doesn't get deleted

    for frame in range(num_frames):

        if frames is not None:
            outputPath = dirstem +"/" + str(frames[frame]).zfill(4) + ".png"
        else:   
            outputPath = dirstem +"/" + str(frame).zfill(4) + ".png"

        U = X[:, :, frame]
        mesh = bt.readNumpyMesh(U, F, location, rotation, scale)
        # bevel_mod = mesh.modifiers.new(name="MY-Bevel2", type='BEVEL')
        if shade_smooth:
            bpy.ops.object.shade_smooth()
  

        if edge_scalars is not None and edge_colormap_path is not None:
            edge_objects = drawEdgeSubset_colored(mesh, E,edge_radius, edge_scalars[:, frame], edge_colormap_path)
        else:
            bt.drawEdgeSubset(mesh, E,edge_radius, edge_color)
            

        if P_source is not None:
            point_source_mesh, created_source_objs = readNumpyPoints(P_source[:, [frame]].T, location, rotation, scale,  np.array(source_color)[None, :], radius=source_radius)
            
            # ptColor_source = bt.colorObj(source_color, 0.5, 1.5, 1.0, 0.0, 0.0)
            # pt_radius = source_radius 
            # bt.setMat_pointCloudColored(point_source_mesh, ptColor_source, pt_radius)
            
        if P_target is not None:
            # point_target_mesh = bt.readNumpyPoints(P_target[:, [frame]].T, location, rotation, scale)
            
            # point_target_mesh = bt.setPointColors(point_target_mesh, np.array(target_color)[None, :])
            
            point_target_mesh, created_target_objs = readNumpyPoints(P_target[:, [frame]].T, location, rotation, scale,  np.array(target_color)[None, :], radius=target_radius)
            
            
            # ptColor_target = bt.colorObj(target_color, 0.5, 1.5, 1.0, 0.0, 0.0)
            # pt_radius = target_radius 
            # bt.setMat_pointCloudColored(point_target_mesh, ptColor_target, pt_radius)


        # bevel_mod.width = 0.0001
        # bpy.data.objects['numpy mesh object'].select_set(True)
        # bt.subdivision(mesh, level = 1)

        # bpy.ops.object.modifier_add(type='bevel')
        # bpy.context.object.modifiers["bevel"].width = 0.001
        # bpy.context.space_data.context = 'MODIFIER'            

        # if tex_png is not None and tex_uv is not None:
        #     uv_layer = mesh.data.uv_layers.new(name='uv')
            
        #     if uv_type=="per_corner":
        #         for face in mesh.data.polygons:
        #             for loopIdx in face.loop_indices:
        #                 uv_layer.data[loopIdx].uv = (tex_uv[loopIdx][0], tex_uv[loopIdx][1])
        #     # per vertex uv  
        #     elif uv_type=="per_vertex":
        #         for face in mesh.data.polygons:
        #             for vIdx, loopIdx in zip(face.vertices, face.loop_indices):
        #                 uv_layer.data[loopIdx].uv = (tex_uv[vIdx, 0], tex_uv[vIdx, 1])

        #     # mesh = vertexScalarToUV_unnormalized(mesh, uv)#, name="phi"+str(i))
        #     useless = (0, 0, 0, 1)
        #     meshColor = bt.colorObj(useless, 0.5, 1, 1, 0.0, 0.0)
        #     bt.setMat_texture(mesh, tex_png, meshColor)
        # else:
        #     meshColor = bt.colorObj(mesh_color, 0.5, 1.3, 1.0, 0.4, 2.0)
        #     AOStrength = 2
        #     bt.setMat_balloon(mesh, meshColor, AOStrength)
        
        meshColor = bt.colorObj(mesh_color, 0.5, 1.5, 1.0, 0.0, 0.0)
        bt.setMat_transparent(mesh, meshColor, alpha, transmission)
        # import os
        # bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test_colormap.blend')
    
        # render the image
        bt.renderImage(outputPath, cam)
        # bpy.ops.wm.save_mainfile(filepath=os.getcwd() +  s.result_dir  + '/' + str(frame_num).zfill(4) + '.blend')

        if save_blend_file and not save_yet:
            save_yet = True
            
            bpy.ops.wm.save_mainfile(filepath=dirstem +  '/test.blend')
        # delte new mesh right after so we don't end up creating them forever
        mesh.select_set(True)
        for obj in edge_objects:
            obj.select_set(True)
        if P_source is not None:
            point_source_mesh.select_set(True)
            for obj in created_source_objs:
                obj.select_set(True)
        if P_target is not None:
            point_target_mesh.select_set(True)
            for obj in created_target_objs:
                obj.select_set(True)
        bpy.ops.object.delete()

    video_from_image_dir(dirstem, path, fps=fps, mogrify=True)
    mp4_to_gif(path, path[:-4] + ".gif", fps=fps)



