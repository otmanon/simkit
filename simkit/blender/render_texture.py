import blendertoolbox as bt
import bpy
import os
import numpy as np

from simkit.blender.vertexScalarToUV_unnormalized import vertexScalarToUV_unnormalized



def render_texture(X, F, uv_path, texture_path,  output_path,
                      look_at=[0, 0, 0], eye_pos=[0, -5, 0], focal_length=50,
                       img_res_x=1920, img_res_y=1080,
                      num_samples=10, exposure=2, 
                      location=[0, 0, 0], rotation=[90, 0, 0],
                      scale=[1, 1, 1], shade_smooth=False, save_blend=False,
                      light_angle=[90, 0, 0], light_strength=2.0,
                      shadow_softness=0.05, shadow_threshold=0.1,
                      light_ambient=[0.1, 0.1, 0.1, 1.]):

    cwd = os.getcwd()
    dim = X.shape[1]

    X = np.append(X, np.zeros((X.shape[0], 3 - dim)), axis=1)
    if isinstance(location, list):
        location = tuple(location)
    if isinstance(rotation, list):
        rotation = tuple(rotation)
    if isinstance(scale, list):
        scale = tuple(scale)
    if isinstance(eye_pos, list):
        eye_pos = tuple(eye_pos)
    if isinstance(look_at, list):
        look_at = tuple(look_at)
    if isinstance(light_angle, list):
        light_angle = tuple(light_angle)
    if isinstance(light_ambient, list):
        light_ambient = tuple(light_ambient)
        
    bt.blenderInit(img_res_x, img_res_y, numSamples=num_samples, exposure=exposure)
    mesh=  bt.readNumpyMesh( X, F, location=location, rotation_euler=rotation, scale=scale) 
    cam = bt.setCamera(eye_pos, look_at, focal_length)

    if shade_smooth:
        bpy.context.view_layer.objects.active = mesh
        mesh.select_set(True)
        bpy.ops.object.shade_smooth() 
    
    bt.setLight_ambient(color=light_ambient)
    
    
    
    if dim == 2:
        # should be pointed flat along the y direction
        sun = bt.setLight_sun(light_angle, light_strength, shadow_softness)
        ## set ambient light
        bt.setLight_ambient(color=light_ambient)
        ## set gray shadow to completely white with a threshold
        bt.shadowThreshold(alphaThreshold=0.02, interpolationMode='CARDINAL')
    else:
        #  not implemented
        raise NotImplementedError
    
    uv = np.load(uv_path)
    uv_layer = mesh.data.uv_layers.new(name='uv')
    for face in mesh.data.polygons:
        for vIdx, loopIdx in zip(face.vertices, face.loop_indices):
            uv_layer.data[loopIdx].uv = (uv[vIdx, 0], uv[vIdx, 1])

    # mesh = vertexScalarToUV_unnormalized(mesh, uv)#, name="phi"+str(i))
    useless = (0, 0, 0, 1)
    meshColor = bt.colorObj(useless, 0.5, 1, 1, 0.0, 0.0)
    bt.setMat_texture(mesh, texture_path, meshColor)
    
    outputPath = os.path.join(cwd, output_path)
        
    if save_blend:
        bpy.ops.wm.save_mainfile(filepath=output_path + '.blend')
    
    # save rendering
    bt.renderImage(outputPath, cam)