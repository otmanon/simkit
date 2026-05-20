from genericpath import isfile
import blendertoolbox as bt
import bpy
import os
import numpy as np

from .vertexScalarToUV_unnormalized import vertexScalarToUV_unnormalized

def render_vectors(X, F, P, D, phi,  output_path, colormap_path,
                      lookAtLocation=[0, 0, 0], camLocation=[0, -5, 0], focal_length=50,
                       imgRes_x=1920, imgRes_y=1080,
                    exposure=2, numSamples=10,
                      location=[0, 0, 0], rotation=[90, 0, 0],
                      scale=[1, 1, 1], shade_smooth=False, save_blend=False,
                      lightAngle=[90, 0, 0], lightStrength=2.0,
                      shadow_softness=0.05, shadow_threshold=0.1,
                      lightAngle2=[45, 45, 45], lightStrength2=1.0,
                      light_ambient=[0.1, 0.1, 0.1, 1.], v_min_max=None,
                      camRotation=None, arrowColor=bt.derekBlue, arrowThickness=0.01, arrowLength=4.0):



    dim  =  X.shape[1]
    phi = phi.reshape(X.shape[0], -1)
    cwd = os.getcwd()
    dim = X.shape[1]

    X = np.append(X, np.zeros((X.shape[0], 3 - dim)), axis=1)
    if isinstance(location, list):
        location = tuple(location)
    if isinstance(camRotation, list):
        camRotation = tuple(camRotation)
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
    if isinstance(light_ambient, list):
        light_ambient = tuple(light_ambient)
    if isinstance(lightAngle2, list):
        lightAngle2 = tuple(lightAngle2)
    if isinstance(arrowColor, list):
        arrowColor = tuple(arrowColor)
        
    bt.blenderInit(imgRes_x, imgRes_y, numSamples=numSamples, exposure=exposure)
    mesh=  bt.readNumpyMesh( X, F, location=location, rotation_euler=rotation, scale=scale) 
    arrow_mesh = bt.createVectorFieldMesh(  P, D, arrowThickness, arrowLength,
                                            location=location, rotation=rotation, scale=scale)
    arrow_color = bt.colorObj(arrowColor, 0.5, 1.0, 1.0, 0.0, 2.0)
    bt.setMat_plastic(arrow_mesh, arrow_color)
    if camRotation is not None:
        x = camRotation[0] * 1.0 / 180.0 * np.pi 
        y = camRotation[1] * 1.0 / 180.0 * np.pi 
        z = camRotation[2] * 1.0 / 180.0 * np.pi 
        bpy.ops.object.camera_add(location = camLocation, rotation=[x, y, z]) # name 'Camera'
        cam = bpy.context.object
        cam.data.lens = focal_length
    else:
        cam = bt.setCamera(camLocation, lookAtLocation, focal_length)
        
    if shade_smooth:
        bpy.context.view_layer.objects.active = mesh
        mesh.select_set(True)
        bpy.ops.object.shade_smooth() 
    

    
    # if output_path is a file, and only one phi, then fine
    is_file = True
    if os.path.isfile(output_path) and phi.shape[1] == 1:
        is_file = True
    elif os.path.isfile(output_path) and phi.shape[1] > 1:
        is_file = False
        output_path = output_path.split('.')[0] + "/"
        os.path.makedirs(output_path, exist_ok=True)
    else:
        is_file = False
    
        # output_path = output_dir + "/" + str(0).zfill(4) + ".png"
    # if output path is a file, and phi has many columns, then convert path to a directory
    
    # if output path is a directory, then save each phi as a s
    
    # should be pointed flat along the y direction
    sun = bt.setLight_sun(lightAngle, lightStrength, shadow_softness)
    
    
    sun2 = bt.setLight_sun(lightAngle2, lightStrength2, shadow_softness)
    
    ## set ambient light
    bt.setLight_ambient(color=light_ambient)
    ## set gray shadow to completely white with a threshold
    bt.shadowThreshold(alphaThreshold=0.02, interpolationMode='CARDINAL')

    for i in range(0, phi.shape[1]): 
        
        if v_min_max is not None:
            vmax = v_min_max[1]
            vmin = v_min_max[0]
        else:
            vmax = np.max(phi[:, i])
            vmin = np.min(phi[:, i]) 
        field = (phi[:, i] - vmin) / (vmax - vmin)
        field = np.clip(field, 1e-2, 1-1e-2)
        mesh = vertexScalarToUV_unnormalized(mesh, field)#, name="phi"+str(i))
        useless = (0, 0, 0, 1)
        meshColor = bt.colorObj(useless, 0.5, 1.0, 1.0, 0.0, 0.0)
        bt.setMat_texture(mesh, colormap_path, meshColor)
        
        
        ## set arrow material
      

        outputPath = os.path.join(cwd, output_path)
        if not is_file:
            outputPath = os.path.join(outputPath, str(i).zfill(4) + ".png")
            
        if save_blend:
            bpy.ops.wm.save_mainfile(filepath=output_path + '.blend')
            
        # save rendering
        bt.renderImage(outputPath, cam)