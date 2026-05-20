import blendertoolbox as bt
import bpy
import os
import numpy as np

from .faceScalarToUV_unnormalized import faceScalarToUV_unnormalized

def render_face_scalars(X, T, labels, output_path, colormap_path,
                  lookAtLocation=[0, 0, 0], camLocation=[0, -5, 0], focal_length=50,
                  normalize_vmax=False, img_res_x=1920, img_res_y=1080,
                  num_samples=10, exposure=2, 
                  location=[0, 0, 0], rotation=[90, 0, 0],
                  scale=[1, 1, 1], shade_smooth=False, save_blend=True,
                  lightAngle=[90, 0, 0], lightStrength=2.0,
                  shadow_softness=0.05, shadow_threshold=0.1,
                    lightAngle2=[0, 0, 0],  lightStrength2=1.0,
                  lightAmbient=[0.1, 0.1, 0.1, 1.], camRotation=None):
    cwd = os.getcwd()
    dim = X.shape[1]

    X = np.append(X, np.zeros((X.shape[0], 3 - dim)), axis=1)
    if isinstance(location, list):
        location = tuple(location)
    if isinstance(rotation, list):
        rotation = tuple(rotation)
    if isinstance(scale, list):
        scale = tuple(scale)
    if isinstance(camLocation, list):
        camLocation = tuple(camLocation)
    if isinstance(camRotation, list):
        camRotation = tuple(camRotation)
    if isinstance(lookAtLocation, list):
        lookAtLocation = tuple(lookAtLocation)
    if isinstance(lightAngle, list):
        lightAngle = tuple(lightAngle)
    if isinstance(lightAmbient, list):
        lightAmbient = tuple(lightAmbient)
    if isinstance(lightAngle2, list):
        lightAngle2 = tuple(lightAngle2)
        
    bt.blenderInit(img_res_x, img_res_y, numSamples=num_samples, exposure=exposure)
    mesh=  bt.readNumpyMesh( X, T, location=location, rotation_euler=rotation, scale=scale) 
    
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
    
    # render labels
    # bt.blenderInit(img_res_x, img_res_y, num_samples, exposure)
    outputPath = os.path.join(cwd, output_path)
    # mesh = bt.readNumpyMesh(X, T, location, rotation, scale)
    phi = labels / labels.max() * 1.0
    phi = np.clip(phi, 1e-2, 1-1e-2)
    mesh = faceScalarToUV_unnormalized(mesh, phi)
    
    
    useless = (0, 0, 0, 1)
    meshColor = bt.colorObj(useless, 0.5, 1, 1, 0.0, 0.0)
    bt.setMat_texture(mesh, colormap_path, meshColor)

    sun = bt.setLight_sun(lightAngle, lightStrength, shadow_softness)
    
    sun2 = bt.setLight_sun(lightAngle2, lightStrength2, shadow_softness)
    bt.setLight_ambient(color=lightAmbient)
    bt.shadowThreshold(alphaThreshold=0.02, interpolationMode='CARDINAL')

    if save_blend:
        bpy.ops.wm.save_mainfile(filepath=output_path + '.blend')
        
    bt.renderImage(outputPath, cam)
    