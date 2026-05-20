import blendertoolbox as bt
import bpy

def render_mesh(X, F, path, color=bt.derekBlue, bI=None, pc_color=bt.cb_purple,
                     radius=0.1,
                     mesh_color= [201/255, 148/255, 199/255, 255/255],
                     imgRes_x=500, imgRes_y=500, numSamples=5, exposure=2,
                     location=[0, 0, 0],
                     rotation=[90, 0, 0],
                     scale=[1, 1, 1], camLocation=[0.75, -0.75, 0.75], lookAtLocation=[0, 0, 0],
                     lightAngle=[-50, 5, -150], lightStrength=1, shadowSoftness=0.05,
                     shadowThreshold=0.1,
                     lightAmbient=(0.01, 0.01, 0.01, 1)):
    meshColor = bt.colorObj(mesh_color, 0.5, 1.3, 1.0, 0.4, 2.0)
    AOStrength = 2
    dim = X.shape[1]
    bt.blenderInit(imgRes_x, imgRes_y, numSamples=numSamples, exposure=exposure)
    mesh=  bt.readNumpyMesh( X, F, location=location, rotation_euler=rotation, scale=scale)  
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    bpy.ops.object.shade_smooth() 
    meshColor = bt.colorObj(color, 0.5, 1.5, 1.0, 0.0, 2.0)
    bt.setMat_balloon(mesh, meshColor, 1)
    bt.setLight_ambient(color=(0.1,0.1,0.1,1)) 
 
    if bI is not None:
        points = bt.readNumpyPoints( X[bI, :], location=location, rotation_euler=rotation, scale=scale)
        pointColor = bt.colorObj(pc_color, 0.5, 1.0, 1.0, 0.0, 2.0)
        bt.setMat_pointCloud(points, pointColor, radius)

    ## set light
    lightAngle = lightAngle 
    strength = lightStrength
    shadowSoftness = shadowSoftness
    sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)
    
    # lightAngle2 = (54, -66, 30) 
    # strength2 = 0.6
    # shadowSoftness2 = 0.3
    # sun2 = bt.setLight_sun(lightAngle2, strength2, shadowSoftness2)


    cam= bt.setCamera(camLocation, lookAtLocation, focalLength=35)
    bt.invisibleGround(shadowBrightness=0.9, location=[0, 0, X[:, 1].min()-0.15])
    # bpy.ops.wm.save_mainfile(filepath=result_dir + '/test.blend')
    bt.renderImage(path, camera=cam)