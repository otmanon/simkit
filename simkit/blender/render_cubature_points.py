import blendertoolbox as bt
import bpy
import os
import numpy as np

def render_cubature(cP, output_path, cW=None,
                    look_at=[0, 0, 0], color=[0, 0, 0.0,1.0], eye_pos=[0, -5, 0], focal_length=50,
                    normalize_vmax=False, img_res_x=1920, img_res_y=1080,
                    num_samples=10, exposure=2, 
                    location=[0, 0, 0], rotation=[90, 0, 0],
                    scale=[1, 1, 1], shade_smooth=False, save_blend=True,
                    light_angle=[90, 0, 0], light_strength=2.0,
                    shadow_softness=0.05, shadow_threshold=0.1,
                    light_ambient=[0.1, 0.1, 0.1, 1.], radius_min=0.01, radius_max = 0.1, edge_width=0.005):
    # exit()
    cwd = os.getcwd()
    dim = cP.shape[1] 

    
    cwd = os.getcwd()
    
    if cW is None:
        cW = np.ones((cP.shape[0], 1))
        radii = cW * radius_min
    else:
        radii = (cW - np.min(cW))/(np.max(cW) - np.min(cW)) * (radius_max - radius_min) + radius_min
      
    if dim == 2:
        cP = np.append(cP, np.zeros((cP.shape[0], 1)), axis=1)
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
    if isinstance(color, list):
        color = tuple(color)
    
    # render cubature points
    bt.blenderInit(img_res_x, img_res_y, num_samples, exposure)
   

    outputPath = os.path.join(cwd, output_path)
    
    spheres = np.empty((cP.shape[0]), dtype=object)
    mat = bpy.data.materials.new(name="Material2")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = color
    
    mat2 = bpy.data.materials.new(name="black")
    mat2.use_nodes = True
    mat2.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
    # mesh2 = bt.readOBJ( "./cache/cubature_point_mesh.obj", location, rotation, scale);
    
    parent = bpy.data.objects.new("GroupParent", None)
    bpy.context.collection.objects.link(parent)
    
    for i in range(cP.shape[0]):
        if dim == 3:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=radii[i], location=cP[i], 
            enter_editmode=False, align='WORLD')
        elif dim == 2:
            bpy.ops.mesh.primitive_circle_add(radius=radii[i],
                                              enter_editmode=False, 
                                              align='WORLD', 
                                              location=cP[i], vertices=128)
            bpy.ops.object.shade_smooth()
            obj = bpy.context.object
            obj.data.materials.append(mat)  # Assign the material
            obj.parent = parent
            bpy.ops.object.editmode_toggle()
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.edge_face_add()            
            bpy.ops.object.editmode_toggle()
            
            
                        # Create a new circle
            bpy.ops.mesh.primitive_circle_add(vertices=128, radius=radii[i], location=cP[i])
            obj = bpy.context.object
            # Switch to Edit Mode
            bpy.ops.object.mode_set(mode='EDIT')

            # Select all vertices
            bpy.ops.mesh.select_all(action='SELECT')

            # Extrude edges without moving
            bpy.ops.mesh.extrude_edges_move()

            # Shrink/Fatten (moves edges along their normals by a fixed distance)
            thickness = edge_width # Set absolute thickness
            bpy.ops.transform.shrink_fatten(value=thickness)
            # Switch back to Object Mode
            bpy.ops.object.mode_set(mode='OBJECT')
            obj.data.materials.append(mat2) 
            obj.parent = parent
        
        
    
    parent.rotation_euler = rotation
    parent.scale = scale
    parent.location = location

    
    cam = bt.setCamera(eye_pos, look_at, focal_length)
    bt.setLight_ambient(color=(0.1, 0.1, 0.1, 1))
    sun = bt.setLight_sun(light_angle, light_strength, shadow_softness)
    
    if save_blend:
        bpy.ops.wm.save_mainfile(filepath=output_path + '.blend')
    bt.renderImage(outputPath, cam)

    #render stiffness
    # bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)
    # output_path = "./stiffness.png"
    # outputPath = os.path.join(cwd, output_path)
    # mesh = bt.readOBJ("./cache/rest.obj", location, rotation, scale);
    # bevel_mod = mesh.modifiers.new(name="MY-Bevel2", type='BEVEL')
    # bevel_mod.width = 0.01
    # color_type = 'face'
    # color_map = colormaps[0];
    # mesh = bt.setMeshScalars(mesh, mu, color_map, color_type)
    # meshVColor = bt.colorObj([], 0.5, 1.0, 1.0, 0.0, 0.5)
    # bpy.ops.object.shade_smooth()
    # alpha =1.0
    # bt.setMat_VColor_transparent(mesh, meshVColor, alpha)
    # bpy.ops.object.shade_smooth()
    # cam = bt.setCamera(camLocation, lookAtLocation, focalLength)
    # sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)
    # bt.renderImage(outputPath, cam)


    # # render labels
    # bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)
    # output_path = "./clusters.png"
    # outputPath = os.path.join(cwd, output_path)
    # mesh = bt.readOBJ("./cache/rest.obj", location, rotation, scale);
    # bevel_mod = mesh.modifiers.new(name="MY-Bevel2", type='BEVEL')
    # bevel_mod.width = 0.01
    # color_type = 'face'
    # color_map = "Pastel1";
    # mesh = bt.setMeshScalars(mesh, l, color_map, color_type)
    # meshVColor = bt.colorObj([], 0.5, 1.5, 1.0, 0.0, 1.0)
    # bpy.ops.object.shade_smooth()
    # alpha =1.0
    # bt.setMat_VColor_transparent(mesh, meshVColor, alpha)
    # bpy.ops.object.shade_smooth()
    # cam = bt.setCamera(camLocation, lookAtLocation, focalLength)
    # sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)
    # bt.setLight_ambient(color=(0.2, 0.2, 0.2, 1))
    # bt.renderImage(outputPath, cam)



    # render transparent labels
    # bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)
    # output_path = "./clusters_transparent.png"
    # outputPath = os.path.join(cwd, output_path)
    # mesh = bt.readOBJ("./cache/rest.obj", location, rotation, scale);
    # bevel_mod = mesh.modifiers.new(name="MY-Bevel2", type='BEVEL')
    # bevel_mod.width = 0.01
    # color_type = 'face'
    # color_map = "Pastel1";
    # mesh = bt.setMeshScalars(mesh, l, color_map, color_type)
    # meshVColor = bt.colorObj([], 0.5, 1.5, 1.0, 0.0, 5.0)
    # bpy.ops.object.shade_smooth()
    # alpha =0.2
    # bt.setMat_VColor_transparent(mesh, meshVColor, alpha)
    # bpy.ops.object.shade_smooth()
    # cam = bt.setCamera(camLocation, lookAtLocation, focalLength)
    # sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)
    # bt.setLight_ambient(color=(0.2, 0.2, 0.2, 1))
    # bt.renderImage(outputPath, cam)
