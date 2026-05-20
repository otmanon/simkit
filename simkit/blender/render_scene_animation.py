import os
import numpy as np
import blendertoolbox as bt
import bpy
from pathlib import Path
from simkit.filesystem.mp4_to_gif import mp4_to_gif
from simkit.filesystem.video_from_image_dir import video_from_image_dir

def render_scene_animation(render_path, imgRes_x, imgRes_y, numSamples, camLocation, lookAtLocation, 
                         lightAngle, lightStrength, lightAngle2, lightStrength2, 
                         animation_kwargs, save_blend_file=False, fps=60, shade_smooth=True,
                         exposure=2, focalLength=45, shadowSoftness=0.05,
                         ambientLightColor=(0.01, 0.01, 0.01, 1), shadowThreshold=0.1,
                         shadowInterpolationMode='CARDINAL', uv_type='per_corner'):
    """
    Render an animation scene with multiple meshes.
    
    Args:
        render_path (str): Path to save the rendered animation
        imgRes_x (int): Image resolution in x direction
        imgRes_y (int): Image resolution in y direction
        numSamples (int): Number of samples for rendering
        camLocation (list): Camera location [x, y, z]
        lookAtLocation (list): Point to look at [x, y, z]
        lightAngle (list): First light angle [x, y, z]
        lightStrength (float): First light strength
        lightAngle2 (list): Second light angle [x, y, z]
        lightStrength2 (float): Second light strength
        animation_kwargs (list): List of dictionaries containing mesh animation parameters:
            Each dict can contain:
            - X: Rest geometry vertices (numpy array)
            - F: Faces (numpy array)
            - U: Animation displacements (numpy array)
            - tex_png: Path to texture image (optional)
            - tex_uv: Texture UV coordinates (optional)
            - translation: Mesh translation [x, y, z] (optional)
            - scale: Mesh scale [x, y, z] (optional)
            - rotation: Mesh rotation [x, y, z] in radians (optional)
            - uv_type: Type of UV mapping ('per_corner' or 'per_vertex') (optional)
        save_blend_file (bool): Whether to save the Blender file
        fps (int): Frames per second
        shade_smooth (bool): Whether to use smooth shading
        exposure (float): Scene exposure value
        focalLength (float): Camera focal length
        shadowSoftness (float): Shadow softness value
        ambientLightColor (tuple): Ambient light color (r,g,b,a)
        shadowThreshold (float): Shadow threshold value
        shadowInterpolationMode (str): Shadow interpolation mode
    """
    # Create output directory
    dirstem = None
    if render_path is not None:
        stem = Path(render_path).stem
        dir = Path(render_path).parent
        dirstem = os.path.join(dir, stem)
        os.makedirs(dirstem, exist_ok=True)

    # Convert parameters to tuples if they are lists
    if isinstance(camLocation, list):
        camLocation = tuple(camLocation)
    if isinstance(lookAtLocation, list):
        lookAtLocation = tuple(lookAtLocation)
    if isinstance(lightAngle, list):
        lightAngle = tuple(lightAngle)
    if isinstance(lightAngle2, list):
        lightAngle2 = tuple(lightAngle2)

    # Initialize Blender scene
    bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure=exposure)
    
    # Set up camera
    cam = bt.setCamera(camLocation, lookAtLocation, focalLength)
    
    # Set up lights
    sun = bt.setLight_sun(lightAngle, lightStrength, shadowSoftness)
    sun2 = bt.setLight_sun(lightAngle2, lightStrength2, shadowSoftness)
    bt.setLight_ambient(color=ambientLightColor)
    bt.shadowThreshold(alphaThreshold=shadowThreshold, interpolationMode=shadowInterpolationMode)

    bpy.ops.object.select_all(action='DESELECT')
    # Get number of frames from first mesh's animation
    if len(animation_kwargs) > 0:
        num_frames = animation_kwargs[0]['U'].shape[-1]
    else:
        raise ValueError("No meshes provided in animation_kwargs")

    # Process each frame
    for frame in range(num_frames):
        outputPath = os.path.join(dirstem, str(frame).zfill(4) + ".png")
        
        # Process each mesh
        for mesh_kwargs in animation_kwargs:
            X = mesh_kwargs.get('X')
            F = mesh_kwargs.get('F')
            U = mesh_kwargs.get('U')
            
            if X is None or F is None or U is None:
                raise ValueError("Mesh missing required parameters X, F, or U")
            
            # Get current frame's vertices
            vertices = X + U[:, :, frame]
            
            # Get mesh transformations
            location = tuple(mesh_kwargs.get('translation', [0, 0, 0]))
            rotation = tuple(mesh_kwargs.get('rotation', [90, 0, 0]))
            scale = tuple(mesh_kwargs.get('scale', [1, 1, 1]))
            
            # Create mesh in Blender
            mesh = bt.readNumpyMesh(vertices, F, location=location, rotation_euler=rotation, scale=scale)
            
            if shade_smooth:
                bpy.context.view_layer.objects.active = mesh
                mesh.select_set(True)
                bpy.ops.object.shade_smooth() 
    
            
            # Apply texture if provided
            if 'tex_png' in mesh_kwargs and 'tex_uv' in mesh_kwargs:
                uv_layer = mesh.data.uv_layers.new(name='uv')
                tex_uv = mesh_kwargs['tex_uv']
                if uv_type=="per_corner":
                    for face in mesh.data.polygons:
                        for loopIdx in face.loop_indices:
                            uv_layer.data[loopIdx].uv = (tex_uv[loopIdx][0], tex_uv[loopIdx][1])
                # per vertex uv  
                elif uv_type=="per_vertex":
                    for face in mesh.data.polygons:
                        for vIdx, loopIdx in zip(face.vertices, face.loop_indices):
                            uv_layer.data[loopIdx].uv = (tex_uv[vIdx, 0], tex_uv[vIdx, 1])
                # Set material with texture
                useless = (0, 0, 0, 1)
                meshColor = bt.colorObj(useless, 0.5, 1, 1, 0.0, 0.0)
                bt.setMat_texture(mesh, mesh_kwargs['tex_png'], meshColor)
            else:
                # Set default material
                meshColor = bt.colorObj([201/255, 148/255, 199/255, 255/255], 0.5, 1.3, 1.0, 0.4, 2.0)
                AOStrength = 2
                bt.setMat_balloon(mesh, meshColor, AOStrength)
            
            mesh.select_set(True)
        
        # Render the frame
        bt.renderImage(outputPath, cam)
        
        # Save Blender file if requested (only on first frame)
        if save_blend_file:
            bpy.ops.wm.save_mainfile(filepath=os.path.join(dirstem, 'test.blend'))

        # Delete all meshes to prepare for next frame
        # bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        
    # Create video from rendered frames
    video_from_image_dir(dirstem, render_path, fps=fps, mogrify=True)
    mp4_to_gif(render_path, render_path[:-4] + ".gif", fps=fps) 