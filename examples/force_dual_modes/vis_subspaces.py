
import re
import igl
import os
import numpy as np
from PIL import Image
import scipy as sp
import igl.triangle

from simkit import common_selections
from simkit.average_onto_simplex import average_onto_simplex
from simkit.blender.render_cubature_points import render_cubature
from simkit.blender.render_face_labels import render_face_scalars
from simkit.blender.render_vertex_scalars import render_vertex_scalars
from simkit.blender.render_texture import render_texture
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.eigs import eigs
from simkit.fold_vector_hessian import fold_vector_hessian

from simkit.massmatrix import massmatrix
from simkit.normalize_and_center import normalize_and_center
from simkit.polyscope.view_cubature import view_cubature
from simkit.polyscope.view_scalar_field import view_scalar_field
from simkit.polyscope.view_scalar_fields import view_scalar_fields
from simkit.polyscope.view_texture import view_texture
from simkit.shape_outlines import circle
from simkit.spectral_cubature import spectral_cubature
from simkit.ympr_to_lame import ympr_to_lame

dir = os.path.dirname(os.path.realpath(__file__))

colormap_dir = dir + "/../../../data/colormaps/"

character_name = "batsy"

result_dir = dir + "/results/" + character_name + "_blender/"
os.makedirs(result_dir, exist_ok=True)

# for 2D
data_dir = dir + "/../../../data/2d/" + character_name + "/"

mesh_path = data_dir + "/" + character_name + ".obj"
texture_path = data_dir + "/" + character_name + ".png"
uv_path = data_dir + "/" + character_name + "_uv.npy"
distribution_path = data_dir + "/" + character_name + "_distribution.npy"
[X, _, _, T, _, _] = igl.read_obj(mesh_path)
dim = 2
X = X[:, :dim]
X = normalize_and_center(X)

variance = np.load(distribution_path)[:, 0].flatten() +1e-8
pinned = common_selections.center_indices(X, 0.15)[1]
mu, lam = ympr_to_lame(ym=1e6, pr=0.45)
M = massmatrix(X=X, T=T)
M_sqrt = sp.sparse.diags(np.sqrt(M.diagonal()))
H = sk.energies.linear_elasticity_hessian(X=X, T=T, mu=mu, lam=lam)
H_pin = dirichlet_penalty(pinned, X[pinned], X.shape[0], 1e8)[0]
H = H + H_pin
sigma_F = M_sqrt @ sp.sparse.diags(variance.flatten()) @ M_sqrt
sigma_F = sp.sparse.kron(sigma_F, sp.sparse.eye(dim))

k = 50
m = 10
W, D, B = force_dual_skinning_eigenmodes_diagonal(X, H, sigma_F, m, M=M)
[cI, cW, labels] = spectral_cubature(X, T, W, k, return_labels=True)


look_at = [0, 0, 0.]
eye_pos = [0, -5, 0]



render_vertex_scalars(X, T, variance, 
                  result_dir + "/variance.png", 
                  colormap_dir + "/Purples_11.png",
                  lookAtLocation=look_at, camLocation=eye_pos,  
                  imgRes_x=1920, imgRes_y=1080,  exposure=2, normalize_vmax=False)



# render_vertex_scalars(X, T, W,  result_dir + "/fd_se_modes/",
#                     colormap_dir + "/RdBu_11.png", 
#                     look_at=look_at, eye_pos=eye_pos,
#                     img_res_x=1920, img_res_y=1080, exposure=2, save_blend=False, normalize_vmax=True)


# cP = average_onto_simplex(X, T[cI])
# render_cubature(cP, result_dir + "/cubature.png",cW = cW,  
#               look_at=look_at,  eye_pos=eye_pos, 
#               radius_min=0.01, radius_max=0.1, color=[0.7, 0.1, 0.1, 1.0], 
#               img_res_x=1920, img_res_y=1080,  exposure=2, edge_width=0.005)


# render_face_scalars(X, T, labels, 
#               result_dir + "/cubature_labels.png",
#               colormap_dir + "/Paired_11.png",  look_at=look_at,  eye_pos=eye_pos, 
#               img_res_x=1920*2, img_res_y=1080*2, exposure=2,)

# render_texture(X, T, uv_path, texture_path, result_dir + "/texture.png", look_at=look_at, eye_pos=eye_pos,
#                )
# view_texture(X, T, uv_path, texture_path, look_at=look_at, eye_pos=eye_pos, 
#              path=result_dir + "/texture.png")


