
import os


from simkit.sims.elastic.ElasticROMPDSim import ElasticROMPDSimParams
from simkit.solvers.BlockCoordSolver import BlockCoordSolverParams
from simkit.subspaces.SpectralLocalizedSkinningEigenmodes import SpectralLocalizedSkinningEigenmodesParams
from simkit.worlds.MeshGrab import MeshGrabWorldParams, MeshGrab
from simkit.subspaces.ModalAnalysis import ModalAnalysisParams
from simkit.subspaces.SkinningEigenmodes import SkinningEigenmodesParams
from simkit.subspaces.RandomImpulseVibes import RandomImpulseVibesParams
class ElephantConfigs():
    
  def __init__(self):
    self.name = "elephant"

    self.mesh_file  =  "/data/3d/pegasus/pegasus.mesh"


dir = os.path.dirname(os.path.realpath(__file__))

c = ElephantConfigs()

skinning_eigenmodes_cache = dir + "./cache/" + c.name + "/skinning_eigenmodes/"
se_subspace_params = SkinningEigenmodesParams( m=20, c=100, k=100, cache_dir = skinning_eigenmodes_cache, read_cache=True)
names = [  "skinning_eigenmodes"]
subspace_params_list = [  se_subspace_params]

spectral_localized_cache = dir + "./cache/" + c.name + "/spectral_localized/"
localized_subspace_params = SpectralLocalizedSkinningEigenmodesParams(d = 20, m=60, c=100, k=100, cache_dir=spectral_localized_cache, read_cache=True, sparse=True)

modal_analysis_cache = dir + "./cache/" + c.name + "/modal_analysis/"
modal_analysis_subspace_params = ModalAnalysisParams( m=20*12, c=100, k=100, cache_dir=modal_analysis_cache, read_cache=True)

random_impulse_vibes_cache = dir + "./cache/" + c.name + "/random_impulse_vibes/"
random_impulse_vibes_subspace_params = RandomImpulseVibesParams( m=20, c=100, k=100, cache_dir=random_impulse_vibes_cache, read_cache=False, sparse=True, h=6e-1, ord=1)

names = ["random", "modal_analysis", "spectral_localized", "skinning_eigenmodes"]
subspace_params_list = [random_impulse_vibes_subspace_params, modal_analysis_subspace_params, localized_subspace_params,  se_subspace_params]

solver_params = BlockCoordSolverParams(max_iter=20, tolerance=1e-16)
world_params_list = []

pre = dir + "/../../../"

for i, subspace_params in enumerate(subspace_params_list):

    sim_params = ElasticROMPDSimParams(ym =1e10, h=1e-2, rho=1e3, solver_params=solver_params, cache_dir=subspace_params.cache_dir + "./sim/", read_cache=False)

    tex_obj = None
    tex_png = None

    world_params = MeshGrabWorldParams(pre +c.mesh_file, subspace_params,sim_params, k_picked=1e9, k_pin=1e7,
                                         sphere_radius=0.3, 
                                         speed=0.1, substeps = 1,
                                           tex_obj=tex_obj, tex_png=tex_png, pinning_method="center")   
    world_params.name = names[i]
    world_params_list.append(world_params)

world = MeshGrab( world_params_list)

from simkit.polyscope.view_scalar_modes import view_scalar_modes

view_scalar_modes(world.X, world.T, world.sub.W.toarray(), eye_pos=[2, 2, -2], eye_target=[0, 0, 0], dir=os.path.dirname(__file__) + "/unicorn_modes_h_6e-1/")
world.play()