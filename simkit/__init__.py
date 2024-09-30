from .backtracking_line_search import backtracking_line_search
from .ympr_to_lame import ympr_to_lame


from .elastic_energy import  elastic_energy_x, elastic_energy_S, elastic_energy_z, ElasticEnergyZPrecomp
from .elastic_gradient import elastic_gradient_dF, elastic_gradient_dx, elastic_gradient_dS, elastic_gradient_dz
from .elastic_hessian import elastic_hessian_d2F, elastic_hessian_d2x, elastic_hessian_d2S, elastic_hessian_d2z

from .quadratic_energy import quadratic_energy
from .quadratic_gradient import quadratic_gradient
from .quadratic_hessian import quadratic_hessian

from .arap_energy import arap_energy_F, arap_energy_S
from .arap_gradient import arap_gradient_dF, arap_gradient_dS
from .arap_hessian import arap_hessian_d2F, arap_hessian_d2S,arap_hessian_d2x, arap_hessian

from .kinetic_energy import kinetic_energy, kinetic_energy_z, KineticEnergyZPrecomp
from .kinetic_gradient import kinetic_gradient, kinetic_gradient_z
from .kinetic_hessian import kinetic_hessian, kinetic_hessian_z

from .stretch import stretch
from .stretch_gradient import stretch_gradient_dx, stretch_gradient_dF, stretch_gradient_dz

from .deformation_jacobian import  deformation_jacobian
from .selection_matrix import selection_matrix
from .symmetric_stretch_map import symmetric_stretch_map

from .massmatrix import massmatrix
from .dirichlet_penalty import dirichlet_penalty

from .grad import grad
from .volume import volume

from .project_into_subspace import project_into_subspace


from .contact_springs_plane_energy import contact_springs_plane_energy
from .contact_springs_plane_gradient import contact_springs_plane_gradient 
from .contact_springs_plane_hessian import contact_springs_plane_hessian

from. contact_springs_sphere_energy import contact_springs_sphere_energy
from .contact_springs_sphere_gradient import contact_springs_sphere_gradient
from .contact_springs_sphere_hessian import contact_springs_sphere_hessian