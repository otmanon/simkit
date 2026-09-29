# Two soft fingers with stiff bones (2D)

`two_finger_2d.py` builds two silicone fingers (E = 300 kPa), each with two
stiff bones (E = 5 GPa), hanging from a pinned top edge. Four SimKit
mass-spring tendons (two per finger, inner side, across the base and middle
joints) are driven by one actuation parameter `a`: `l0 = (1 - 0.3 a) l_rest`.

It writes to `output/`:

* `fingers.png` - mesh, materials, tendons; open and statically closed
* `eigenfunctions.png` - lowest modes of `A = H_elastic + H_tendons + Q_pin`
* `coarse_meshes.png` - fine (2607 v) vs mesh4PDE vs 2D shortest-edge (150 v)
* `two_fingers.mp4` - the closing motion (backward Euler) on the fine mesh
  and in both coarse subspaces `x = X + (P kron I) z`, with fingertip error

Run: `MESH4PDE=~/mesh4pde python two_finger_2d.py` (needs `triangle`,
`imageio-ffmpeg`, and a built mesh4PDE for `lib.coarsen`). mesh4PDE has no 2D
shortest-edge baseline, so a small Python one is included (link condition,
no flips, boundary corners kept) with the same barycentric prolongation.
