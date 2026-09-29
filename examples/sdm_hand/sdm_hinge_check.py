"""Hinge check: how much each joint rotates about a wrong axis in the first 20 modes.

For every flexure the rigid rotation of the block above relative to the block below
(least-squares fit on the stiff vertices) is split into the hinge-axis part and the rest,
with the hinge pins off (k_pin = 0) and at K_PIN and 10x K_PIN.

    python sdm_hinge_check.py
"""
import numpy as np, simkit, scipy.sparse as sps
from sdm_sim import Hand
d=dict(np.load('output/sdm_tets.npz')); h=Hand(d)
x=h.X.reshape(-1); fr=h.free; M=sps.diags(h.m[fr])
def rigid(ids,u):
    P=h.X[ids]; c=P.mean(0); r=P-c
    A=np.zeros((3*len(ids),6))
    for i,v in enumerate(r):
        A[3*i:3*i+3,:3]=np.array([[0,v[2],-v[1]],[-v[2],0,v[0]],[v[1],-v[0],0]]); A[3*i:3*i+3,3:]=np.eye(3)
    s=np.linalg.lstsq(A,u[ids].ravel(),rcond=None)[0]; return s[:3]
joints=[(b,a,(q2-q1)/np.linalg.norm(q2-q1)) for b,a,q1,q2 in h._pin_specs]
blk={n:h.part_vertices(n,surface=False) for b,a,_ in joints for n in (b,a)}
for kp in [0.0,1e7,1e8]:
    A=(h.hessian(x,0.0)-h.H_pin*(1-kp/h.k_pin))[fr][:,fr].tocsc()
    lam,B=simkit.eigs(A,k=20,M=M); o=np.argsort(lam); lam,B=lam[o],B[:,o]
    rows=[]
    for i in range(20):
        u=np.zeros(3*h.n); u[fr]=B[:,i]; u=u.reshape(-1,3)
        num=den=0;wr=0
        for b,a,ax in joints:
            w=rigid(blk[a],u)-rigid(blk[b],u); num+=(w@ax)**2; den+=w@w; wr+=w@w-(w@ax)**2
        rows.append((num/den,np.sqrt(wr),np.sqrt(num)))
    rows=np.array(rows)
    print(f"k_pin={kp:.0e}: freq", np.round(np.sqrt(lam)/2/np.pi).astype(int))
    print("   |wrong-axis rot| rad/(kg^.5 m):", np.round(rows[:,1],2))
    print("   |hinge rot|                  :", np.round(rows[:,2],2))
    print(f"   max wrong {rows[:,1].max():.2f}")
