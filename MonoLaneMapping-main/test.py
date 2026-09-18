# python
import numpy as np
import matplotlib.pyplot as plt

def catmull_rom_chain(P, n_per_seg=50):
    P = np.asarray(P)
    assert P.shape[0] >= 4
    curve = []
    seg_u = []
    for i in range(len(P) - 3):
        p0, p1, p2, p3 = P[i], P[i+1], P[i+2], P[i+3]
        t = np.linspace(0, 1, n_per_seg, endpoint=False)
        t2 = t*t; t3 = t2*t
        # Catmull-Rom formula (uniform)
        pt = 0.5 * ((2*p1) + (-p0 + p2)*t[:,None] +
                    (2*p0 - 5*p1 + 4*p2 - p3)*t2[:,None] +
                    (-p0 + 3*p1 - 3*p2 + p3)*t3[:,None])
        curve.append(pt)
        seg_u.append(t)
    curve = np.vstack(curve)
    # estimated global u in [0,1]
    u_global = np.linspace(0, 1, curve.shape[0])
    return curve, u_global

# demo control points (these correspond to LinkedPoints.items -> node.item)
ctrl_pts = np.array([
    [0.0, 0.0, 0.0],
    [5.0, 1.0, 0.0],
    [10.0, 0.0, 0.0],
    [15.0, -2.0, 0.0],
    [20.0, -1.0, 0.0],
    [25.0, 0.5, 0.0]
])
# produce spline (2D view using x,y)
spline_pts, u = catmull_rom_chain(ctrl_pts, n_per_seg=100)

# an observation point (pt_w from frame.transform_to_world(point_c))
obs = np.array([12.0, 2.5, 0.0])

# find footpoint by dense sampling (prototype of find_footpoint + parameterization)
dists = np.linalg.norm(spline_pts - obs, axis=1)
idx_min = np.argmin(dists)
foot = spline_pts[idx_min]
u_at_foot = u[idx_min]
error = dists[idx_min]

# plot
plt.figure(figsize=(8,5))
plt.plot(spline_pts[:,0], spline_pts[:,1], '-b', label='Catmull-Rom spline (map lane)')
plt.plot(ctrl_pts[:,0], ctrl_pts[:,1], 'ko-', label='control points (ctrl_pts)')
plt.plot(obs[0], obs[1], 'r*', markersize=12, label='observation (pt_w)')
plt.plot(foot[0], foot[1], 'go', markersize=10, label=f'footpoint (u={u_at_foot:.3f})')
# residual vector = factor (measurement - model)
plt.arrow(foot[0], foot[1], obs[0]-foot[0], obs[1]-foot[1], color='magenta', width=0.05, length_includes_head=True, label='residual (factor)')
plt.axis('equal')
plt.legend()
plt.title('Point → Spline → Footpoint → Residual (factor)')
plt.xlabel('x'); plt.ylabel('y')
plt.grid(True)
plt.show()

# printed numbers
print("u (approx):", u_at_foot, "error (distance):", error)