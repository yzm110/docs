import numpy as np
import matplotlib.pyplot as plt
import time

from lane_slam.lane_feature import LaneFeature

def visualize_skeleton_demo(num_iter_pause=0.6):
    # synthetic lane: mainly x->y curve with noise
    x = np.linspace(0.0, 50.0, 200)
    y = 2.0 * np.sin(x / 5.0)
    z = np.zeros_like(x)
    pts = np.vstack((x, y, z)).T + np.random.normal(scale=0.15, size=(len(x), 3))

    lf = LaneFeature(id=0, points=pts, category=0)
    lf.fitting()  # build polyline used by get_next_node

    origin = lf.get_xyzs()
    # init ctrl pts same as get_skeleton: first point
    inital_point = origin[0]
    lf.ctrl_pts.add(inital_point)

    origin_points = origin.copy()
    origin_points_debug = origin.copy()

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 6))

    it = 0
    while True:
        it += 1
        origin_points = lf.get_pts_to_add(origin_points)
        if origin_points.shape[0] == 0:
            print("no more points to add, finished")
            break
        no_assigned = np.arange(origin_points.shape[0]).tolist()

        # choose find_border method like get_skeleton
        if origin_points.shape[0] <= 15:
            inner_border, outer_border = lf.find_border_point(inital_point, origin_points, no_assigned)
        else:
            inner_border, outer_border = lf.find_border_point_kdtree(inital_point, origin_points, no_assigned)

        # decide head or tail and compute next_initial
        if outer_border is None:
            # end-case: extend head or tail and finish
            d_head = np.linalg.norm(lf.ctrl_pts.get_xyz(0) - inner_border)
            d_tail = np.linalg.norm(lf.ctrl_pts.get_xyz(-1) - inner_border)
            if d_head <= d_tail:
                center = lf.ctrl_pts.get_xyz(0)
                if lf.ctrl_pts.size() >= 2:
                    inner_border = lf.get_query(lf.ctrl_pts.get_xyz(1), lf.ctrl_pts.get_xyz(0), lf.polyline)
                next_initial = lf.get_next_node(inner_border, lf.ctrl_pts.get_xyz(0), lf.ctrl_points_chord, lf.polyline)
                lf.ctrl_pts.add(next_initial)
            else:
                center = lf.ctrl_pts.get_xyz(-1)
                if lf.ctrl_pts.size() >= 2:
                    inner_border = lf.get_query(lf.ctrl_pts.get_xyz(-2), lf.ctrl_pts.get_xyz(-1), lf.polyline)
                next_initial = lf.get_next_node(inner_border, lf.ctrl_pts.get_xyz(-1), lf.ctrl_points_chord, lf.polyline)
                lf.ctrl_pts.append(next_initial)

            # plot final step then break
            ax.clear()
            ax.scatter(origin[:,0], origin[:,1], s=6, c='0.7', label='origin points')
            ctrl = np.array(lf.get_ctrl_xyz())
            ax.plot(ctrl[:,0], ctrl[:,1], 'k-o', label='ctrl pts')
            if inner_border is not None:
                ax.scatter(inner_border[0], inner_border[1], c='g', s=90, label='inner_border')
            ax.scatter(next_initial[0], next_initial[1], c='b', s=90, label='next_initial')
            circle = plt.Circle((center[0], center[1]), lf.ctrl_points_chord, color='g', fill=False, linestyle='--', alpha=0.6)
            ax.add_patch(circle)
            ax.set_title(f'iter {it} (final extend)')
            ax.set_aspect('equal', 'box')
            ax.legend()
            plt.draw(); plt.pause(num_iter_pause)
            break

        # normal case: extend towards outer_border
        d_head = np.linalg.norm(lf.ctrl_pts.get_xyz(0) - outer_border)
        d_tail = np.linalg.norm(lf.ctrl_pts.get_xyz(-1) - outer_border)
        if d_head <= d_tail:
            center = lf.ctrl_pts.get_xyz(0)
            if lf.ctrl_pts.size() >= 2:
                outer_border = lf.get_query(lf.ctrl_pts.get_xyz(1), lf.ctrl_pts.get_xyz(0), lf.polyline)
            next_initial = lf.get_next_node(outer_border, lf.ctrl_pts.get_xyz(0), lf.ctrl_points_chord, lf.polyline)
            lf.ctrl_pts.add(next_initial)
        else:
            center = lf.ctrl_pts.get_xyz(-1)
            if lf.ctrl_pts.size() >= 2:
                outer_border = lf.get_query(lf.ctrl_pts.get_xyz(-2), lf.ctrl_pts.get_xyz(-1), lf.polyline)
            next_initial = lf.get_next_node(outer_border, lf.ctrl_pts.get_xyz(-1), lf.ctrl_points_chord, lf.polyline, points_debug=origin_points_debug)
            lf.ctrl_pts.append(next_initial)

        # update inital_point and remaining origin_points like get_skeleton
        inital_point = next_initial
        origin_points = origin_points[no_assigned]

        # plot current iteration
        ax.clear()
        ax.scatter(origin[:,0], origin[:,1], s=6, c='0.7', label='origin points')
        ctrl = np.array(lf.get_ctrl_xyz())
        ax.plot(ctrl[:,0], ctrl[:,1], 'k-o', label='ctrl pts')
        if inner_border is not None:
            ax.scatter(inner_border[0], inner_border[1], c='g', s=80, label='inner_border')
        if outer_border is not None:
            ax.scatter(outer_border[0], outer_border[1], c='r', s=80, label='outer_border')
        ax.scatter(next_initial[0], next_initial[1], c='b', s=80, label='next_initial')
        circle = plt.Circle((center[0], center[1]), lf.ctrl_points_chord, color='g', fill=False, linestyle='--', alpha=0.6)
        ax.add_patch(circle)
        ax.set_title(f'iter {it}')
        ax.set_aspect('equal', 'box')
        ax.legend(loc='upper right')
        plt.draw()
        plt.pause(num_iter_pause)

    plt.ioff()
    # final plot
    fig2, ax2 = plt.subplots(figsize=(9,6))
    ax2.scatter(origin[:,0], origin[:,1], s=6, c='0.7', label='origin points')
    ctrl = np.array(lf.get_ctrl_xyz())
    ax2.plot(ctrl[:,0], ctrl[:,1], 'k-o', label='ctrl pts (final)')
    ax2.set_aspect('equal', 'box')
    ax2.legend()
    plt.show()

if __name__ == "__main__":
    visualize_skeleton_demo()