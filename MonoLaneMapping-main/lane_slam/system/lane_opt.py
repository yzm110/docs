#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk

import open3d as o3d
import numpy as np
from functools import partial
import gtsam
from gtsam.symbol_shorthand import L, X
from misc.config import cfg
from misc.lie_utils import inv_se3, so3_to_rotvec, rad2deg, se3_to_euler_xyz, make_noisy_pose2d, rot_to_angle, se3_log
from lane_slam.linked_points import Node
from time import perf_counter
from copy import deepcopy
from misc.curve.catmull_rom import parameterization
from lane_slam.factors import error_catmull_rom, p2tan_catmull_rom, chord_factor, p2tan_factor, \
    chord_factor2, p2tan_factor3, PosePointFactor, PosePointTangentFactor, PoseCurveTangentFactor, skew
from lane_slam.system.lane_tracker import LaneTracker
from misc.curve.catmull_rom import CatmullRomSpline

class LaneOptimizer(LaneTracker):
    def __init__(self, bag_file):
        super(LaneOptimizer, self).__init__(bag_file)

    def map_update(self):
        self.add_keyframe()
        # lanes_id = [it_f.frame_id for it_f in self.sliding_window]
        # print("lanes_id: ", lanes_id)
        if self.add_odo_noise: #false
            self.update_pose()
        self.build_graph()
        self.optimization()
        self.slide_window()

        if cfg.debug_flag:
            lanes_opted = []
            for lane_id in self.lm_in_window:
                lanes_opted.append(deepcopy(self.lanes_in_map[lane_id]))
            self.visualize_optimization(self.lanes_prev, lanes_opted, self.lane_meas, self.pts_cp_valid)

    def add_keyframe(self):
        if self.use_isam:
            self.sliding_window = [self.cur_frame] ## 创建一个包含 self.cur_frame 的列表， 元素是引用
            return
        if len(self.sliding_window) < self.window_size + 1:
            self.sliding_window.append(self.cur_frame)
            return
        else:
            second_latest_frame = self.sliding_window[-2]
            third_latest_frame = self.sliding_window[-3]
            delta_pose = np.dot(third_latest_frame.T_cw, second_latest_frame.T_wc)
            delta_xyz = np.linalg.norm(delta_pose[:3, 3])
            delta_angle = rot_to_angle(delta_pose[:3, :3])
            if delta_xyz > 3 or delta_angle > 10:
                self.margin_old = True
            else:
                self.margin_old = False
            self.sliding_window[-1] = self.cur_frame

# 为滑动窗口内的每个“检测点观测（det point）”生成或挑选一个用于优化的因子（factor），并构造/初始化 GTSAM 的因子图与初值（initial_estimate）。同时做观测下采样、观测过滤、关键点（control-point）符号分配与稳定性检测，以控制图规模和提高鲁棒性
    def build_graph(self):
        t0 = perf_counter()
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.lane_meas = [] #用于debug

        if not cfg.lane_mapping.init_after_opt: #init_after_opt ==false, so go inside
            self.create_new_lane() #将det线提取关键点后 ，加入到地图lanes_in_map中，

        # self.odo_graph()
        # record the landmarks in the sliding window
            # 等价于：
            # all_ids = []
            # for frame in self.sliding_window:                    # 遍历滑动窗口中的每一帧
            #     for lf in frame.get_lane_features():             # 遍历该帧的每条车道线
            #         if lf.id != -1 and lf.id in self.lanes_in_map:  # 条件过滤
            #             all_ids.append(lf.id)                     # 收集ID
        self.lm_in_window = list(set([lf.id for frame in self.sliding_window  #sliding_window里只有当前帧
                                      for lf in frame.get_lane_features()
                                      if lf.id != -1 and lf.id in self.lanes_in_map]))

        # prepare the grid for downsample
        self.lane_grid = {}
        for lm_id in self.lm_in_window:
            self.lanes_in_map[lm_id].ctrl_pts.update_kdtree()
            self.lane_grid[lm_id] = {}
            # use_isam == true, so skip this for loop
            for ctrl_pt in [] if self.use_isam else self.lanes_in_map[lm_id].get_ctrl_nodes():
                self.lane_grid[lm_id][ctrl_pt.id] = []
                for j in range(cfg.lane_mapping.lane_sample_num):
                    self.lane_grid[lm_id][ctrl_pt.id].append([np.inf])

        pts_cp = []
        for frame in self.sliding_window:
            for i, lf in enumerate(frame.get_lane_features()): #遍历det线
                if lf.id == -1 or lf.id not in self.lanes_in_map:
                    continue
                lm_id = lf.id
                lm = self.lanes_in_map[lm_id] #取对应的lm线，和det的区别是提取了关键点
                for j, pt_c in enumerate(lf.get_xyzs()): #enumerate操作，每次取了det线的一行，即一个点
                    pt_w = frame.T_wc[:3, :3].dot(pt_c[:3]) + frame.T_wc[:3, 3]
                    ctrl_pts, u, error = lm.ctrl_pts.find_footpoint(pt_w) #ctrl_pts为对应的lm和pt_w的垂足最近的前后4个点
                    # 大概估计一下误差，如果误差太大，就不要加入因子了
                    # 只有当 ctrl_pts 非 None、error 小于阈值（例如 10.0 或 pose_update 中更严格的阈值）且 u 在 [0,1] 时，才把该观测作为有效观测加入 pts_cp（后续用于构造因子）。
                    if ctrl_pts is None or error > 10.0 or u < 0 or u > 1:
                        continue

                    frame_id = frame.frame_id
                    pt_w = np.concatenate((pt_w, [u])) #将u附加到pt_w后面， pt_w是世界坐标系下的点
                    if self.use_isam:
                        pts_cp.append([pt_w, pt_c, ctrl_pts, lm_id, frame_id, lf.noise[j]])
                    else:
                        interval_id = np.clip(int(u * cfg.lane_mapping.lane_sample_num), 0, cfg.lane_mapping.lane_sample_num - 1)
                        if lf.noise[j] < self.lane_grid[lm_id][ctrl_pts[1].id][interval_id][-1]:
                            self.lane_grid[lm_id][ctrl_pts[1].id][interval_id] = [pt_w, pt_c, ctrl_pts, lm_id, frame_id, lf.noise[j]]

                # if self.visualization: # for visualization only
                self.lanes_in_map[lm_id].update_raw_pts(frame.transform_to_world(lf), frame.frame_id)
                if cfg.debug_flag:
                    self.lane_meas.append(deepcopy(frame.transform_to_world(lf)))

        #如果self.use_isam为True：遍历空列表[] → 不执行循环， so skip this
        for lm_id in [] if self.use_isam else self.lane_grid.keys():
            for ctrl_pt_id in self.lane_grid[lm_id].keys():
                for interval_id in range(cfg.lane_mapping.lane_sample_num):
                    if self.lane_grid[lm_id][ctrl_pt_id][interval_id][-1] < np.inf:
                        pts_cp.append(self.lane_grid[lm_id][ctrl_pt_id][interval_id])


        #这段代码是SLAM优化中构建因子图的关键步骤，主要效果是：为每个车道线观测创建优化因子，并初始化/更新状态变量
        #pts_cp里保存 了每一条det线的：
        # 1.世界坐标系下的点pt_w，2.相机坐标系下的点pt_c，控制点ctrl_pts(4个)，车道线ID lm_id，帧ID frame_id，噪声noise
        key_status_cur = {}
        for pt_w, pt_c, ctrl_pts, lm_id, frame_id, noise in pts_cp:
            # insert initial estimate
            for ctrl_pt in ctrl_pts:
                in_graph = self.set_gtsam_symbol(lm_id, ctrl_pt.id, ctrl_pt)
                key = ctrl_pt.get_key() #key 是一个 gtsam.Key（由 L(idx) 生成），唯一标识图中的一个变量（这里是一个 Point3 类型变量）
                #insert_or_assign 会把 key→value 存入 initial_estimate：如果 key 不存在则插入，存在则覆盖
                #在 ISAM 增量模式下，已存在的 key 的估计由 ISAM 内部维护，重复覆盖会破坏增量状态或丢失历史估计。因此只为新分配的 key 提供初值；已有 key 由 ISAM 保留原估计。
                if self.use_isam:
                    if not in_graph:
                        self.initial_estimate.insert_or_assign(key, gtsam.Point3(ctrl_pt.item))
                else:
                    self.initial_estimate.insert_or_assign(key, gtsam.Point3(ctrl_pt.item))
                #gtsam.Values 是 key→值 的映射容器（初值或优化结果），self.initial_estimate是 gtsam.Values
            u = pt_w[-1]
            gf = self.lane_factor(pt_w, pt_c, u, ctrl_pts, noise, frame_id)
            self.factor_candidates.append([gf, pt_w, pt_c, np.asarray([node.item for node in ctrl_pts]), lm_id, frame_id, noise])
            self.update_key_status(ctrl_pts, key_status_cur)

        added_cp_d = []
        if self.use_isam:
            added_num = 0
            # merge key_status
            for key in key_status_cur.keys():
                if key not in self.key_status:
                    self.key_status[key] = key_status_cur[key]
                else:
                    self.key_status[key][0] += key_status_cur[key][0]
                    self.key_status[key][1] += key_status_cur[key][1]
            remove_idx = []
            for i_gf, (gf, pt_w, pt_c, ctrl_pts, lm_id, frame_id, noise) in enumerate(self.factor_candidates):
                valid = True
                for key in gf.keys():
                    if key not in self.key_in_graph.keys():
                        continue
                    if self.key_status[key][0] < 4:
                        valid = False
                        break
                if valid:
                    self.graph.add(gf)
                    remove_idx.append(i_gf)
                    added_num += 1
                    added_cp_d.append([pt_w, pt_c, ctrl_pts, lm_id, frame_id, noise, self.graph.size() - 1])
            for i in remove_idx[::-1]:
                self.factor_candidates.pop(i)
            # print('added factor num: ', added_num)
            self.add_chordal_factor(key_status_cur)
        else:
            for i_gf, (gf, pt_w, pt_c, ctrl_pts, lm_id, frame_id, noise) in enumerate(self.factor_candidates):
                self.graph.add(gf)
                added_cp_d.append([pt_w, pt_c, ctrl_pts, lm_id, frame_id, noise, self.graph.size() - 1])
            self.factor_candidates = []
            self.add_ctrl_factor(key_status_cur)

        self.graph_build_timer.update(perf_counter() - t0)
        if cfg.debug_flag:
            self.lanes_prev = []
            for lane_id in self.lm_in_window:
                self.lanes_prev.append(deepcopy(self.lanes_in_map[lane_id]))
        self.pts_cp_valid = added_cp_d

    def lane_factor(self, pt_w, pt_c, u, ctrl_pts, noise, frame_id):

        # noise_model = self.get_pt_noise_model(noise, huber=False)
        noise_model = self.get_pt_noise_model(noise, huber=True, huber_thresh=0.5)
        # gf = gtsam.CustomFactor(noise_model,
        #                         [ctrl_pts[0].get_key(), ctrl_pts[1].get_key(),
        #                          ctrl_pts[2].get_key(), ctrl_pts[3].get_key(), X(frame_id)],
        #                         partial(PoseCurveTangentFactor, [pt_c, u]))
        
        #自定义因子， 输入为噪声， key, 残差方程
        gf = gtsam.CustomFactor(noise_model,
                                [ctrl_pts[0].get_key(), ctrl_pts[1].get_key(),
                                 ctrl_pts[2].get_key(), ctrl_pts[3].get_key()],
                                partial(error_catmull_rom, pt_w))
        #partial是python的functools 模块中的一个高阶函，用于固定参数，返回一个新函数
        # gf = gtsam.CustomFactor(noise_model,
        #                         [ctrl_pts[0].get_key(), ctrl_pts[1].get_key(),
        #                          ctrl_pts[2].get_key(), ctrl_pts[3].get_key()],
        #                         partial(p2tan_catmull_rom, pt_w))
        # gf = gtsam.CustomFactor(noise_model,
        #                         [ctrl_pts[1].get_key(),
        #                          ctrl_pts[2].get_key()],
        #                         partial(p2tan_factor, [pt_w, pt_c, ctrl_pts[0].item, ctrl_pts[3].item]))
        # gf = gtsam.CustomFactor(noise_model,
        #                         [ctrl_pts[1].get_key(), ctrl_pts[2].get_key(), ctrl_pts[3].get_key()],
        #                         partial(p2tan_factor3, [pt_w, pt_c, ctrl_pts[0].item]))
        return gf
    def odo_graph(self):
        # add pose to graph
        odo_noise = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        # odo_noise[:3] = [theta * np.pi / 180.0 for theta in odo_noise[:3]]
        odo_noise_model = gtsam.noiseModel.Diagonal.Sigmas(odo_noise)
        if self.cur_frame.frame_id == 0:
            self.graph.add(gtsam.PriorFactorPose3(X(self.cur_frame.frame_id),
                                                  gtsam.Pose3(self.cur_frame.T_wc),
                                                  odo_noise_model))
            self.initial_estimate.insert(X(self.cur_frame.frame_id), gtsam.Pose3(self.cur_frame.T_wc))
            self.gtsam_key_frame[X(self.cur_frame.frame_id)] = self.cur_frame
        else:
            self.graph.add(gtsam.PriorFactorPose3(X(self.cur_frame.frame_id - 1),
                                                  gtsam.Pose3(self.prev_frame.T_wc),
                                                  odo_noise_model))
            self.graph.add(gtsam.BetweenFactorPose3(X(self.cur_frame.frame_id - 1),
                                                    X(self.cur_frame.frame_id),
                                                    gtsam.Pose3(self.odo_meas),
                                                    odo_noise_model))
            self.initial_estimate.insert(X(self.cur_frame.frame_id), gtsam.Pose3(self.cur_frame.T_wc))
            self.gtsam_key_frame[X(self.cur_frame.frame_id)] = self.cur_frame
            self.gtsam_key_frame[X(self.cur_frame.frame_id - 1)] = self.prev_frame
        return True

    def update_pose(self):
        self.lm_in_window = list(set([lf.id for frame in self.sliding_window
                                      for lf in frame.get_lane_features()
                                      if lf.id != -1 and lf.id in self.lanes_in_map]))
        for lm_id in self.lm_in_window:
            self.lanes_in_map[lm_id].ctrl_pts.update_kdtree()
        last_pose = self.cur_frame.T_wc
        for i in range(1):
            # add pose to graph
            graph = gtsam.NonlinearFactorGraph()
            initial_estimate = gtsam.Values()
            odo_noise = deepcopy(cfg.pose_update.odom_noise)
            odo_noise[:3] = [theta * np.pi / 180.0 for theta in odo_noise[:3]]
            odo_noise_model = gtsam.noiseModel.Diagonal.Sigmas(odo_noise)
            if self.cur_frame.frame_id == 0:
                graph.add(gtsam.PriorFactorPose3(X(self.cur_frame.frame_id),
                                                 gtsam.Pose3(self.cur_frame.T_wc),
                                                 odo_noise_model))
                initial_estimate.insert(X(self.cur_frame.frame_id), gtsam.Pose3(self.cur_frame.T_wc))
                self.gtsam_key_frame[X(self.cur_frame.frame_id)] = self.cur_frame
            else:
                graph.add(gtsam.PriorFactorPose3(X(self.prev_frame.frame_id),
                                                 gtsam.Pose3(self.prev_frame.T_wc),
                                                 odo_noise_model))
                initial_estimate.insert(X(self.prev_frame.frame_id), gtsam.Pose3(self.prev_frame.T_wc))
                graph.add(gtsam.BetweenFactorPose3(X(self.prev_frame.frame_id),
                                                   X(self.cur_frame.frame_id),
                                                   gtsam.Pose3(self.odo_meas),
                                                   odo_noise_model))
                initial_estimate.insert(X(self.cur_frame.frame_id), gtsam.Pose3(self.cur_frame.T_wc))
                self.gtsam_key_frame[X(self.cur_frame.frame_id)] = self.cur_frame
                self.gtsam_key_frame[X(self.cur_frame.frame_id - 1)] = self.prev_frame

            directions = []
            for i, lf in enumerate(self.cur_frame.get_lane_features()):
                if lf.id == -1 or lf.id not in self.lanes_in_map:
                    continue
                lm_id = lf.id
                lm = self.lanes_in_map[lm_id]
                for j, pt_c in enumerate(lf.get_xyzs()):
                    if np.linalg.norm(pt_c) > cfg.pose_update.max_range:
                        continue
                    pt_w = self.cur_frame.T_wc[:3, :3].dot(pt_c[:3]) + self.cur_frame.T_wc[:3, 3]
                    # pt_w = Twc_noise[:3, :3].dot(pt_c[:3]) + Twc_noise[:3, 3]
                    # 找到离得最近的两个控制点，并且node1->next = node2
                    ctrl_pts, u, error = lm.ctrl_pts.find_footpoint(pt_w)
                    if ctrl_pts is None or error > 2.0:
                        continue
                    ctrl_pts_np = np.array([node.item for node in ctrl_pts])
                    if cfg.pose_update.meas_noise > 0:
                        noise = cfg.pose_update.meas_noise
                    else:
                        noise = lf.noise[j]
                    if cfg.pose_update.use_huber:
                        noise_model = self.get_pt_noise_model(noise, huber=True, huber_thresh=cfg.pose_update.huber_thresh)
                    else:
                        noise_model = self.get_pt_noise_model(noise, huber=False)
                    gf = gtsam.CustomFactor(noise_model, [X(self.cur_frame.frame_id)],
                                            partial(PosePointTangentFactor, [pt_c, u, ctrl_pts_np]))
                    graph.add(gf)

                    spline = CatmullRomSpline(ctrl_pts_np)
                    di = spline.get_derivative(u).reshape(3, 1)
                    directions.append(di)

            # optimize
            params = gtsam.GaussNewtonParams()
            params.setMaxIterations(5)
            optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate, params)
            result = optimizer.optimize()
            key = X(self.cur_frame.frame_id)
            # print("true: frame_id: {}, T_wc: {}".format(self.cur_frame.frame_id, se3_to_euler_xyz(self.gtsam_key_frame[key].T_wc)))
            # print("init: frame_id: {}, T_wc: {}".format(self.cur_frame.frame_id, se3_to_euler_xyz(initial_estimate.atPose3(key).matrix())))
            # print("esti: frame_id: {}, T_wc: {}".format(self.cur_frame.frame_id, se3_to_euler_xyz(result.atPose3(key).matrix())))
            if cfg.pose_update.reproject and len(directions) > 0:
                update_pose = inv_se3(self.cur_frame.T_wc) @ result.atPose3(key).matrix()
                degeneracy_d = np.hstack(directions).mean(axis=1).reshape(3, 1)
                degeneracy_d = self.cur_frame.T_wc[:3, :3].T @ degeneracy_d
                degeneracy_d /= np.linalg.norm(degeneracy_d)
                update_xyz = update_pose[:3, 3].reshape(3, 1)
                nex_xyz = (np.eye(3) - degeneracy_d @ degeneracy_d.T) @ update_xyz
                update_pose[:3, 3] = nex_xyz.reshape(3)
                # np.set_printoptions(precision=3, suppress=True)
                # print("update_xyz: ", update_xyz.reshape(3), "degeneracy_d: ", degeneracy_d.reshape(3), "nex_xyz: ", nex_xyz.reshape(3))
                self.cur_frame.T_wc = self.cur_frame.T_wc @ update_pose
            else:
                self.cur_frame.T_wc = result.atPose3(key).matrix()
            delta = np.dot(inv_se3(last_pose), self.cur_frame.T_wc)
            if rot_to_angle(delta[:3, :3], deg=True) < 0.1:
                break
            last_pose = self.cur_frame.T_wc

        return True
    def optimization(self):
        t0 = perf_counter()
        if self.use_isam:
            self.isam.update(self.graph, self.initial_estimate)
            result = self.isam.calculateEstimate()
        else:
            params = gtsam.GaussNewtonParams()
            params.setMaxIterations(5)
            # print('graph size: ', self.graph.size(), ', key size: ', len(self.initial_estimate.keys()))
            # print("graph keys: ", list(map(gtsam.DefaultKeyFormatter, self.graph.keyVector())))
            # print("initial estimate keys: ", list(map(gtsam.DefaultKeyFormatter, self.initial_estimate.keys())))
            optimizer = gtsam.GaussNewtonOptimizer(self.graph, self.initial_estimate, params)
            result = optimizer.optimize()

        # print factor error
        if cfg.debug_flag:
            for pt_w, pt_c, ctrl_pts, lm_id, frame_id, noise, factor_id in self.pts_cp_valid:
                if lm_id != 0:
                    continue
                gf = self.graph.at(factor_id)
                # error = gf.error(result)
                # print('factor id: ', factor_id, ', error: ', error)
                # if error > 10:
                #     print("gf: ", gf)
        # print("Initial error: ", self.graph.error(self.initial_estimate))
        # print("Final error: ", self.graph.error(result))
        # print('0th factor error: ', self.graph.at(0).error(result))
        # print('0th factor keys: ', list(map(DefaultKeyFormatter, self.graph.at(0).keys())))
        # self.graph.printErrors(result)
        #
        # # print node value
        # for key in result.keys():
        #     print('key: ', gtsam.DefaultKeyFormatter(key), 'value: ', result.atPoint3(key))
        for key in result.keys():
            # print('key', gtsam.DefaultKeyFormatter(key), ', in graph: ', key in self.key_in_graph.keys())
            if key in self.key_in_graph.keys():
                node_update = self.key_in_graph[key]
                node_update.set_item(result.atPoint3(key)) # node in map also updated
            elif key in self.gtsam_key_frame.keys():
                frame = self.gtsam_key_frame[key]
                frame.T_wc = result.atPose3(key).matrix()
            else:
                print('key not in graph: ', gtsam.DefaultKeyFormatter(key))

        for lane in self.lanes_in_map.values():
            lane.smooth()

        if cfg.lane_mapping.init_after_opt:
            self.create_new_lane()
        self.opt_timer.update(perf_counter() - t0)

    def create_new_lane(self):
        for lf in self.cur_frame.get_lane_features():#关联后的det线（id==lm的id, 同时新的线是在lm线id的最大值上+1）
            if lf.id == -1:
                continue

            lane_feature_w = self.cur_frame.transform_to_world(lf) #det线从cam坐标系转到world坐标系
            lane_feature_w.fitting() #三次多项式拟合

            if lane_feature_w.id not in self.lanes_in_map:
                # add new lane, build new KDTree
                self.lanes_in_map[lane_feature_w.id] = lane_feature_w
                self.lanes_in_map[lane_feature_w.id].init_ctrl_pts(lane_feature_w, self.cur_frame.T_cw) #起始点是线的第一个点
            else:
                self.lanes_in_map[lane_feature_w.id].update_ctrl_pts(lane_feature_w) #起始点是控制点的最后一个点 （控制点就是抽稀后的能表达线几何的关键点）

    def add_chordal_factor(self, key_status_cur):
        chordal_factors_cur = {}
        constrainted_keys = self.graph.keyVector()
        # constrainted_keys = key_status_cur.keys()
        for key in constrainted_keys:
            if key not in self.key_in_graph.keys():
                continue
            node = self.key_in_graph[key]
            lane_id = node.get_lane_id()
            if self.lanes_in_map[lane_id].size() < 2:
                continue
            idx = self.lanes_in_map[lane_id].get_ctrl_pt_idx(node)
            if idx != 0:
                adj_key = self.lanes_in_map[lane_id].get_ctrl_node(idx - 1).get_key()
                key_pair = (adj_key, key)
            else:
                adj_key = self.lanes_in_map[lane_id].get_ctrl_node(idx + 1).get_key()
                key_pair = (key, adj_key)
            if adj_key is not None and adj_key in constrainted_keys:
                chordal_factors_cur[key_pair] = True
            if idx != self.lanes_in_map[lane_id].size() - 1:
                adj_key = self.lanes_in_map[lane_id].get_ctrl_node(idx + 1).get_key()
                key_pair = (key, adj_key)
            if adj_key is not None and adj_key in constrainted_keys:
                chordal_factors_cur[key_pair] = True

        # merge
        for key_pair in chordal_factors_cur.keys():
            if key_pair not in self.chordal_factors:
                self.chordal_factors[key_pair] = True
                # noise_model = self.get_pt_noise_model(0.2, huber=False, dim=1)
                # self.graph.add(gtsam.CustomFactor(noise_model,
                #                                   [key_pair[0], key_pair[1]],
                #                                   partial(chord_factor, cfg.lane_mapping.ctrl_points_chord)))
                # self.graph.add(gtsam.CustomFactor(noise_model,
                #                                   [key_pair[1]],
                #                                   partial(chord_factor2, [cfg.lane_mapping.ctrl_points_chord, key_pair[0]])))
                noise_model = self.get_pt_noise_model(1, huber=False, dim=3)
                meas = self.key_in_graph[key_pair[1]].item - self.key_in_graph[key_pair[0]].item
                self.graph.add(gtsam.BetweenFactorPoint3(key_pair[0], key_pair[1], meas, noise_model))
                # print('chordal factor: ', self.graph.at(self.graph.size() - 1).error(self.initial_estimate))
    def update_key_status(self, ctrl_pts, key_status):
        # first element: observation times, second element: main observation times, third element: factor id
        for ctrl_pt in ctrl_pts:
            if ctrl_pt.get_key() not in key_status.keys():
                key_status[ctrl_pt.get_key()] = [1, 0]
            else:
                key_status[ctrl_pt.get_key()][0] += 1
        key_status[ctrl_pts[1].get_key()][1] += 1
        key_status[ctrl_pts[2].get_key()][1] += 1

    def add_ctrl_factor(self, key_status):
        if self.use_isam:
            constrainted_keys = self.graph.keys()

        key_unstable = []
        for key, obs in key_status.items():
            if obs[0] < 4 or obs[1] < 1:
            # if obs[1] < 1:
                key_unstable.append(key)
        for key in key_unstable:
            node = self.key_in_graph[key]
            # print("add_ctrl_factor key: ", gtsam.DefaultKeyFormatter(key))
            self.initial_estimate.insert_or_assign(key, gtsam.Point3(node.item))
            ctrlpt_noise = deepcopy(cfg.lane_mapping.ctrl_noise)
            ctrlpt_noise_model = gtsam.noiseModel.Diagonal.Sigmas(ctrlpt_noise)
            self.graph.add(gtsam.PriorFactorPoint3(key, gtsam.Point3(node.item), ctrlpt_noise_model))
        return

    def slide_window(self):
        if self.use_isam:
            return
        if len(self.sliding_window) < self.window_size + 1:
            return
        latest_frame = self.sliding_window[-1]
        if self.margin_old:
            self.sliding_window.pop(0)
            self.sliding_window.append(latest_frame)
        else:
            self.sliding_window.pop(-2)
            self.sliding_window.insert(-1, latest_frame)

    def set_gtsam_symbol(self, lane_id, ctrl_pt_id, node:Node):
        if (lane_id, ctrl_pt_id) in self.lanes_in_graph:
            return True
        else:
            idx = len(self.lanes_in_graph)
            self.lanes_in_graph[(lane_id, ctrl_pt_id)] = L(idx) 
            #L(idx) 是来自 gtsam.symbol_shorthand 的快捷函数,  from gtsam.symbol_shorthand import L
            #用来生成一个 GTSAM 变量 key（整数）——它把字母符号 'L' 和索引 idx 打包成一个 gtsam Key（相当于 gtsam.Symbol('L', idx)）,比如L(5),就是创建了key:L5
            self.key_in_graph[L(idx)] = node
            node.set_key(L(idx))
            node.set_lane_id(lane_id)
            return False
    def graph_init(self):
        self.gtsam_key_frame = {}
        self.lanes_in_graph = {} # lane id -> control points in graph
        self.key_in_graph = {} # key -> lane id, control point id
        self.key_status = {} # key -> [observation times, main observation times, factor id]
        self.factor_candidates = []
        self.chordal_factors = {}
        self.sliding_window = [] # window_size + 1
        self.window_size = cfg.lane_mapping.window_size
        self.margin_old = True
        self.use_isam = True

        if self.use_isam:
            parameters = gtsam.ISAM2Params()
            parameters.setRelinearizeThreshold(0.0) # default 0.1
            parameters.relinearizeSkip = 0 # default 10
            parameters.enableRelinearization = False # default true
            self.isam = gtsam.ISAM2(parameters)


    def get_pt_noise_model(self, noise, huber=False, dim=3, huber_thresh=1.0):
        ctrlpt_noise_model = gtsam.noiseModel.Isotropic.Sigma(dim, noise)
        if huber:
            ctrlpt_noise_model = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber(huber_thresh),
                ctrlpt_noise_model)
        return ctrlpt_noise_model