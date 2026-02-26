import sm
import aslam_backend as aopt
import aslam_cv as cv
import numpy as np

def addPoseDesignVariable(problem, T0=sm.Transformation()):
    q_Dv = aopt.RotationQuaternionDv( T0.q() )
    q_Dv.setActive( True )
    problem.addDesignVariable(q_Dv)
    t_Dv = aopt.EuclideanPointDv( T0.t() )
    t_Dv.setActive( True )
    problem.addDesignVariable(t_Dv)
    return aopt.TransformationBasicDv( q_Dv.toExpression(), t_Dv.toExpression() )

def stereoCalibrate(camL_geometry, camH_geometry, obslist, distortionActive=False, baseline=None):
    #####################################################
    ## find initial guess as median of  all pnp solutions
    #####################################################
    if baseline is None:
        r=[]; t=[]
        for obsL, obsH in obslist:
            #if we have observations for both camss
            if obsL is not None and obsH is not None:
                success, T_L = camL_geometry.geometry.estimateTransformation(obsL)
                success, T_H = camH_geometry.geometry.estimateTransformation(obsH)
                
                baseline = T_H.inverse()*T_L
                t.append(baseline.t())
                rv=sm.RotationVector()
                r.append(rv.rotationMatrixToParameters( baseline.C() ))
        
        r_median = np.median(np.asmatrix(r), axis=0).flatten().T
        R_median = rv.parametersToRotationMatrix(r_median)
        t_median = np.median(np.asmatrix(t), axis=0).flatten().T
        
        baseline_HL = sm.Transformation( sm.rt2Transform(R_median, t_median) )
    else:
        baseline_HL = baseline
    
    #verbose output
    if sm.getLoggingLevel()==sm.LoggingLevel.Debug:
        dL = camL_geometry.geometry.projection().distortion().getParameters().flatten()
        pL = camL_geometry.geometry.projection().getParameters().flatten()
        dH = camH_geometry.geometry.projection().distortion().getParameters().flatten()
        pH = camH_geometry.geometry.projection().getParameters().flatten()
        sm.logDebug("initial guess for stereo calib: {0}".format(baseline_HL.T()))
        sm.logDebug("initial guess for intrinsics camL: {0}".format(pL))
        sm.logDebug("initial guess for intrinsics camH: {0}".format(pH))
        sm.logDebug("initial guess for distortion camL: {0}".format(dL))
        sm.logDebug("initial guess for distortion camH: {0}".format(dH))    
    
    ############################################
    ## solve the bundle adjustment
    ############################################
    problem = aopt.OptimizationProblem()

    #baseline design variable        
    baseline_dv = addPoseDesignVariable(problem, baseline_HL)
        
    #target pose dv for all target views (=T_camL_w)
    target_pose_dvs = list()
    for obsL, obsH in obslist:
        if obsL is not None: #use camL if we have an obs for this one
            success, T_t_cL = camL_geometry.geometry.estimateTransformation(obsL)
        else:
            success, T_t_cH = camH_geometry.geometry.estimateTransformation(obsH)
            T_t_cL = T_t_cH*baseline_HL #apply baseline for the second camera
            
        target_pose_dv = addPoseDesignVariable(problem, T_t_cL)
        target_pose_dvs.append(target_pose_dv)
    
    #add camera dvs
    camL_geometry.setDvActiveStatus(True, distortionActive, False)
    camH_geometry.setDvActiveStatus(True, distortionActive, False)
    problem.addDesignVariable(camL_geometry.dv.distortionDesignVariable())
    problem.addDesignVariable(camL_geometry.dv.projectionDesignVariable())
    problem.addDesignVariable(camL_geometry.dv.shutterDesignVariable())
    problem.addDesignVariable(camH_geometry.dv.distortionDesignVariable())
    problem.addDesignVariable(camH_geometry.dv.projectionDesignVariable())
    problem.addDesignVariable(camH_geometry.dv.shutterDesignVariable())
    
    ############################################
    ## add error terms
    ############################################
    
    #corner uncertainty
    # \todo pass in the detector uncertainty somehow.
    cornerUncertainty = 1.0
    R = np.eye(2) * cornerUncertainty * cornerUncertainty
    invR = np.linalg.inv(R)
        
    #Add reprojection error terms for both cameras
    reprojectionErrors0 = []; reprojectionErrors1 = []
            
    for cidx, cam in enumerate([camL_geometry, camH_geometry]):
        sm.logDebug("stereoCalibration: adding camera error terms for {0} calibration targets".format(len(obslist)))

        #get the image and target points corresponding to the frame
        target = cam.ctarget.detector.target()
        
        #add error terms for all observations
        for view_id, obstuple in enumerate(obslist):
            
            #add error terms if we have an observation for this cam
            obs=obstuple[cidx]
            if obs is not None:
                T_cam_w = target_pose_dvs[view_id].toExpression().inverse()
            
                #add the baseline for the second camera
                if cidx!=0:
                    T_cam_w =  baseline_dv.toExpression() * T_cam_w
                    
                for i in range(0, target.size()):
                    p_target = aopt.HomogeneousExpression(sm.toHomogeneous(target.point(i)));
                    valid, y = obs.imagePoint(i)
                    if valid:
                        # Create an error term.
                        rerr = cam.model.reprojectionError(y, invR, T_cam_w * p_target, cam.dv)
                        rerr.idx = i
                        problem.addErrorTerm(rerr)
                    
                        if cidx==0:
                            reprojectionErrors0.append(rerr)
                        else:
                            reprojectionErrors1.append(rerr)
                                                        
        sm.logDebug("stereoCalibrate: added {0} camera error terms".format( len(reprojectionErrors0)+len(reprojectionErrors1) ))
        
    ############################################
    ## solve
    ############################################       
    options = aopt.Optimizer2Options()
    options.verbose = True if sm.getLoggingLevel()==sm.LoggingLevel.Debug else False
    options.nThreads = 4
    options.convergenceDeltaX = 1e-3
    options.convergenceDeltaJ = 1
    options.maxIterations = 200
    options.trustRegionPolicy = aopt.LevenbergMarquardtTrustRegionPolicy(10)

    optimizer = aopt.Optimizer2(options)
    optimizer.setProblem(problem)

    #verbose output
    if sm.getLoggingLevel()==sm.LoggingLevel.Debug:
        sm.logDebug("Before optimization:")
        e2 = np.array([ e.evaluateError() for e in reprojectionErrors0 ])
        sm.logDebug( " Reprojection error squarred (camL):  mean {0}, median {1}, std: {2}".format(np.mean(e2), np.median(e2), np.std(e2) ) )
        e2 = np.array([ e.evaluateError() for e in reprojectionErrors1 ])
        sm.logDebug( " Reprojection error squarred (camH):  mean {0}, median {1}, std: {2}".format(np.mean(e2), np.median(e2), np.std(e2) ) )
    
        sm.logDebug("baseline={0}".format(baseline_dv.toTransformationMatrix()))
    
    try: 
        retval = optimizer.optimize()
        if retval.linearSolverFailure:
            sm.logError("stereoCalibrate: Optimization failed!")
        success = not retval.linearSolverFailure
    except:
        sm.logError("stereoCalibrate: Optimization failed!")
        success = False
    
    if sm.getLoggingLevel()==sm.LoggingLevel.Debug:
        sm.logDebug("After optimization:")
        e2 = np.array([ e.evaluateError() for e in reprojectionErrors0 ])
        sm.logDebug( " Reprojection error squarred (camL):  mean {0}, median {1}, std: {2}".format(np.mean(e2), np.median(e2), np.std(e2) ) )
        e2 = np.array([ e.evaluateError() for e in reprojectionErrors1 ])
        sm.logDebug( " Reprojection error squarred (camH):  mean {0}, median {1}, std: {2}".format(np.mean(e2), np.median(e2), np.std(e2) ) )
    
    #verbose output
    if sm.getLoggingLevel()==sm.LoggingLevel.Debug:
        dL = camL_geometry.geometry.projection().distortion().getParameters().flatten()
        pL = camL_geometry.geometry.projection().getParameters().flatten()
        dH = camH_geometry.geometry.projection().distortion().getParameters().flatten()
        pH = camH_geometry.geometry.projection().getParameters().flatten()
        sm.logDebug("guess for intrinsics camL: {0}".format(pL))
        sm.logDebug("guess for intrinsics camH: {0}".format(pH))
        sm.logDebug("guess for distortion camL: {0}".format(dL))
        sm.logDebug("guess for distortion camH: {0}".format(dH))    
    
    if success:
        baseline_HL = sm.Transformation(baseline_dv.toTransformationMatrix())
        return success, baseline_HL
    else:
        #return the intiial guess if we fail
        return success, baseline_HL


def _dump_calibrateIntrinsics_state(tag, cam_geometry, obslist, extra=None):
    """Dump the full state when NaN is detected during calibrateIntrinsics."""
    import dill, datetime, traceback
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dump_path = f"/data/nan_calibrateIntrinsics_{tag}_{ts}.pkl"

    proj = cam_geometry.geometry.projection().getParameters().flatten()
    dist = cam_geometry.geometry.projection().distortion().getParameters().flatten()

    print(f"\n{'='*72}")
    print(f"[NaN DETECTED] calibrateIntrinsics — tag='{tag}'")
    print(f"  projection: {proj}")
    print(f"  distortion: {dist}")
    print(f"  num observations: {len(obslist)}")
    traceback.print_stack()
    print(f"{'='*72}\n")

    state = {
        "tag": tag,
        "projection_params": proj,
        "distortion_params": dist,
        "num_observations": len(obslist),
        "obs_corner_counts": [obs.getCornersImageFrame().shape[0] for obs in obslist],
    }
    if extra:
        state.update(extra)

    try:
        with open(dump_path, "wb") as f:
            dill.dump(state, f)
        print(f"[NaN DUMP] saved to {dump_path}")
    except Exception as e:
        print(f"[NaN DUMP] failed: {e}")


def calibrateIntrinsics(cam_geometry, obslist, distortionActive=True, intrinsicsActive=True):
    d_init = cam_geometry.geometry.projection().distortion().getParameters().flatten()
    p_init = cam_geometry.geometry.projection().getParameters().flatten()

    mode_label = "full" if distortionActive else "projOnly"
    print(f"  [calibrateIntrinsics:{mode_label}] ENTRY  proj={p_init}  dist={d_init}")

    if np.any(np.isnan(p_init)) or np.any(np.isnan(d_init)):
        _dump_calibrateIntrinsics_state(f"entry_{mode_label}", cam_geometry, obslist)
        print(f"  [calibrateIntrinsics:{mode_label}] input params already NaN — aborting")
        return False

    #verbose output
    if sm.getLoggingLevel()==sm.LoggingLevel.Debug:
        sm.logDebug("calibrateIntrinsics: intrinsics guess: {0}".format(p_init))
        sm.logDebug("calibrateIntrinsics: distortion guess: {0}".format(d_init))
    
    ############################################
    ## solve the bundle adjustment
    ############################################
    problem = aopt.OptimizationProblem()
    
    #add camera dvs
    cam_geometry.setDvActiveStatus(intrinsicsActive, distortionActive, False)
    problem.addDesignVariable(cam_geometry.dv.distortionDesignVariable())
    problem.addDesignVariable(cam_geometry.dv.projectionDesignVariable())
    problem.addDesignVariable(cam_geometry.dv.shutterDesignVariable())
    
    #corner uncertainty
    cornerUncertainty = 1.0
    R = np.eye(2) * cornerUncertainty * cornerUncertainty
    invR = np.linalg.inv(R)
    
    #get the image and target points corresponding to the frame
    target = cam_geometry.ctarget.detector.target()
    proj = cam_geometry.geometry.projection()

    grid_size = getattr(cam_geometry, 'weight_map_grid', 0)
    use_weight_map = grid_size > 0
    weight_map_power  = getattr(cam_geometry, 'weight_map_power',  2.0)
    weight_map_cutoff = getattr(cam_geometry, 'weight_map_cutoff', 0.0)
    
    sm.logDebug("calibrateIntrinsics: adding camera error terms for {0} calibration targets".format(len(obslist)))
    nan_transform_count = 0

    ############################################
    ## Pass 1: estimate poses, compute Jacobian norms, build weight map
    ############################################
    obs_records = []
    all_jp_for_map = []
    all_uv_for_map = []
    all_err_for_map = []

    for obs_idx, obs in enumerate(obslist):
        success, T_t_c = cam_geometry.geometry.estimateTransformation(obs)
        T_mat = T_t_c.T()
        if np.any(np.isnan(T_mat)):
            nan_transform_count += 1
            if nan_transform_count <= 3:
                print(f"  [calibrateIntrinsics:{mode_label}] estimateTransformation NaN for obs {obs_idx}, "
                      f"corners={obs.getCornersImageFrame().shape[0]}")

        T_c_t_mat = np.linalg.inv(T_mat)
        corners_this = obs.getCornersImageFrame()

        corner_list = []
        for i in range(target.size()):
            valid, y = obs.imagePoint(i)
            if valid:
                p_t = np.append(target.point(i), 1.0)
                p_cam = T_c_t_mat @ p_t
                try:
                    kp, Jp, jvalid = proj.euclideanToKeypointJp(p_cam[:3])
                    jp_norm = np.linalg.norm(Jp) if jvalid else float('nan')
                    err_init = np.linalg.norm(y - kp) if jvalid else float('nan')
                except:
                    jp_norm = float('nan')
                    err_init = float('nan')
                corner_list.append((i, y, jp_norm, err_init))
                all_jp_for_map.append(jp_norm)
                all_uv_for_map.append(y)
                all_err_for_map.append(err_init)

        obs_records.append({
            "T_t_c": T_t_c,
            "corners_this": corners_this,
            "corner_list": corner_list,
        })

    # Build weight map — gradient-contribution squared regularisation
    # g_i = |e_i| * ||J_i||  (gradient contribution of each corner)
    # Per-cell mean: gc_cell = mean(g_i)   Reference: median_gc over all cells
    # weight = min(1, (median_gc / gc_cell)^power)
    # Optional hard cutoff: gc_cell > cutoff * median_gc  →  weight = 0
    weight_map = None
    if use_weight_map and len(all_uv_for_map) > 0:
        img_w = int(obslist[0].imCols())
        img_h = int(obslist[0].imRows())
        n_gx = (img_w + grid_size - 1) // grid_size
        n_gy = (img_h + grid_size - 1) // grid_size

        gc_sum = np.zeros((n_gy, n_gx))
        gc_cnt = np.zeros((n_gy, n_gx))
        for uv, jp, err in zip(all_uv_for_map, all_jp_for_map, all_err_for_map):
            gc = float(np.abs(err) * jp) if (np.isfinite(jp) and np.isfinite(err)) else float('nan')
            if not np.isfinite(gc):
                continue
            gx = min(int(uv[0]) // grid_size, n_gx - 1)
            gy = min(int(uv[1]) // grid_size, n_gy - 1)
            gc_sum[gy, gx] += gc
            gc_cnt[gy, gx] += 1

        occupied = gc_cnt > 0
        gc_mean_map = np.where(occupied, gc_sum / np.where(occupied, gc_cnt, 1.0), np.nan)

        median_gc = float(np.nanmedian(gc_mean_map[occupied]))
        if median_gc <= 0 or not np.isfinite(median_gc):
            median_gc = 1.0

        # w = min(1, (median_gc / gc_cell)^power);  cutoff → 0 if gc_cell > cutoff*median
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = np.where(occupied, median_gc / gc_mean_map, 1.0)
        w_raw = np.where(occupied, np.clip(ratio ** weight_map_power, 0.0, 1.0), 1.0)
        if weight_map_cutoff > 0:
            w_raw = np.where(occupied & (gc_mean_map > weight_map_cutoff * median_gc),
                             0.0, w_raw)
        weight_map = w_raw

        n_occupied = int(np.sum(occupied))
        wmin, wmax = float(weight_map[occupied].min()), float(weight_map[occupied].max())
        print(f"  [calibrateIntrinsics:{mode_label}] Weight map (grad-contrib^2): "
              f"grid={grid_size}px  power={weight_map_power}  cutoff={weight_map_cutoff}  "
              f"median_gc={median_gc:.2f}  occupied_cells={n_occupied}  "
              f"weight range=[{wmin:.4f}, {wmax:.4f}]")

    ############################################
    ## Pass 2: add design variables and weighted error terms
    ############################################
    reprojectionErrors = []
    target_pose_dvs = list()
    nan_reproj_count = 0
    per_obs_errors = []
    per_obs_meta = []
    diag_uv = []
    diag_err = []
    diag_jp_norm = []
    diag_weight = []

    for obs_idx, rec in enumerate(obs_records):
        T_t_c = rec["T_t_c"]
        corners_this = rec["corners_this"]

        target_pose_dv = addPoseDesignVariable(problem, T_t_c)
        target_pose_dvs.append(target_pose_dv)
        T_cam_w = target_pose_dv.toExpression().inverse()

        obs_rerrs_this = []
        for (pt_idx, y, jpn, _err_init) in rec["corner_list"]:
            if weight_map is not None:
                gx = min(int(y[0]) // grid_size, weight_map.shape[1] - 1)
                gy = min(int(y[1]) // grid_size, weight_map.shape[0] - 1)
                w = float(weight_map[gy, gx])
                if w <= 1e-12:
                    diag_uv.append(y.copy())
                    diag_err.append(float('nan'))
                    diag_jp_norm.append(jpn)
                    diag_weight.append(0.0)
                    continue
                invR_w = invR * w
            else:
                w = 1.0
                invR_w = invR

            p_target = aopt.HomogeneousExpression(sm.toHomogeneous(target.point(pt_idx)))
            rerr = cam_geometry.model.reprojectionError(y, invR_w, T_cam_w * p_target, cam_geometry.dv)
            err_val = rerr.evaluateError()
            obs_rerrs_this.append(err_val)
            if np.isnan(err_val):
                nan_reproj_count += 1
                if nan_reproj_count <= 5:
                    print(f"  [calibrateIntrinsics:{mode_label}] NaN reproj error: obs={obs_idx} pt={pt_idx} y={y}")
            problem.addErrorTerm(rerr)
            reprojectionErrors.append(rerr)

            diag_uv.append(y.copy())
            diag_err.append(err_val)
            diag_jp_norm.append(jpn)
            diag_weight.append(w)

        obs_err_arr = np.array(obs_rerrs_this)
        per_obs_errors.append(float(np.nanmean(obs_err_arr)) if len(obs_err_arr) > 0 else float('nan'))
        per_obs_meta.append({
            "n_corners": corners_this.shape[0],
            "x_min": float(corners_this[:,0].min()), "x_max": float(corners_this[:,0].max()),
            "y_min": float(corners_this[:,1].min()), "y_max": float(corners_this[:,1].max()),
        })

    diag_uv = np.array(diag_uv) if diag_uv else np.empty((0, 2))
    diag_err = np.array(diag_err)
    diag_jp_norm = np.array(diag_jp_norm)
    diag_weight = np.array(diag_weight)
    diag_grad_contrib = np.abs(diag_err) * diag_jp_norm

    # Collect per-frame camera pose data for visualization.
    # T_t_c = pose of camera expressed in the target frame (T_{target <- camera}).
    # T_t_c.t() gives the camera origin position in target coordinates.
    _pose_t_list = []
    _pose_rv_list = []
    for _rec in obs_records:
        _T_local = _rec["T_t_c"]
        _T_mat_local = _T_local.T()
        if not np.any(np.isnan(_T_mat_local)):
            _pose_t_list.append(_T_local.t().flatten())
            try:
                from scipy.spatial.transform import Rotation as _ScipyR
                _rvec = _ScipyR.from_matrix(_T_local.C()).as_rotvec()
            except Exception:
                _rvec = np.full(3, np.nan)
            _pose_rv_list.append(_rvec)
        else:
            _pose_t_list.append(np.full(3, np.nan))
            _pose_rv_list.append(np.full(3, np.nan))
    _pose_translations = np.array(_pose_t_list) if _pose_t_list else np.empty((0, 3))
    _pose_rotvecs = np.array(_pose_rv_list) if _pose_rv_list else np.empty((0, 3))

    if not hasattr(cam_geometry, '_diag_per_stage'):
        cam_geometry._diag_per_stage = {}
    cam_geometry._diag_per_stage[mode_label] = {
        "uv": diag_uv,
        "reproj_err": diag_err,
        "jp_norm": diag_jp_norm,
        "grad_contrib": diag_grad_contrib,
        "weight": diag_weight,
        "weight_map": weight_map,
        "proj_params": p_init.copy(),
        "dist_params": d_init.copy(),
        "pose_translations": _pose_translations,
        "pose_rotations_rotvec": _pose_rotvecs,
    }

    if nan_transform_count > 0:
        print(f"  [calibrateIntrinsics:{mode_label}] Total NaN transforms: {nan_transform_count}/{len(obslist)}")
    if nan_reproj_count > 0:
        print(f"  [calibrateIntrinsics:{mode_label}] Total NaN reproj errors: {nan_reproj_count}/{len(reprojectionErrors)}")

    sm.logDebug("calibrateIntrinsics: added {0} camera error terms".format(len(reprojectionErrors)))

    # Pre-optimization error statistics
    e2_pre = np.array([e.evaluateError() for e in reprojectionErrors])
    nan_e2_count = np.sum(np.isnan(e2_pre))
    print(f"  [calibrateIntrinsics:{mode_label}] PRE-OPT errors: "
          f"mean={np.nanmean(e2_pre):.4f} median={np.nanmedian(e2_pre):.4f} "
          f"std={np.nanstd(e2_pre):.4f} nan_count={nan_e2_count}/{len(e2_pre)}")

    # Per-observation error distribution
    poe = np.array(per_obs_errors)
    finite_poe = poe[np.isfinite(poe)]
    if len(finite_poe) > 0:
        pcts = np.percentile(finite_poe, [0, 10, 25, 50, 75, 90, 95, 99, 100])
        print(f"  [calibrateIntrinsics:{mode_label}] PER-OBS mean-error distribution (n={len(finite_poe)}):")
        print(f"    p0={pcts[0]:.2f}  p10={pcts[1]:.2f}  p25={pcts[2]:.2f}  p50={pcts[3]:.2f}  "
              f"p75={pcts[4]:.2f}  p90={pcts[5]:.2f}  p95={pcts[6]:.2f}  p99={pcts[7]:.2f}  p100={pcts[8]:.2f}")

        bad_threshold = pcts[7]  # p99
        bad_mask = finite_poe > bad_threshold
        good_mask = finite_poe <= pcts[3]  # <= median

        bad_indices = np.where(poe > bad_threshold)[0]
        good_indices = np.where((poe <= pcts[3]) & np.isfinite(poe))[0]

        print(f"    WORST {len(bad_indices)} obs (>p99={bad_threshold:.1f}):")
        for rank, bi in enumerate(bad_indices[np.argsort(poe[bad_indices])[::-1][:10]]):
            m = per_obs_meta[bi]
            print(f"      obs[{bi}]: mean_err={poe[bi]:.1f}  corners={m['n_corners']}  "
                  f"x=[{m['x_min']:.0f},{m['x_max']:.0f}]  y=[{m['y_min']:.0f},{m['y_max']:.0f}]")

        # Spatial summary: good vs bad observations
        if len(good_indices) > 5 and len(bad_indices) > 0:
            g_xmin = np.mean([per_obs_meta[i]["x_min"] for i in good_indices])
            g_xmax = np.mean([per_obs_meta[i]["x_max"] for i in good_indices])
            g_ymin = np.mean([per_obs_meta[i]["y_min"] for i in good_indices])
            g_ymax = np.mean([per_obs_meta[i]["y_max"] for i in good_indices])
            b_xmin = np.mean([per_obs_meta[i]["x_min"] for i in bad_indices])
            b_xmax = np.mean([per_obs_meta[i]["x_max"] for i in bad_indices])
            b_ymin = np.mean([per_obs_meta[i]["y_min"] for i in bad_indices])
            b_ymax = np.mean([per_obs_meta[i]["y_max"] for i in bad_indices])
            print(f"    SPATIAL: good(n={len(good_indices)}) avg bbox x=[{g_xmin:.0f},{g_xmax:.0f}] y=[{g_ymin:.0f},{g_ymax:.0f}]")
            print(f"    SPATIAL: bad (n={len(bad_indices)}) avg bbox x=[{b_xmin:.0f},{b_xmax:.0f}] y=[{b_ymin:.0f},{b_ymax:.0f}]")

    ############################################
    ## solve
    ############################################       
    options = aopt.Optimizer2Options()
    options.verbose = True if sm.getLoggingLevel()==sm.LoggingLevel.Debug else False
    options.nThreads = 4
    options.convergenceDeltaX = 1e-3
    options.convergenceDeltaJ = 1
    options.maxIterations = 200
    options.trustRegionPolicy = aopt.LevenbergMarquardtTrustRegionPolicy(10)

    optimizer = aopt.Optimizer2(options)
    optimizer.setProblem(problem)

    #verbose output
    if sm.getLoggingLevel()==sm.LoggingLevel.Debug:
        sm.logDebug("Before optimization:")
        sm.logDebug( " Reprojection error squarred (camL):  mean {0}, median {1}, std: {2}".format(np.mean(e2_pre), np.median(e2_pre), np.std(e2_pre) ) )
    
    #run intrinsic calibration
    try: 
        retval = optimizer.optimize()
        if retval.linearSolverFailure:
            sm.logError("calibrateIntrinsics: Optimization failed!")
        success = not retval.linearSolverFailure

    except Exception as exc:
        sm.logError("calibrateIntrinsics: Optimization failed! Exception: {0}".format(exc))
        success = False

    # Post-optimization check
    p_post = cam_geometry.geometry.projection().getParameters().flatten()
    d_post = cam_geometry.geometry.projection().distortion().getParameters().flatten()
    print(f"  [calibrateIntrinsics:{mode_label}] EXIT   proj={p_post}  dist={d_post}  success={success}")

    if np.any(np.isnan(p_post)) or np.any(np.isnan(d_post)):
        _dump_calibrateIntrinsics_state(f"post_opt_{mode_label}", cam_geometry, obslist, extra={
            "proj_before": p_init, "dist_before": d_init,
            "proj_after": p_post, "dist_after": d_post,
            "optimizer_success": success,
            "nan_transform_count": nan_transform_count,
            "nan_reproj_count": nan_reproj_count,
            "pre_opt_errors_mean": float(np.nanmean(e2_pre)),
            "pre_opt_errors_nan_count": int(nan_e2_count),
        })

    #verbose output
    if sm.getLoggingLevel()==sm.LoggingLevel.Debug:
        sm.logDebug("calibrateIntrinsics: guess for intrinsics cam: {0}".format(p_post))
        sm.logDebug("calibrateIntrinsics: guess for distortion cam: {0}".format(d_post))
    
    return success


def solveFullBatch(cameras, baseline_guesses, graph):    
    ############################################
    ## solve the bundle adjustment
    ############################################
    problem = aopt.OptimizationProblem()
    
    #add camera dvs
    for cam in cameras:
        cam.setDvActiveStatus(True, True, False)
        problem.addDesignVariable(cam.dv.distortionDesignVariable())
        problem.addDesignVariable(cam.dv.projectionDesignVariable())
        problem.addDesignVariable(cam.dv.shutterDesignVariable())
    
    baseline_dvs = list()
    for baseline_idx in range(0, len(cameras)-1): 
        baseline_dv = aopt.TransformationDv(baseline_guesses[baseline_idx])
        
        for i in range(0, baseline_dv.numDesignVariables()):
            problem.addDesignVariable(baseline_dv.getDesignVariable(i))
        
        baseline_dvs.append( baseline_dv )
    
    #corner uncertainty
    cornerUncertainty = 1.0
    R = np.eye(2) * cornerUncertainty * cornerUncertainty
    invR = np.linalg.inv(R)
    
    #get the target
    target = cameras[0].ctarget.detector.target()

    #Add calibration target reprojection error terms for all camera in chain
    target_pose_dvs = list()
      
    #shuffle the views
    reprojectionErrors = [];    
    timestamps = graph.obs_db.getAllViewTimestamps()
    for view_id, timestamp in enumerate(timestamps):
        
        #get all observations for all cams at this time
        obs_tuple = graph.obs_db.getAllObsAtTimestamp(timestamp)

        #create a target pose dv for all target views (= T_cam0_w)
        T0 = graph.getTargetPoseGuess(timestamp, cameras, baseline_guesses)
        target_pose_dv = addPoseDesignVariable(problem, T0)
        target_pose_dvs.append(target_pose_dv)
        

        for cidx, obs in obs_tuple:
            cam = cameras[cidx]
              
            #calibration target coords to camera X coords
            T_cam0_calib = target_pose_dv.toExpression().inverse()

            #build pose chain (target->cam0->baselines->camN)
            T_camN_calib = T_cam0_calib
            for idx in range(0, cidx):
                T_camN_calib = baseline_dvs[idx].toExpression() * T_camN_calib
                
        
            ## add error terms
            for i in range(0, target.size()):
                p_target = aopt.HomogeneousExpression(sm.toHomogeneous(target.point(i)));
                valid, y = obs.imagePoint(i)
                if valid:
                    rerr = cameras[cidx].model.reprojectionError(y, invR, T_camN_calib * p_target, cameras[cidx].dv)
                    problem.addErrorTerm(rerr)
                    reprojectionErrors.append(rerr)
                                                    
    sm.logDebug("solveFullBatch: added {0} camera error terms".format(len(reprojectionErrors)))
    
    ############################################
    ## solve
    ############################################       
    options = aopt.Optimizer2Options()
    options.verbose = True if sm.getLoggingLevel()==sm.LoggingLevel.Debug else False
    options.nThreads = 4
    options.convergenceDeltaX = 1e-3
    options.convergenceDeltaJ = 1
    options.maxIterations = 250
    options.trustRegionPolicy = aopt.LevenbergMarquardtTrustRegionPolicy(10)

    optimizer = aopt.Optimizer2(options)
    optimizer.setProblem(problem)

    #verbose output
    if sm.getLoggingLevel()==sm.LoggingLevel.Debug:
        sm.logDebug("Before optimization:")
        e2 = np.array([ e.evaluateError() for e in reprojectionErrors ])
        sm.logDebug( " Reprojection error squarred (camL):  mean {0}, median {1}, std: {2}".format(np.mean(e2), np.median(e2), np.std(e2) ) )
    
    #run intrinsic calibration
    try:
        retval = optimizer.optimize()
        if retval.linearSolverFailure:
            sm.logError("calibrateIntrinsics: Optimization failed!")
        success = not retval.linearSolverFailure

    except:
        sm.logError("calibrateIntrinsics: Optimization failed!")
        success = False

    baselines=list()
    for baseline_dv in baseline_dvs:
        baselines.append( sm.Transformation(baseline_dv.T()) )
    
    return success, baselines

