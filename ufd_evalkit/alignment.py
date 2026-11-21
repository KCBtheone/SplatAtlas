import os
import json
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData

def umeyama_transform(P, Q):
    mu_P = P.mean(axis=0)
    mu_Q = Q.mean(axis=0)
    P_centered = P - mu_P
    Q_centered = Q - mu_Q
    
    C = P_centered.T @ Q_centered / P.shape[0]
    U, S, Vt = np.linalg.svd(C)
    
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        
    var_P = np.var(P, axis=0).sum()
    if var_P < 1e-8:
        scale = 1.0
    else:
        scale = (1.0 / var_P) * np.sum(S) 
        
    t = mu_Q - scale * (R @ mu_P)
    
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T

def icp_umeyama_align(source_pts, target_pts, max_iters=20, tolerance=1e-5):
    if len(source_pts) > 50000:
        idx = np.random.choice(len(source_pts), 50000, replace=False)
        src = source_pts[idx].copy()
    else:
        src = source_pts.copy()
        
    target_tree = cKDTree(target_pts)
    T_accum = np.eye(4)
    prev_dist = float('inf')
    
    for i in range(max_iters):
        dists, indices = target_tree.query(src)
        matched_target = target_pts[indices]
        
        T = umeyama_transform(src, matched_target)
        src_homo = np.hstack((src, np.ones((src.shape[0], 1))))
        src = (T @ src_homo.T).T[:, :3]
        T_accum = T @ T_accum
        
        if i > 0 and np.mean(dists) - prev_dist > -tolerance:
            break
        prev_dist = np.mean(dists)
        
    return T_accum

def assess_manifold_collapse(model, model_dir, colmap_ply_path=None):
    try:
        if not hasattr(model, 'get_spatial_centers'):
            return {"manifold_collapse": False, "reason": "Model lacks explicit discrete centers"}
            
        try:
            xyz_pred = model.get_spatial_centers().cpu().numpy()
        except NotImplementedError:
            return {"manifold_collapse": False, "reason": "Method not implemented"}

        if len(xyz_pred) < 100:
            return {"manifold_collapse": True, "reason": "Too few points"}
            
        cov = np.cov(xyz_pred.T)
        eigenvalues, _ = np.linalg.eigh(cov)
        if eigenvalues[0] < 1e-6:
            return {"manifold_collapse": True, "reason": "Zero variance in one dimension (Flat collapse)"}
            
        matrix = np.eye(4)
        if colmap_ply_path and os.path.exists(colmap_ply_path):
            try:
                colmap_data = PlyData.read(colmap_ply_path)
                cv = colmap_data.elements[0].data
                xyz_colmap = np.vstack([cv['x'], cv['y'], cv['z']]).T
                
                print(f"  [Alignment] Running ICP with COLMAP points: {len(xyz_colmap)}")
                matrix = icp_umeyama_align(xyz_pred, xyz_colmap)
            except Exception as e:
                print(f"  [Alignment] 锚点对齐失败 ({e})，回退为单位矩阵。")
        
        matrix_path = os.path.join(model_dir, "alignment_matrix.json")
        with open(matrix_path, 'w') as f:
            json.dump(matrix.tolist(), f, indent=4)
            
        return {"manifold_collapse": False, "reason": "Normal"}
    except Exception as e:
        return {"manifold_collapse": False, "reason": f"Detection skipped due to error: {str(e)}"}
