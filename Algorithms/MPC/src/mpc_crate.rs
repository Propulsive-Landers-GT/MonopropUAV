// A crate for Nonlinear Model Predictive Control
// Hansel Zhang

use ndarray::{Array1, Array2};
use ndarray::s;
use ndarray_linalg::Inverse;
use ndarray_linalg::Norm;
use ndarray::ArrayView1;
use optimization_engine::{panoc::*, *};
use std::time::Instant;

pub struct MPCDebugInfo {
    pub iterations: usize,
}

impl MPCDebugInfo {
    pub fn new() -> Self {
        Self {
            iterations: 0,
        }
    }
}

// pub fn dynamics(x: &Array1<f64>, u: &Array1<f64>, mass: f64, moi: &Array2<f64>, dt: f64) -> Array1<f64> {
//     // let mut x_next = Array1::<f64>::zeros(x.len());
//     // Constants
//     let m = mass;
//     let g = 9.81;
//     // let ixx = 50.0;
//     // let iyy = 50.0;
//     // let izz = 3.0;
//     // let ixx = 71.0;
//     // let iyy = 71.0;
//     // let izz = 3.8;

//     // Inertia matrix and inverse
//     // let i = Array2::from_diag(&Array1::from(vec![ixx, iyy, izz]));
//     let i = moi.clone();
//     let i_inv = i.clone().inv().unwrap();

//     // Unpack state
//     let x_pos = x[0];
//     let y_pos = x[1];
//     let z_pos = x[2];
//     let qx = x[3];
//     let qy = x[4];
//     let qz = x[5];
//     let qw = x[6];
//     let x_dot = x[7];
//     let y_dot = x[8];
//     let z_dot = x[9];
//     let wx = x[10];
//     let wy = x[11];
//     let wz = x[12];

//     // Unpack control
//     let gimbal_theta = u[0];
//     let gimbal_phi = u[1];
//     let mut thrust = u[2];
//     if thrust < 0.0 { thrust = 0.0; }

//     // Thrust direction in body frame
//     let tx = gimbal_theta.sin() * gimbal_phi.cos();
//     let ty = gimbal_phi.sin();
//     let tz = gimbal_theta.cos() * gimbal_phi.cos();
//     let thrust_b = Array1::from(vec![tx * thrust, ty * thrust, tz * thrust]);

//     // Quaternion normalization
//     let mut q = Array1::from(vec![qx, qy, qz, qw]);
//     let q_norm = q.dot(&q).sqrt();
//     if q_norm < 1e-8 {
//         q = Array1::from(vec![0.0, 0.0, 0.0, 1.0]);
//     } else {
//         q = q.mapv(|v| v / q_norm);
//     }
//     let qx = q[0];
//     let qy = q[1];
//     let qz = q[2];
//     let qw = q[3];

//     // Rotation matrix from quaternion
//     let r = Array2::from_shape_vec((3, 3), vec![
//         1.0 - 2.0 * qy * qy - 2.0 * qz * qz, 2.0 * qx * qy - 2.0 * qz * qw, 2.0 * qx * qz + 2.0 * qy * qw,
//         2.0 * qx * qy + 2.0 * qz * qw, 1.0 - 2.0 * qx * qx - 2.0 * qz * qz, 2.0 * qy * qz - 2.0 * qx * qw,
//         2.0 * qx * qz - 2.0 * qy * qw, 2.0 * qy * qz + 2.0 * qx * qw, 1.0 - 2.0 * qx * qx - 2.0 * qy * qy,
//     ]).unwrap();

//     // Acceleration in world frame
//     let acc = r.dot(&thrust_b) / m - Array1::from(vec![0.0, 0.0, g]);
//     let x_ddot = acc[0];
//     let y_ddot = acc[1];
//     let z_ddot = acc[2];

//     // TVC torque
//     let d = 0.663834203726;
//     let r_cp = Array1::from(vec![0.0, 0.0, -d]);
//     let torque_b = Array1::from(vec![
//         r_cp[1] * thrust_b[2] - r_cp[2] * thrust_b[1],
//         r_cp[2] * thrust_b[0] - r_cp[0] * thrust_b[2],
//         r_cp[0] * thrust_b[1] - r_cp[1] * thrust_b[0],
//     ]);

//     // Angular velocity
//     let omega = Array1::from(vec![wx, wy, wz]);
//     // TODO: make it full matrix
//     let omega_cross_i_omega = Array1::from(vec![
//         omega[1] * (i[(1, 1)] * omega[2]) - omega[2] * (i[(2, 2)] * omega[1]),
//         omega[2] * (i[(2, 2)] * omega[0]) - omega[0] * (i[(0, 0)] * omega[2]),
//         omega[0] * (i[(0, 0)] * omega[1]) - omega[1] * (i[(1, 1)] * omega[0]),
//     ]);
//     let omega_dot = i_inv.dot(&(torque_b - omega_cross_i_omega));

//     // Quaternion derivative
//     let dqdt = Array1::from(vec![
//         qw * wx + qy * wz - qz * wy,
//         qw * wy + qz * wx - qx * wz,
//         qw * wz + qx * wy - qy * wx,
//         -qx * wx - qy * wy - qz * wz,
//     ]).mapv(|v| 0.5 * v);

//     // Euler integration
//     // let x_pos_new = x_pos + dt * x_dot;
//     // let y_pos_new = y_pos + dt * y_dot;
//     // let z_pos_new = z_pos + dt * z_dot;
//     let x_pos_new = x_pos + dt * x_dot + 0.5 * dt * dt * x_ddot;
//     let y_pos_new = y_pos + dt * y_dot + 0.5 * dt * dt * y_ddot;
//     let z_pos_new = z_pos + dt * z_dot + 0.5 * dt * dt * z_ddot;
//     let mut q_new = q.clone() + dqdt.mapv(|v| dt * v);
//     let q_new_norm = q_new.dot(&q_new).sqrt();
//     q_new = q_new.mapv(|v| v / q_new_norm);
//     let x_dot_new = x_dot + dt * x_ddot;
//     let y_dot_new = y_dot + dt * y_ddot;
//     let z_dot_new = z_dot + dt * z_ddot;
//     let wx_new = wx + dt * omega_dot[0];
//     let wy_new = wy + dt * omega_dot[1];
//     let wz_new = wz + dt * omega_dot[2];

//     let x_next = Array1::from(vec![
//         x_pos_new, y_pos_new, z_pos_new,
//         q_new[0], q_new[1], q_new[2], q_new[3],
//         x_dot_new, y_dot_new, z_dot_new,
//         wx_new, wy_new, wz_new,
//     ]);

//     x_next
// }
pub fn dynamics(x: &Array1<f64>, u: &Array1<f64>, mass: f64, moi: &Array2<f64>, dt: f64) -> Array1<f64> {
    let m = mass;
    let g = 9.81;

    // Inertia matrix and inverse
    let i = moi.clone();
    let i_inv = i.clone().inv().unwrap();

    // Unpack state
    let x_pos = x[0];
    let y_pos = x[1];
    let z_pos = x[2];
    let qx = x[3];
    let qy = x[4];
    let qz = x[5];
    let qw = x[6];
    let x_dot = x[7];
    let y_dot = x[8];
    let z_dot = x[9];
    let wx = x[10];
    let wy = x[11];
    let wz = x[12];

    // Unpack control
    let gimbal_theta = u[0];
    let gimbal_phi = u[1];
    let mut thrust = u[2];
    if thrust < 0.0 { thrust = 0.0; }

    // Thrust direction in body frame
    let tx = gimbal_theta.sin() * gimbal_phi.cos();
    let ty = gimbal_phi.sin();
    let tz = gimbal_theta.cos() * gimbal_phi.cos();
    let thrust_b = Array1::from(vec![tx * thrust, ty * thrust, tz * thrust]);

    // Quaternion normalization
    let mut q = Array1::from(vec![qx, qy, qz, qw]);
    let q_norm = q.dot(&q).sqrt();
    if q_norm < 1e-8 {
        q = Array1::from(vec![0.0, 0.0, 0.0, 1.0]);
    } else {
        q = q.mapv(|v| v / q_norm);
    }
    let qx = q[0];
    let qy = q[1];
    let qz = q[2];
    let qw = q[3];

    // Rotation matrix from quaternion
    let r = Array2::from_shape_vec((3, 3), vec![
        1.0 - 2.0 * qy * qy - 2.0 * qz * qz, 2.0 * qx * qy - 2.0 * qz * qw, 2.0 * qx * qz + 2.0 * qy * qw,
        2.0 * qx * qy + 2.0 * qz * qw, 1.0 - 2.0 * qx * qx - 2.0 * qz * qz, 2.0 * qy * qz - 2.0 * qx * qw,
        2.0 * qx * qz - 2.0 * qy * qw, 2.0 * qy * qz + 2.0 * qx * qw, 1.0 - 2.0 * qx * qx - 2.0 * qy * qy,
    ]).unwrap();

    // Acceleration in world frame
    let acc = r.dot(&thrust_b) / m - Array1::from(vec![0.0, 0.0, g]);
    let x_ddot = acc[0];
    let y_ddot = acc[1];
    let z_ddot = acc[2];

    // TVC torque
    let d = 0.663834203726; // TODO: make ts customizable
    let r_cp = Array1::from(vec![0.0, 0.0, -d]);
    let torque_b = Array1::from(vec![
        r_cp[1] * thrust_b[2] - r_cp[2] * thrust_b[1],
        r_cp[2] * thrust_b[0] - r_cp[0] * thrust_b[2],
        r_cp[0] * thrust_b[1] - r_cp[1] * thrust_b[0],
    ]);

    // --- MODIFICATION 1: Full Matrix Gyroscopic Precession ---
    let omega = Array1::from(vec![wx, wy, wz]);
    let i_omega = i.dot(&omega);
    
    let omega_cross_i_omega = Array1::from(vec![
        omega[1] * i_omega[2] - omega[2] * i_omega[1],
        omega[2] * i_omega[0] - omega[0] * i_omega[2],
        omega[0] * i_omega[1] - omega[1] * i_omega[0],
    ]);
    let omega_dot = i_inv.dot(&(torque_b - omega_cross_i_omega));

    // Linear Integration
    let x_pos_new = x_pos + dt * x_dot + 0.5 * dt * dt * x_ddot;
    let y_pos_new = y_pos + dt * y_dot + 0.5 * dt * dt * y_ddot;
    let z_pos_new = z_pos + dt * z_dot + 0.5 * dt * dt * z_ddot;
    
    let x_dot_new = x_dot + dt * x_ddot;
    let y_dot_new = y_dot + dt * y_ddot;
    let z_dot_new = z_dot + dt * z_ddot;
    
    let wx_new = wx + dt * omega_dot[0];
    let wy_new = wy + dt * omega_dot[1];
    let wz_new = wz + dt * omega_dot[2];

    // --- MODIFICATION 2: Quaternion Exponential Integration ---
    let omega_norm = (wx * wx + wy * wy + wz * wz).sqrt();
    let mut q_new = Array1::zeros(4);

    if omega_norm < 1e-8 {
        q_new = q.clone();
    } else {
        let half_theta = 0.5 * omega_norm * dt;
        let cos_half_theta = half_theta.cos();
        let sin_half_theta = half_theta.sin();
        
        let q_step = Array1::from(vec![
            (wx / omega_norm) * sin_half_theta,
            (wy / omega_norm) * sin_half_theta,
            (wz / omega_norm) * sin_half_theta,
            cos_half_theta,
        ]);

        // FIXED: Post-multiplication (q_old * q_step) applies rotation in the Local Body Frame!
        q_new[0] = q[3]*q_step[0] + q[0]*q_step[3] + q[1]*q_step[2] - q[2]*q_step[1];
        q_new[1] = q[3]*q_step[1] - q[0]*q_step[2] + q[1]*q_step[3] + q[2]*q_step[0];
        q_new[2] = q[3]*q_step[2] + q[0]*q_step[1] - q[1]*q_step[0] + q[2]*q_step[3];
        q_new[3] = q[3]*q_step[3] - q[0]*q_step[0] - q[1]*q_step[1] - q[2]*q_step[2];
    }

    let q_new_norm = q_new.dot(&q_new).sqrt();
    q_new = q_new.mapv(|v| v / q_new_norm);
    // ----------------------------------------------------------

    let x_next = Array1::from(vec![
        x_pos_new, y_pos_new, z_pos_new,
        q_new[0], q_new[1], q_new[2], q_new[3],
        x_dot_new, y_dot_new, z_dot_new,
        wx_new, wy_new, wz_new,
    ]);

    x_next
}

fn compute_jacobian(x: &Array1<f64>, u: &Array1<f64>, mass: f64, moi: &Array2<f64>, dt: f64) -> (Array2<f64>, Array2<f64>) {
    let n = x.len();
    let m = u.len();
    let mut A = Array2::<f64>::zeros((n, n));
    let mut B = Array2::<f64>::zeros((n, m));
    // compute Jacobians A and B here via central difference

    let eps = 1e-6;
    for i in 0..n {
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();
        x_plus[i] += eps;
        x_minus[i] -= eps;
        let fx_plus = dynamics(&x_plus, u, mass, moi, dt);
        let fx_minus = dynamics(&x_minus, u, mass, moi, dt);
        A.column_mut(i).assign(&((fx_plus - fx_minus) / (2.0 * eps)));
    }
    for i in 0..m {
        let mut u_plus = u.clone();
        let mut u_minus = u.clone();
        u_plus[i] += eps;
        u_minus[i] -= eps;
        let fx_plus = dynamics(x, &u_plus, mass, moi, dt);
        let fx_minus = dynamics(x, &u_minus, mass, moi, dt);
        B.column_mut(i).assign(&((fx_plus - fx_minus) / (2.0 * eps)));
    }

    (A, B)
}

fn rollout(x0: &Array1<f64>, U: &Vec<Array1<f64>>, mass: f64, moi: &Array2<f64>, dt: f64) -> Vec<Array1<f64>> {
    let mut xs = Vec::new();
    let mut x = x0.clone();
    xs.push(x.clone());
    for u in U.iter() {
        x = dynamics(&x, u, mass, moi, dt);
        xs.push(x.clone());
    }
    xs
}

fn linearize_trajectory(xs: &Vec<Array1<f64>>, U: &Vec<Array1<f64>>, mass: f64, moi: &Array2<f64>, dt: f64) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
    let mut A_seq = Vec::new();
    let mut B_seq = Vec::new();

    for k in 0..U.len() {
        let (A, B) = compute_jacobian(&xs[k], &U[k], mass, moi, dt);
        A_seq.push(A.clone());
        B_seq.push(B.clone());
    }
    (A_seq, B_seq)
}

fn build_Su(A_seq: &Vec<Array2<f64>>, B_seq: &Vec<Array2<f64>>) -> Array2<f64> {
    let n = A_seq[0].nrows();
    let m = B_seq[0].ncols();
    let N = A_seq.len();
    let mut Su = Array2::<f64>::zeros((n * N, m * N));

    for k in 0..N {
        for j in 0..=k {
            let mut Phi = Array2::<f64>::eye(n);
            if j + 1 <= k {
                for t in (j+1)..k {
                    Phi = Phi.dot(&A_seq[t]);
                }
            }
            let block = Phi.dot(&B_seq[j]);
            Su.slice_mut(s![k*n..(k+1)*n, j*m..(j+1)*m]).assign(&block);
        }
    }

    Su
}

fn assemble_qp_increment(xs: &Vec<Array1<f64>>, U: &Vec<Array1<f64>>, xref_traj: &Vec<Array1<f64>>, A_seq: &Vec<Array2<f64>>, B_seq: &Vec<Array2<f64>>, Q: &Array2<f64>, R: &Array2<f64>, QN: &Array2<f64>, u_min: &Array1<f64>, u_max: &Array1<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let N = U.len();
    let n = xs[0].len();
    let m = U[0].len();

    let mut Q_blocks: Vec<Array2<f64>> = vec![Q.clone(); N-1]; 
    Q_blocks.push(QN.clone());

    // Stack the Q blocks vertically to form the block-diagonal Q_bar
    // Q_bar = np.block([[Q_blocks[i] if i==j else np.zeros_like(Q) for j in range(N)] for i in range(N)])
    let mut Q_bar = Array2::<f64>::zeros((n*N, n*N)); // Placeholder for block-diagonal matrix
    for i in 0..N {
        for j in 0..N {
            if i == j {
                Q_bar.slice_mut(s![i*n..(i+1)*n, j*n..(j+1)*n]).assign(&Q_blocks[i]);
            } else {
                Q_bar.slice_mut(s![i*n..(i+1)*n, j*n..(j+1)*n]).assign(&Array2::<f64>::zeros((n, n)));
            }
        }
    }

    let mut R_bar = Array2::<f64>::zeros((m * N, m * N));
    for i in 0..N {
        R_bar.slice_mut(s![i*m..(i+1)*m, i*m..(i+1)*m]).assign(R);
    }

    // calculate residual r (flattened)
    //  residual r = [x1 - xref1; ... ; xN - xrefN]

    let mut r = Array1::<f64>::zeros(n * N);
    for k in 0..N {
        let x_diff = &xs[k+1] - &xref_traj[k+1];
        r.slice_mut(s![k*n..(k+1)*n]).assign(&x_diff);
    }

    let Su = build_Su(A_seq, B_seq);
    let mut H = Su.t().dot(&Q_bar).dot(&Su) + &R_bar;
    let g = Su.t().dot(&Q_bar).dot(&r);

    let mut Ustack = Array1::<f64>::zeros(m * N);
    for k in 0..N {
        Ustack.slice_mut(s![k*m..(k+1)*m]).assign(&U[k]);
    }

    let mut U_max = Array1::<f64>::zeros(m * N);
    let mut U_min = Array1::<f64>::zeros(m * N);
    for k in 0..N {
        U_max.slice_mut(s![k*m..(k+1)*m]).assign(u_max);
        U_min.slice_mut(s![k*m..(k+1)*m]).assign(u_min);
    }

    let dU_min = &U_min - &Ustack;
    let dU_max = &U_max - &Ustack;

    H = 0.5 * (H.clone() + H.t()) + 1e-8 * Array2::<f64>::eye(H.nrows()); // ensure symmetry and positive definiteness

    (H, g, dU_min, dU_max, Su, Q_bar, R_bar)
}

fn solve_qp_pgd(H: &Array2<f64>, g: &Array1<f64>, dU_min: &Array1<f64>, dU_max: &Array1<f64>, dU_init: &Array1<f64>, max_iters: usize, alpha: f64) -> Array1<f64> {
    let mut dU = dU_init.clone();
    let n = dU.len();
    
    // Initial projection to ensure feasibility
    for i in 0..n {
        dU[i] = dU[i].max(dU_min[i]).min(dU_max[i]);
    }

    for iter in 0..max_iters {
        // Compute gradient
        let grad = H.dot(&dU) + g;
        let grad_norm = grad.norm();
        
        if grad_norm < 1e-6 {
            break;
        }

        let cost_old = 0.5 * dU.t().dot(H).dot(&dU) + g.t().dot(&dU);
        
        // Try full step first
        let mut dU_new = &dU - &(alpha * &grad);
        
        // Project to bounds
        for i in 0..n {
            dU_new[i] = dU_new[i].max(dU_min[i]).min(dU_max[i]);
        }
        
        let cost_new = 0.5 * dU_new.t().dot(H).dot(&dU_new) + g.t().dot(&dU_new);
        
        // Backtracking line search if cost doesn't decrease
        let mut alpha_ls = alpha;
        let mut dU_candidate = dU_new.clone();
        
        if cost_new >= cost_old {
            alpha_ls = alpha;
            while alpha_ls > 1e-6 {
                dU_candidate = &dU - &(alpha_ls * &grad);
                
                // Project to bounds
                for i in 0..n {
                    dU_candidate[i] = dU_candidate[i].max(dU_min[i]).min(dU_max[i]);
                }
                
                let cost_candidate = 0.5 * dU_candidate.t().dot(H).dot(&dU_candidate) + g.t().dot(&dU_candidate);
                
                if cost_candidate < cost_old {
                    dU_new = dU_candidate;
                    break;
                }
                alpha_ls *= 0.5;
            }
        }

        // Update if we found a better solution
        let cost_updated = 0.5 * dU_new.t().dot(H).dot(&dU_new) + g.t().dot(&dU_new);
        if cost_updated < cost_old {
            dU = dU_new;
        } else {
            // No improvement, break early
            break;
        }
        
        // println!("Iter {}: grad_norm = {:.6}, cost = {:.6}", iter, grad_norm, cost_old);
    }

    dU
}

// TO DO: implement QP solver using active set method with BOX CONSTRAINTS. 

fn true_cost(x0: &Array1<f64>, U: &Vec<Array1<f64>>, xref_traj: &Vec<Array1<f64>>, Q: &Array2<f64>, R: &Array2<f64>, QN: &Array2<f64>, mass: f64, moi: &Array2<f64>, dt: f64) -> f64 {
    let xs = rollout(x0, U, mass, moi, dt);
    let N = U.len();
    let mut cost = 0.0;

    // add up stage costs
    for k in 0..N {
        let x_diff = &xs[k+1] - &xref_traj[k+1];
        cost += x_diff.t().dot(Q).dot(&x_diff);
        cost += U[k].t().dot(R).dot(&U[k]);
    }

    // add terminal cost
    let xN_diff = &xs[N] - &xref_traj[N];
    cost += xN_diff.t().dot(QN).dot(&xN_diff);

    cost
}

fn nmpc_step(x0: &Array1<f64>, U_init: &Vec<Array1<f64>>, xref_traj: &Vec<Array1<f64>>, Q: &Array2<f64>, R: &Array2<f64>, QN: &Array2<f64>, u_min: &Array1<f64>, u_max: &Array1<f64>, sqp_iters: usize, alpha_pgd: f64, mass: f64, moi: &Array2<f64>, dt: f64) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
    let N = U_init.len();
    let m = U_init[0].len();
    // initialize xs trajectory
    let mut xs = vec![Array1::<f64>::zeros(x0.len()); N + 1];
    let mut U = U_init.clone();
    let start_time = Instant::now();
    for _ in 0..sqp_iters {
        xs = rollout(x0, &U, mass, moi, dt);
        let (A_seq, B_seq) = linearize_trajectory(&xs, &U, mass, moi, dt);
        let (H, g, dU_min, dU_max, _Su, _Q_bar, _R_bar) = assemble_qp_increment(&xs, &U, &xref_traj, &A_seq, &B_seq, &Q, &R, &QN, &u_min, &u_max);
        let dU0 = Array1::<f64>::zeros(m * N);
        let dU = solve_qp_pgd(&H, &g, &dU_min, &dU_max, &dU0, 20, alpha_pgd);
        let mut flatU = Array1::<f64>::zeros(m * N);
        // flatten U
        // Flatten U into flatU
        for k in 0..N {
            flatU.slice_mut(s![k*m..(k+1)*m]).assign(&U[k]);
        }
        let J0 = true_cost(x0, &U, xref_traj, Q, R, QN, mass, moi, dt);
        let mut step = 1.0;
        while step > 1e-3 {
            let U_try_flat = &flatU + &(step * &dU);
            // Reshape U_try_flat into U_try (Vec<Array1<f64>>)
            let mut U_try = Vec::with_capacity(N);
            for k in 0..N {
                let mut u_k = U_try_flat.slice(s![k*m..(k+1)*m]).to_owned();
                // Project nominal U_try to bounds
                for i in 0..m {
                    u_k[i] = u_k[i].max(u_min[i]).min(u_max[i]);
                }
                U_try.push(u_k);
            }
            let Jtry = true_cost(x0, &U_try, xref_traj, Q, R, QN, mass, moi, dt);
            if Jtry < J0 {
                U = U_try;
                break;
            }
            step *= 0.5;
        }

        // test for convergence ?
        
    }
    // print time taken
    let duration = start_time.elapsed();
    println!("PGD solve frequency {:?}Hz", 1.0 / duration.as_secs_f64());

    (U, xs)
}

// Using OpEn to solve

pub fn QP_cost(H: &Array2<f64>, g: &Array1<f64>, U: &Array1<f64>) -> f64 {
    let cost = 0.5 * U.t().dot(H).dot(U) + g.t().dot(U);
    cost
}
pub fn QP_gradient(H: &Array2<f64>, g: &Array1<f64>, U_flat: &Array1<f64>) -> Array1<f64> {
    let grad = H.dot(U_flat) + g;
    grad
}

// control smoothing cost
pub fn calculate_delta_U(U: &Array1<f64>, m: usize) -> Array1<f64> {
    let N = U.len() / m; // number of control steps
    let mut delta_U = Array1::<f64>::zeros((N-1) * m);

    for k in 0..(N-1) {
        for i in 0..m {
            delta_U[k*m + i] = U[(k+1)*m + i] - U[k*m + i];
        }
    }
    
    delta_U
}

pub fn smoothing_cost(U: &Array1<f64>, weight: &Array1<f64>, m: usize) -> f64 {
    let delta_U = calculate_delta_U(U, m);
    let N_delta = delta_U.len() / m; // number of delta steps (N-1)
    
    let mut cost = 0.0;
    for k in 0..N_delta {
        for i in 0..m {
            let delta_u_ki = delta_U[k*m + i];
            cost += weight[i] * delta_u_ki * delta_u_ki;
        }
    }
    
    cost
}

pub fn smoothing_cost_grad(U: &Array1<f64>, weight: &Array1<f64>, m: usize) -> Array1<f64> {
    let N = U.len() / m; // number of control steps
    let mut grad = Array1::<f64>::zeros(U.len());

    for k in 0..(N-1) {
        for i in 0..m {
            let delta_u_ki = U[(k+1)*m + i] - U[k*m + i];
            grad[k*m + i] -= 2.0 * weight[i] * delta_u_ki;
            grad[(k+1)*m + i] += 2.0 * weight[i] * delta_u_ki;
        }
    }

    grad
}

// Fixed OpEn solver function
pub fn OpEnSolve(
    x0: &Array1<f64>,
    U: &Vec<Array1<f64>>,
    xref_traj: &Vec<Array1<f64>>,
    Q: &Array2<f64>,
    R: &Array2<f64>,
    QN: &Array2<f64>,
    smoothing_weight: &Array1<f64>,
    panoc_cache: &mut PANOCCache,
    mass: f64,
    moi: &Array2<f64>,
    min_thrust: f64,
    max_thrust: f64,
    gimbal_limit_rad: f64,
    dt: f64,
) -> (Array1<f64>, Array2<f64>, MPCDebugInfo) {
    use ndarray::ArrayView1;
    let mut debug_info = MPCDebugInfo::new();

    let N = U.len();
    let m = U[0].len();
    let n_dim_u = N * m;

    // Build nominal flat U for bound shifting
    let mut u_nom_flat: Vec<f64> = Vec::with_capacity(n_dim_u);
    for k in 0..N {
        u_nom_flat.extend_from_slice(U[k].as_slice().unwrap());
    }
    let u_nom = Array1::from(u_nom_flat.clone());

    // Linearize about current rollout, assemble QP in delta-U space
    let xs = rollout(x0, U, mass, moi, dt);
    let (A_seq, B_seq) = linearize_trajectory(&xs, U, mass, moi, dt);
    let (H, g, _dU_min, _dU_max, _Su, _Q_bar, _R_bar) =
        assemble_qp_increment(&xs, U, xref_traj, &A_seq, &B_seq, Q, R, QN,
                              &Array1::zeros(m), &Array1::zeros(m));

    // Gradient and cost are in delta-U space: min 0.5 dU^T H dU + g^T dU
    let df = |du_slice: &[f64], grad_slice: &mut [f64]| -> Result<(), SolverError> {
        let du = Array1::from(du_slice.to_vec());
        let grad = QP_gradient(&H, &g, &du);
        let smooth_grad = smoothing_cost_grad(&du, smoothing_weight, m);
        let total_grad = &grad + &smooth_grad;
        grad_slice.copy_from_slice(total_grad.as_slice().unwrap());
        Ok(())
    };

    let f = |du_slice: &[f64], cost: &mut f64| -> Result<(), SolverError> {
        let du = Array1::from(du_slice.to_vec());
        let smooth_cost = smoothing_cost(&du, smoothing_weight, m);
        *cost = QP_cost(&H, &g, &du) + smooth_cost;
        Ok(())
    };

    // Box bounds shifted into delta-U space: dU_min = u_min - u_nom, dU_max = u_max - u_nom
    let mut du_min_flat: Vec<f64> = Vec::with_capacity(n_dim_u);
    let mut du_max_flat: Vec<f64> = Vec::with_capacity(n_dim_u);
    for k in 0..N {
        let u_nom_k = &U[k];
        du_min_flat.push(-gimbal_limit_rad - u_nom_k[0]);
        du_max_flat.push( gimbal_limit_rad - u_nom_k[0]);
        du_min_flat.push(-gimbal_limit_rad - u_nom_k[1]);
        du_max_flat.push( gimbal_limit_rad - u_nom_k[1]);
        du_min_flat.push(min_thrust - u_nom_k[2]);
        du_max_flat.push(max_thrust - u_nom_k[2]);
    }
    let bounds = constraints::Rectangle::new(
        Some(&du_min_flat[..]),
        Some(&du_max_flat[..]),
    );

    let problem = Problem::new(&bounds, df, f);
    let mut panoc = PANOCOptimizer::new(problem, panoc_cache).with_max_iter(150);

    // Initialize at zero — meaning "start from the warm-start U, apply no delta"
    let mut du_flat: Vec<f64> = vec![0.0; n_dim_u];
    let status = panoc.solve(&mut du_flat).unwrap();
    debug_info.iterations = status.iterations();

    // Recover absolute U: U_new = U_warm + dU, clamped to hard bounds
    let du_seq = Array1::from(du_flat);
    let u_seq = &u_nom + &du_seq;

    // Clamp final result to absolute bounds (safety)
    let u_seq_clamped: Vec<f64> = (0..N).flat_map(|k| {
        let base = k * m;
        vec![
            u_seq[base + 0].clamp(-gimbal_limit_rad, gimbal_limit_rad),
            u_seq[base + 1].clamp(-gimbal_limit_rad, gimbal_limit_rad),
            u_seq[base + 2].clamp(min_thrust, max_thrust),
        ]
    }).collect();

    let u_seq_arr = Array1::from(u_seq_clamped);
    let u_apply = u_seq_arr.slice(s![0..m]).to_owned();
    let U_seq = Array2::from_shape_vec((N, m), u_seq_arr.to_vec()).unwrap();

    (u_apply, U_seq, debug_info)
}

// Main MPC function to be called externally
pub fn mpc_main(x: &Array1<f64>, U_warm: &mut Vec<Array1<f64>>, xref_traj: &Vec<Array1<f64>>, Q: &Array2<f64>, R: &Array2<f64>, QN: &Array2<f64>, u_min: &Array1<f64>, u_max: &Array1<f64>, sqp_iters: usize, alpha_pgd: f64, mass: f64, moi: &Array2<f64>, dt: f64) -> (Vec<Array1<f64>>, Array1<f64>, Array1<f64>) {
    let (U, _xs) = nmpc_step(x, &U_warm, &xref_traj, &Q, &R, &QN, &u_min, &u_max, sqp_iters, alpha_pgd, mass, moi, dt);
    let u_apply = U[0].clone(); // apply first control

    let x_new = dynamics(x, &u_apply, mass, moi, dt);
    *U_warm = U[1..].to_vec(); // shift warm start
    U_warm.push(U_warm.last().unwrap().clone()); // repeat last control for warm start

    // return the next warm start sequence, the new state, and the applied control.
    // Note that x_new will not be used in the real implementation. Filtered sensor data will provide the next state.
    (U_warm.clone(), x_new, u_apply)
}
