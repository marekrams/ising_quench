import csv
import os
from pathlib import Path
import time
from tqdm import tqdm  # progressbar
import yastn
import yastn.tn.fpeps as peps
import numpy as np
import scipy
import ray
import glob
from itertools import product


def mean(x):
    return sum(x) / len(x)



def gate_Ising_cluster(Jxx, g, step, ops, net):
    Z = ops.z()
    X = ops.x()
    I = ops.I()
    Hl = -g * Z
    Hnn = -Jxx * peps.gates.fkron(X, X)
    Gnn = peps.gates.gate_nn_exp(step, I, Hnn)
    Gl = peps.gates.gate_local_exp(step, I, Hl)
    return peps.gates.distribute(net, gates_local=Gl, gates_nn=Gnn)


# @ray.remote(num_cpus=1)
# def run_quench(g, D, chi, which='NN+BP', dt=0.01, sym='Z2'):
#     #
#     geometry = peps.CheckerboardLattice()
#     #
#     # Define quench protocol
#     #
#     # Load operators.
#     ops = yastn.operators.Spin12(sym=sym)
#     #
#     Z = ops.z()
#     X = ops.x()
#     vec_z1 = ops.vec_z(val=1)
#     #
#     path = Path(f"./quench/{g=:0.4f}/{which=:s}/{sym=:s}")
#     path.mkdir(parents=True, exist_ok=True)
#     print(path)
#     #
#     fname = path / f"env_{D=:d}_{chi=:d}_{dt=:0.2f}.npy"
#     try:
#         data = np.load(tmp, allow_pickle=True).item()
#         env_ctm = yastn.from_dict(data['env'])
#         psi = env_ctm.psi.ket
#         si = float(tmp.split("_")[-2][3:9])
#         #
#         fname = fname.replace('env', 'ev')
#         data = np.load(tmp, allow_pickle=True).item()
#         Eng = data['Eng']
#         infoss = data['infoss'].copy()

#     except FileNotFoundError:
#         # Initialize system in the product ground state at s=0.
#         si = 0
#         psi = peps.product_peps(geometry=geometry, vectors=vec_z1)
#         env_ctm = peps.EnvCTM(psi, init='eye')
#         Eng = -hz
#         infoss = []
#     #
#     # simulation parameters
#     opts_svd_ntu = {"D_total": D, "D_block": D // 2, 'tol': 1e-12} if sym == 'Z2' else {"D_total": D, 'tol': 1e-12}
#     #
#     if 'BP' in which:
#         env = peps.EnvBP(psi, which=which)
#         env.iterate_(max_sweeps=100, diff_tol=1e-8)
#     else:
#         env = peps.EnvNTU(psi, which=which)
#     #
#     t = 0
#     #
#     ss = np.arange(0.05, 6.01, 0.05)
#     ss = [s for s in ss if s > si + 0.0001]
#     #
#     dtold = dt
#     for sf in ss:
#         print(sf)
#         ds = sf - si
#         si = sf
#         steps = round(np.ceil((ds - 0.0001) / dtold))
#         dt = ds / steps
#         dt2 = 1j * dt / 2
#         gates = gate_Ising_cluster(1, hz, dt2, ops, geometry)
#         #
#         for _ in range(steps):
#             t += dt / 2
#             infos = peps.evolution_step_(env, gates, opts_svd=opts_svd_ntu)
#             if 'BP' in which:
#                 env.iterate_(max_sweeps=100, diff_tol=1e-8)
#             infoss.append(infos)
#             t += dt / 2

#         Delta = peps.accumulated_truncation_error(infoss, statistics='mean')
#         #
#         opts_svd_env = {'D_total': chi}
#         env_ctm = peps.EnvCTM(psi, init='eye')
#         info = env_ctm.ctmrg_(opts_svd=opts_svd_env, max_sweeps=100, corner_tol=1e-5)
#         if info.converged is False:
#             info = env_ctm.ctmrg_(opts_svd=opts_svd_env, max_sweeps=100, corner_tol=1e-5)
#         #
#         # Calculating 1-site <X_i> for all sites
#         #
#         Ex = mean(env_ctm.measure_1site(X).values()).real
#         #
#         Ez = mean(env_ctm.measure_1site(Z).values()).real
#         #
#         Exx = mean(env_ctm.measure_nn(X, X).values()).real
#         #
#         Eng = -hz * Ez - 2 * Exx
#         #
#         if abs(Eng + hz) > 0.1:
#             return None
#         #
#         Exxs = env_ctm.measure_2site(X, X, xrange=(0, 20), yrange=(0, 1), pairs='corner <=', dirn='v')
#         Exxs = dict(sorted(Exxs.items()))
#         Exxs = np.array(list(Exxs.values())).real
#         #

#         fname = path / f"ev_{D=:d}_{chi=:d}_{sf=:0.4f}_{dt=:0.2f}.npy"
#         data = {'Delta': Delta,
#                 'which': which,
#                 'sym': sym,
#                 'hz': hz,
#                 'dt': dtold,
#                 'D': D,
#                 'chi': chi,
#                 'sf': sf,
#                 'ctm_info': info,
#                 'infoss': infoss,
#                 'Eng': Eng,
#                 'Exxs': Exxs,
#                 'Exx': Exx,
#                 'Ex': Ex,
#                 'Ez': Ez,
#                 'eat_metric_error': max([0] + [x.eat_metric_error for x in infoss[-1] if x.eat_metric_error])
#                 }
#         np.save(fname, data, allow_pickle=True)

#         fname = path / f"env_{D=:d}_{chi=:d}_{dt=:0.2f}.npy"
#         data = {'D': D,
#                 'chi': chi,
#                 'sf': sf,
#                 'env': env_ctm.to_dict()
#                 }
#         np.save(fname, data, allow_pickle=True)
#
#
@ray.remote(num_cpus=1)
def run_quench_new(g, D, chi, which='NN+BP', dt=0.001, sym='Z2'):
    #
    geometry = peps.CheckerboardLattice()
    #
    # Define quench protocol
    #
    # Load operators.
    ops = yastn.operators.Spin12(sym=sym)
    #
    Z = ops.z()
    X = ops.x()
    vec_z1 = ops.vec_z(val=1)
    #
    path = Path(f"./data/{g=:0.4f}/{which=:s}/{sym=:s}")
    path.mkdir(parents=True, exist_ok=True)
    #
    try:
        fname_env = path / f"env_{D=:d}_{chi=:d}_{dt=:0.4f}.npy"
        data_env = np.load(fname_env, allow_pickle=True).item()
        print("Found", fname_env)
        env_ctm = yastn.from_dict(data_env['env'])
        psi = env_ctm.psi.ket
        si = data_env['sf']
        infoss = data_env['infoss'].copy()
        #
        fname_res = path / f"res_{D=:d}_{chi=:d}_{dt=:0.4f}.npy"
        datas = np.load(fname_res, allow_pickle=True).item()

        if abs(datas[si]['Eng'] + g) > 0.1:
            return None

    except FileNotFoundError:
        # Initialize system in the product ground state at s=0.
        print("NOT Found", fname_env)
        psi = peps.product_peps(geometry=geometry, vectors=vec_z1)
        env_ctm = peps.EnvCTM(psi, init='eye')
        si = 0
        infoss = []
        datas = {'which': which,
                 'sym': sym,
                 'g': g,
                 'dt': dt,
                 'D': D,
                 'chi': chi,
                }
    #
    # simulation parameters
    opts_svd_ntu = {"D_total": D, "D_block": D // 2, 'tol': 1e-12} if sym == 'Z2' else {"D_total": D, 'tol': 1e-12}
    #
    if 'BP' in which:
        env = peps.EnvBP(psi, which=which)
        env.iterate_(max_sweeps=100, diff_tol=1e-8)
    else:
        env = peps.EnvNTU(psi, which=which)
    #
    t = 0
    #
    ss = np.arange(0.01, 8.01, 0.01)
    ss = [s for s in ss if s > si + 0.0001]
    #
    dtold = dt
    for sf in ss:
        ds = sf - si
        si = sf
        steps = round(np.ceil((ds - 0.0001) / dtold))
        dt = ds / steps
        idt2 = 1j * dt / 2
        gates = gate_Ising_cluster(1, g, idt2, ops, geometry)
        #
        for _ in range(steps):
            t += dt / 2
            infos = peps.evolution_step_(env, gates, opts_svd=opts_svd_ntu)
            if 'BP' in which:
                env.iterate_(max_sweeps=100, diff_tol=1e-8)
            infoss.append(infos)
            t += dt / 2
        #
        Delta = peps.accumulated_truncation_error(infoss, statistics='mean')
        #
        opts_svd_env = {'D_total': chi}
        try:
            info = env_ctm.ctmrg_(opts_svd=opts_svd_env, max_sweeps=200, corner_tol=1e-5)
        except yastn.YastnError:
            env_ctm = peps.EnvCTM(psi, init='eye')
            info = env_ctm.ctmrg_(opts_svd=opts_svd_env, max_sweeps=200, corner_tol=1e-5)
        #
        # Calculating 1-site <X_i> for all sites
        #
        Ex = mean(env_ctm.measure_1site(X).values()).real
        Ez = mean(env_ctm.measure_1site(Z).values()).real
        Exx = mean(env_ctm.measure_nn(X, X).values()).real
        Eng = -g * Ez - 2 * Exx
        #
        Exxs = env_ctm.measure_2site(X, X, xrange=(0, 20), yrange=(0, 1), pairs='corner <=', dirn='v')
        Exxs = dict(sorted(Exxs.items()))
        Exxs = np.array(list(Exxs.values())).real
        #
        fname_res = path / f"res_{D=:d}_{chi=:d}_{dt=:0.4f}.npy"
        iii  = sum(infoss[-steps:], start=[])
        datas[sf] = {'Delta': Delta,
                     'ctm_info': info,
                     'Eng': Eng,
                     'Exxs': Exxs,
                     'Exx': Exx,
                     'Ex': Ex,
                     'Ez': Ez,
                     'eat_metric_error': max([0] + [x.eat_metric_error for x in iii if x.eat_metric_error]),
                     'wrong_eigenvalues': max([0] + [x.wrong_eigenvalues for x in iii if x.wrong_eigenvalues]),
                    }
        np.save(fname_res, datas, allow_pickle=True)

        fname_env = path / f"env_{D=:d}_{chi=:d}_{dt=:0.4f}.npy"
        data_env = {'which': which,
                'sym': sym,
                'g': g,
                'dt': dt,
                'D': D,
                'chi': chi,
                'sf': sf,
                'env': env_ctm.to_dict(),
                'infoss': infoss
                }
        np.save(fname_env, data_env, allow_pickle=True)

        if abs(Eng + g) > 0.1:
            return None



if __name__ == '__main__':
    ray.init()
    refs = []
    dt = 0.01
    gc = 3.04438

    gs = [gc / 10, gc, 2 * gc]
    for D in [10]:
        chi = 4 * D
        for g in gs:
            for which in ["NN", "NN+", "BP", "NN+BP", "Ladder+BP"]:
                for sym in ['dense', 'Z2']:
                    job = run_quench_new.remote(g, D, chi, which, dt, sym)
                    refs.append(job)
    ray.get(refs)
