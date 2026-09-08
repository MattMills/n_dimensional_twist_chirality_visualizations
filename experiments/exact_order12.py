"""Experiment F12: the exact reformulation with 12th roots of unity.

Same construction as experiment F (order 4) but phases are powers of
zeta_12, so the lattice spacing is pi/6, beat envelopes are three times wider
and the exact phase group has twelve elements.  Real quantities live in
Z[sqrt 3]; results are rendered as exactly truncated decimals of elements of
Q(sqrt 3).  Order 4 is recomputed on matched point sets for comparison.
"""
import json
import os
from fractions import Fraction

from common import RES_DIR, Timer

import numpy as np

from twistchiral import exact as ex4
from twistchiral import exact12 as ex


def dec(x, digits: int = 4) -> str:
    if x is None:
        return "–"
    if isinstance(x, ex.R3):
        return x.decimal(digits) + "…" if x.b != 0 or (isinstance(x.a, Fraction) and x.a.denominator != 1) else x.decimal(digits)
    if isinstance(x, Fraction):
        if x.denominator == 1:
            return str(x.numerator)
        q, r = divmod(x.numerator, x.denominator)
        return f"{q}.{(r * 10**digits) // x.denominator:0{digits}d}…"
    return str(x)


def pair_system(n: int, module, order: int, sidebands: int = 3):
    K1 = [1, 0, 2, 0, 1, 0, 1, 0][:n]
    side = []
    for j in range(min(sidebands, n)):
        off = [0] * n
        off[j] = 1 if j != 2 else -1
        side.append(off)
    if sidebands > n:
        for j in range(sidebands - n):
            off = [0] * n
            off[j % n] = -2 if j % 2 == 0 else 2
            side.append(off)
    kw = {"order": order} if module is ex else {}
    p1 = module.Packet.carrier_with_sidebands(K1, side, (2, 0), (1, 1), center=[0] * n, **kw)
    R = module.plane_rotation_90(n, 0, 1)
    if n >= 4:
        R2 = module.plane_rotation_90(n, 2, 3)
        R = np.array([[sum(int(R2[i][k]) * int(R[k][j]) for k in range(n)) for j in range(n)] for i in range(n)], dtype=object)
    scale = order // 4
    mu2 = [scale * v for v in [1, 0, 3, 1, 0, 1, 1, 0][:n]]
    p2 = p1.twisted(R, center=mu2)
    if module is ex:
        S = ex.LatticeSystem([p1, p2]).with_phase(1, 1)
    else:
        S = ex4.LatticeSystem([p1, p2.with_phase(1)])
    return S, K1


def subsample(idx, cap):
    idx = list(idx)
    if len(idx) <= cap:
        return idx
    step = len(idx) // cap
    return idx[::step][:cap]


def two_packet_stats(n: int, module, order: int, P: int, sidebands: int = 3, full_torus: bool = False):
    S, K1 = pair_system(n, module, order, sidebands)
    M = module.torus_points(n, order) if full_torus else module.lcg_points(n, P, seed=100 * n + order + sidebands, period=order)
    Psi, T = S.evaluate(M)
    g = module.exact_geometry(Psi, T)
    pf = module.pair_factors(Psi, T)
    ea = module.RingExterior(n) if module is ex else module.IntExterior(n)
    rho1, rho2 = pf["rho_a"]
    band = module.visibility_band(rho1, rho2, Fraction(1, 2))
    cf = module.closed_form_identity(g, pf, ea)
    fi = module.factor_identity(g, pf, ea)
    C = ea.wedge(g["A"], 1, ea.from_antisymmetric(g["F"]), 2)
    nz = (lambda row: module.nonzero(row)) if module is ex else (lambda row: any(int(v) != 0 for v in row))
    band_idx = [p for p in range(M.shape[0]) if band[p] and nz(C[p])]
    stats = module.direction_statistics(C[subsample(band_idx, 260)]) if band_idx else {"effective_dimension": None, "population_effective_dimension": None}
    mir = module.mirror_identity(ea, C[band_idx[:30]], 3) if band_idx else None
    ranks = sorted({module.bareiss_rank(g["F"][p].tolist()) for p in range(M.shape[0])})
    out = {"points": int(M.shape[0]), "band": int(band.sum()), "closed_form": f"{int(cf.sum())}/{M.shape[0]}",
           "factor": bool(fi.all()), "ranks": ranks,
           "d_eff": dec(stats["effective_dimension"]), "d_pop": dec(stats["population_effective_dimension"]),
           "mirror_ok": bool(mir["identity"] and mir["mean_cosine"] == mir["predicted"]) if mir else None,
           "mirror_pred": dec(mir["predicted"]) if mir else None}
    if module is ex:
        ph = module.phase_invariance(S, M[:10], ea)
        out["phase_invariant"] = bool(ph["F_invariant"] and ph["chirality_invariant"] and ph["intensity_changes"])
        sat = module.saturation_design_average(K1, n)
        out["saturation_ok"] = bool(sat["average"] == sat["predicted"])
    return out


def run() -> dict:
    res = {}
    lines = ["# Exact reformulation with 12th roots of unity — results", "",
             "Phases are powers of ζ₁₂ = e^{iπ/6}; every real quantity is an element a + b√3 of ℤ[√3] (integer pairs), "
             "with rationals only in the final ratios. Decimals are exactly truncated expansions of elements of ℚ(√3); "
             "nothing was computed in floating point.", ""]

    with Timer("F12-1 two packets: order 12 versus order 4 on matched point sets"):
        lines += ["## F12-1. Two twisted packets: spread of chirality directions, order 12 vs order 4", "",
                  "| n | side-bands | order | points | band | closed form | F~=2x∧y | phase-inv. | ranks | d_eff | d_pop | mirror = 1−6/n | saturation |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        res["F1"] = []
        for n in (3, 4, 5, 6):
            for sb in (3, 6):
                for module, order in ((ex4, 4), (ex, 12)):
                    P = 600 if n > 3 else 0
                    r = two_packet_stats(n, module, order, P, sb, full_torus=(n == 3))
                    r.update({"n": n, "sidebands": sb, "order": order})
                    res["F1"].append(r)
                    lines.append(f"| {n} | {sb} | {order} | {r['points']} | {r['band']} | {r['closed_form']} | {'yes' if r['factor'] else 'NO'} | "
                                 f"{'yes' if r.get('phase_invariant', True) else 'NO'} | {r['ranks']} | {r['d_eff']} | {r['d_pop']} | "
                                 f"{'yes' if r['mirror_ok'] else 'NO'} ({r['mirror_pred']}) | {'yes' if r.get('saturation_ok', True) else 'NO'} |")
                    print(f"  n={n} sidebands={sb} order={order}: P={r['points']} band={r['band']} closed={r['closed_form']} d_pop={r['d_pop']}")
        lines.append("")

    with Timer("F12-2 N packets at order 12: ladder, additivity, junction cubes"):
        ns = [4, 5, 6, 7, 8]
        Ns = [2, 3, 4, 5, 6]
        table = {}
        add_ok = True
        lines += ["## F12-2. N packets (order 12): highest non-zero chirality degree versus min(n, 2N − 1)", "",
                  "| n \\ N | " + " | ".join(str(N) for N in Ns) + " |", "|---|" + "---|" * len(Ns)]
        for n in ns:
            row = []
            for N in Ns:
                packets = []
                for a in range(N):
                    carrier = [0] * n
                    carrier[a % n] = 1
                    carrier[(a + 2) % n] += 1 if a % 2 == 0 else 2
                    side = []
                    for j in ((a + 1) % n, (a + 3) % n):
                        off = [0] * n
                        off[j] = 1
                        side.append(off)
                    center = [(3 * a * (j + 1) + j * j) % 12 for j in range(n)]
                    amp = [(2, 0), (1, 1), (1, -1), (0, 1), (2, 1), (1, 2)][a]
                    packets.append(ex.Packet.carrier_with_sidebands(carrier, side, amp, (1, 0), center=center))
                S = ex.LatticeSystem(packets)
                M = ex.lcg_points(n, 24, seed=10 * n + N)
                g = ex.exact_geometry(*S.evaluate(M))
                lad = ex.exact_ladder(ex.RingExterior(n), g["A"], g["F"])
                top = max((p for p, c in lad.items() if ex.nonzero(c)), default=0)
                rank = max(ex.bareiss_rank(g["F"][p].tolist()) for p in range(M.shape[0]))
                if N >= 3:
                    add_ok &= bool(ex.pairwise_identity(S, M[:6]).all())
                row.append((top, rank))
            table[n] = row
            lines.append(f"| {n} | " + " | ".join(f"{t}{'★' if t == n else ''} (rank {r})" for t, r in row) + " |")
            print(f"  n={n}: " + ", ".join(f"N={N}: {t}/{r}" for N, (t, r) in zip(Ns, row)))
        pred_ok = all(table[n][j][0] == min(n, 2 * N - 1) for n in ns for j, N in enumerate(Ns))
        res["F2"] = {"top_degree_and_rank": {str(n): table[n] for n in ns}, "all_match_prediction": pred_ok, "pairwise_additivity_exact": add_ok}
        lines += ["", f"Every entry equals min(n, 2N − 1): **{'yes' if pred_ok else 'NO'}**. Pairwise additivity F~ = Σ F~_ab exact: **{'yes' if add_ok else 'NO'}**.", ""]
        jl = []
        for n in (3, 4):
            for N in (3, 4):
                packets = []
                for a in range(N):
                    carrier = [0] * n
                    carrier[a % n] = 1
                    side = [[1 if j == (a + 1) % n else 0 for j in range(n)]]
                    center = [(6 * a + 3 * j) % 12 for j in range(n)]
                    packets.append(ex.Packet.carrier_with_sidebands(carrier, side, (2, 0), (1, 0), center=center))
                S = ex.LatticeSystem(packets)
                M = ex.torus_points(n, 12)
                Psi, _ = S.evaluate(M)
                rho = np.stack([ex.gabs2((Psi[0][:, a], Psi[1][:, a])) for a in range(N)], axis=1)
                dom = ex.dominance(rho)
                labels = {tuple(int(v) for v in M[p]): int(dom[p]) for p in range(M.shape[0])}
                jl.append((n, N, [int((dom == a).sum()) for a in range(N)], ex.junction_cubes(labels, n, 12, 3),
                           ex.junction_cubes(labels, n, 12, 4) if N >= 4 else 0))
        res["F2"]["junctions"] = jl
        lines += ["Dominance cells and junction cubes on the full torus (ℤ/12)ⁿ:", "",
                  "| n | N | cell sizes | ≥3-junction cubes | ≥4-junction cubes |", "|---|---|---|---|---|"]
        for n, N, sizes, c3, c4 in jl:
            lines.append(f"| {n} | {N} | {sizes} | {c3} | {c4} |")
        lines.append("")

    with Timer("F12-3 two packets at high n (Gram determinants over Z[sqrt3])"):
        lines += ["## F12-3. Two packets in high dimension (order 12), component-free", "",
                  "| n | points | closed form | rank 2 | d_eff | d_pop | mirror mean = 1−6/n |", "|---|---|---|---|---|---|---|"]
        res["F3"] = {}
        for n in (16, 32, 64, 128, 256):
            K1 = [0] * n
            K1[0], K1[2], K1[4] = 1, 2, 1
            side = []
            for j in (1, 3, 5, 6, 7):
                off = [0] * n
                off[j] = 1
                side.append(off)
            p1 = ex.Packet.carrier_with_sidebands(K1, side, (2, 0), (1, 1), center=[0] * n)
            RR = ex.compose(ex.plane_rotation_90(n, 2, 3), ex.plane_rotation_90(n, 0, 1))
            mu2 = [(j * j + 1) % 7 for j in range(n)]
            p2 = p1.twisted(RR, center=mu2)
            S = ex.LatticeSystem([p1, p2]).with_phase(1, 5)
            P = 60 if n <= 64 else 40
            M = ex.lcg_points(n, P, seed=n)
            Psi, T = S.evaluate(M)
            pf = ex.pair_factors(Psi, T)
            Atil = pf["u"][0] + pf["u"][1]
            geo = {"rho": pf["rho_a"][0] + pf["rho_a"][1], "A": Atil}
            cf = ex.closed_form_identity_highn(geo, pf)
            rank2 = all(not ex.gram_det([pf["x"][p], pf["y"][p]]).is_zero() for p in range(P))
            triples = [(Atil[p], pf["x"][p], pf["y"][p]) for p in range(P)]
            st = ex.decomposable_direction_statistics(triples)
            mir = ex.decomposable_mirror_mean(triples, n)
            res["F3"][n] = {"points": P, "closed_form": f"{int(cf.sum())}/{P}", "rank2": rank2,
                            "d_eff": dec(st["effective_dimension"]), "d_pop": dec(st["population_effective_dimension"]),
                            "mirror_ok": mir["all_equal_predicted"]}
            lines.append(f"| {n} | {P} | {int(cf.sum())}/{P} | {'yes' if rank2 else 'NO'} | {dec(st['effective_dimension'])} | "
                         f"{dec(st['population_effective_dimension'])} | {'yes' if mir['all_equal_predicted'] else 'NO'} ({dec(mir['predicted'])}) |")
            print(f"  n={n}: closed {int(cf.sum())}/{P} d_pop={dec(st['population_effective_dimension'])} mirror={mir['all_equal_predicted']}")
        lines.append("")

    lines += ["## Notes", "",
              "* ℚ(ζ₁₂) = ℚ(i, √3): the real subfield is ℚ(√3), so exact real quantities are integer pairs a + b√3 (after scaling every phase to 2ζ, which is a harmless global factor). Only orders 3, 4 and 6 have a rational real subfield.",
              "* Ranks use fraction-free Bareiss elimination in ℤ[√3] (exact division); statistics are elements of ℚ(√3) rendered as exactly truncated decimals.",
              "* Order 4 rows recompute experiment F on matched point counts; order 12 rows use the same packets with three-times-wider beat envelopes and a twelve-element phase group."]
    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, "F12_exact_order12.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(RES_DIR, "F12_exact_order12.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print("  wrote results/F12_exact_order12.md and .json")
    return res


if __name__ == "__main__":
    run()
