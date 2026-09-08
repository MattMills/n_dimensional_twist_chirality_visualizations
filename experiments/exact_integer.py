"""Experiment F: the whole story in exact integer arithmetic (no floats).

Positions are lattice points x = (pi/2) m, wavevectors are integer vectors,
phases are powers of i, packets are carriers with side-bands (their beat
pattern is the envelope), twists are signed permutations.  Every quantity is
a Python integer or an exact rational, and every identity is checked by
exact equality at every point examined.
"""
import json
import os
import time
from fractions import Fraction
from itertools import product

from common import RES_DIR, Timer

import numpy as np

from twistchiral import exact as ex


def frac(x, digits: int = 4) -> str:
    """Render an exact rational as an integer part plus a truncated decimal
    computed by integer division (the full fraction is kept in the JSON)."""
    if x is None:
        return "–"
    if isinstance(x, Fraction):
        if x.denominator == 1:
            return str(x.numerator)
        sign = "-" if x < 0 else ""
        x = abs(x)
        q, r = divmod(x.numerator, x.denominator)
        tail = (r * 10**digits) // x.denominator
        return f"{sign}{q}.{tail:0{digits}d}…"
    return str(x)


def pair_system(n: int):
    K1 = [1, 0, 2, 0, 1, 0, 1, 0][:n]
    side = []
    for j in range(min(3, n)):
        off = [0] * n
        off[j] = 1 if j != 2 else -1
        side.append(off)
    p1 = ex.Packet.carrier_with_sidebands(K1, side, (2, 0), (1, 1), center=[0] * n, )
    R = ex.plane_rotation_90(n, 0, 1)
    if n >= 4:
        R2 = ex.plane_rotation_90(n, 2, 3)
        R = np.array([[sum(int(R2[i][k]) * int(R[k][j]) for k in range(n)) for j in range(n)] for i in range(n)], dtype=object)
    mu2 = [1, 0, 3, 1, 0, 1, 1, 0][:n]      # chosen so that the carrier screw (K1^K2)^(mu2-mu1) is non-zero in n = 3
    p2 = p1.twisted(R, center=mu2).with_phase(1)
    return ex.LatticeSystem([p1, p2]), K1, R, mu2


def subsample(idx, cap):
    idx = list(idx)
    if len(idx) <= cap:
        return idx
    step = len(idx) // cap
    return idx[::step][:cap]


def run() -> dict:
    res = {}
    lines = ["# Exact integer reformulation — results", "",
             "Every number below is an exact integer or rational; nothing was computed in floating point.", ""]

    # ------------------------------------------------------------------ F1
    with Timer("F1 two packets on the full torus, n = 3..6"):
        lines += ["## F1. Two twisted packets on the torus (Z/4)^n", "",
                  "| n | points | ρ>0 | dominant 1 / 2 | exact ties | band V²≥½ | closed form exact | F~ = 2x∧y | phase-invariant | ranks of F~ | d_eff (band) | d_pop (band) | mean mirror cos | 1−6/n | saturation avg | 1−3/n+2/n² |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        res["F1"] = {}
        for n in (3, 4, 5, 6):
            S, K1, R, mu2 = pair_system(n)
            M = ex.torus_points(n)
            Psi, T = S.evaluate(M)
            g = ex.exact_geometry(Psi, T)
            pf = ex.pair_factors(Psi, T)
            ea = ex.IntExterior(n)
            rho1, rho2 = pf["rho_a"]
            P = M.shape[0]
            pos = int(sum(1 for r in g["rho"] if int(r) > 0))
            dom = ex.dominance(np.stack([rho1, rho2], axis=1))
            ties = int(sum(1 for a, b in zip(rho1, rho2) if int(a) == int(b) and int(a) > 0))
            band = ex.visibility_band(rho1, rho2, Fraction(1, 2))
            cf = ex.closed_form_identity(g, pf, ea)
            fi = ex.factor_identity(g, pf, ea)
            ph = ex.phase_invariance(S, M[: min(P, 128)], ea)
            ranks = sorted({ex.bareiss_rank(g["F"][p].tolist()) for p in range(P)})
            C = ea.wedge(g["A"], 1, ea.from_antisymmetric(g["F"]), 2)
            band_idx = [p for p in range(P) if band[p] and any(int(v) != 0 for v in C[p])]
            stats = ex.direction_statistics(C[subsample(band_idx, 300)]) if band_idx else {"effective_dimension": None, "population_effective_dimension": None}
            mir = ex.mirror_identity(ea, C[band_idx[:50]], 3) if band_idx else {"identity": True, "mean_cosine": None, "predicted": 1 - Fraction(6, n)}
            sat = ex.saturation_design_average(K1, n)
            row = {"points": P, "rho_positive": pos, "dominant_counts": [int((dom == 0).sum()), int((dom == 1).sum())],
                   "exact_ties": ties, "band": int(band.sum()), "closed_form_all": bool(cf.all()), "closed_form_count": int(cf.sum()),
                   "factor_all": bool(fi.all()), "phase": ph, "ranks": ranks,
                   "d_eff": str(stats["effective_dimension"]), "d_pop": str(stats["population_effective_dimension"]),
                   "mirror_identity": mir["identity"], "mirror_mean": str(mir["mean_cosine"]), "mirror_pred": str(mir["predicted"]),
                   "saturation": str(sat["average"]), "saturation_pred": str(sat["predicted"])}
            if n == 3:
                signs = [int(C[p][0]) for p in band_idx]
                screw = ea.wedge_vectors(ex.ints(np.array([K1], dtype=object)),
                                         ex.ints(np.array([[sum(int(R[i][j]) * K1[j] for j in range(n)) for i in range(n)]], dtype=object)),
                                         ex.ints(np.array([mu2], dtype=object)))[0][0]
                row["sign_counts_band"] = {"positive": sum(1 for s in signs if s > 0), "negative": sum(1 for s in signs if s < 0),
                                           "zero": sum(1 for s in signs if s == 0)}
                row["screw_chirality_(K1^K2)^(mu2-mu1)"] = int(screw)
            res["F1"][n] = row
            lines.append(f"| {n} | {P} | {pos} | {row['dominant_counts'][0]} / {row['dominant_counts'][1]} | {ties} | {row['band']} | "
                         f"{int(cf.sum())}/{P} | {'yes' if fi.all() else 'NO'} | {'yes' if ph['F_invariant'] and ph['chirality_invariant'] else 'NO'} | {ranks} | "
                         f"{frac(stats['effective_dimension'])} | {frac(stats['population_effective_dimension'])} | {frac(mir['mean_cosine'])} | {frac(mir['predicted'])} | "
                         f"{frac(sat['average'])} | {frac(sat['predicted'])} |")
            print(f"  n={n}: P={P} band={int(band.sum())} closed-form {int(cf.sum())}/{P} ranks={ranks} d_pop={frac(stats['population_effective_dimension'])}")
        r3 = res["F1"][3]
        lines += ["", f"In n = 3 the chirality A~∧F~ on the visibility band is a signed integer: "
                  f"{r3['sign_counts_band']['positive']} points positive, {r3['sign_counts_band']['negative']} negative, "
                  f"{r3['sign_counts_band']['zero']} zero; the integer screw chirality (K₁∧K₂)∧(μ₂−μ₁) of the carriers is {r3['screw_chirality_(K1^K2)^(mu2-mu1)']}.", ""]

    # ------------------------------------------------------------------ F2
    with Timer("F2 N packets: exact ladder degree, pairwise additivity, junction cubes"):
        ns = [4, 5, 6, 7, 8]
        Ns = [2, 3, 4, 5, 6]
        table = {}
        add_ok = True
        lines += ["## F2. N packets: highest non-zero chirality degree (exact) versus min(n, 2N − 1)", "",
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
                    center = [(a * (j + 1) + j * j) % 4 for j in range(n)]
                    amp = [(2, 0), (1, 1), (1, -1), (0, 1), (2, 1), (1, 2)][a]
                    packets.append(ex.Packet.carrier_with_sidebands(carrier, side, amp, (1, 0), center=center))
                S = ex.LatticeSystem(packets)
                M = ex.lcg_points(n, 30, seed=10 * n + N)
                Psi, T = S.evaluate(M)
                g = ex.exact_geometry(Psi, T)
                iea = ex.IntExterior(n)
                lad = ex.exact_ladder(iea, g["A"], g["F"])
                top = max((p for p, c in lad.items() if any(int(v) != 0 for v in np.asarray(c, dtype=object).reshape(-1))), default=0)
                rank = max(ex.bareiss_rank(g["F"][p].tolist()) for p in range(M.shape[0]))
                add_ok &= bool(ex.pairwise_identity(S, M[:10]).all()) if N >= 3 else True
                row.append((top, rank))
            table[n] = row
            lines.append(f"| {n} | " + " | ".join(f"{t}{'★' if t == n else ''} (rank {r})" for t, r in row) + " |")
            print(f"  n={n}: " + ", ".join(f"N={N}: degree {t} rank {r}" for N, (t, r) in zip(Ns, row)))
        pred_ok = all(table[n][j][0] == min(n, 2 * N - 1) for n in ns for j, N in enumerate(Ns))
        res["F2"] = {"ns": ns, "Ns": Ns, "top_degree_and_rank": {str(n): table[n] for n in ns}, "all_match_prediction": pred_ok,
                     "pairwise_additivity_exact": add_ok}
        lines += ["", f"Every entry equals min(n, 2N − 1): **{'yes' if pred_ok else 'NO'}**.  ★ = pseudoscalar.  "
                  f"Rank of F~ equals min(n − n mod 2, 2(N − 1)) at the best point.  Pairwise additivity F~ = Σ F~_ab exact at every tested point: **{'yes' if add_ok else 'NO'}**.", ""]
        # junction cubes on the full torus for n = 4, 5
        jl = []
        for n in (4, 5):
            for N in (3, 4):
                packets = []
                for a in range(N):
                    carrier = [0] * n
                    carrier[a % n] = 1
                    side = [[1 if j == (a + 1) % n else 0 for j in range(n)]]
                    center = [(2 * a + j) % 4 for j in range(n)]
                    packets.append(ex.Packet.carrier_with_sidebands(carrier, side, (2, 0), (1, 0), center=center))
                S = ex.LatticeSystem(packets)
                M = ex.torus_points(n)
                Psi, T = S.evaluate(M)
                rho = np.stack([ex.gabs2((Psi[0][:, a], Psi[1][:, a])) for a in range(N)], axis=1)
                dom = ex.dominance(rho)
                labels = {tuple(int(v) for v in M[p]): int(dom[p]) for p in range(M.shape[0])}
                cubes3 = ex.junction_cubes(labels, n, 4, 3)
                cubes4 = ex.junction_cubes(labels, n, 4, 4) if N >= 4 else 0
                jl.append((n, N, [int((dom == a).sum()) for a in range(N)], cubes3, cubes4))
        res["F2"]["junctions"] = jl
        lines += ["Dominance cells and junction cubes on the full torus (unit hypercubes whose corners carry ≥ 3, ≥ 4 distinct dominant packets):", "",
                  "| n | N | cell sizes | ≥3-junction cubes | ≥4-junction cubes |", "|---|---|---|---|---|"]
        for n, N, sizes, c3, c4 in jl:
            lines.append(f"| {n} | {N} | {sizes} | {c3} | {c4} |")
        lines.append("")

    # ------------------------------------------------------------------ F3
    with Timer("F3 two packets at high n (Gram determinants only)"):
        lines += ["## F3. Two packets in high dimension, component-free (Gram determinants)", "",
                  "| n | points | closed form exact | rank F~ = 2 | d_eff | d_pop | mirror mean = 1−6/n | 1−6/n |", "|---|---|---|---|---|---|---|---|"]
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
            R = ex.plane_rotation_90(n, 0, 1)
            R2 = ex.plane_rotation_90(n, 2, 3)
            RR = ex.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    RR[i, j] = sum(int(R2[i][k]) * int(R[k][j]) for k in range(n))
            mu2 = [(j * j + 1) % 3 for j in range(n)]
            p2 = p1.twisted(RR, center=mu2).with_phase(3)
            S = ex.LatticeSystem([p1, p2])
            P = 80 if n <= 64 else 50
            M = ex.lcg_points(n, P, seed=n)
            Psi, T = S.evaluate(M)
            g = ex.exact_geometry(Psi, T) if n <= 32 else None
            pf = ex.pair_factors(Psi, T)
            # A~ without the n x n curvature: A~_j = sum_a Re(conj(psi_a) T_aj)
            Atil = pf["u"][0] + pf["u"][1]
            geo = {"rho": pf["rho_a"][0] + pf["rho_a"][1], "A": Atil}
            if g is not None:
                assert bool(np.all(g["A"] == Atil))
            cf = ex.closed_form_identity_highn(geo, pf)
            rank2 = all(ex.gram_det([pf["x"][p], pf["y"][p]]) != 0 for p in range(P))
            triples = [(Atil[p], pf["x"][p], pf["y"][p]) for p in range(P)]
            stats = ex.decomposable_direction_statistics(triples)
            mir = ex.decomposable_mirror_mean(triples, n)
            res["F3"][n] = {"points": P, "closed_form_count": int(cf.sum()), "rank2_everywhere": rank2,
                            "d_eff": str(stats["effective_dimension"]), "d_pop": str(stats["population_effective_dimension"]),
                            "mirror_all_equal_predicted": mir["all_equal_predicted"], "mirror_pred": str(mir["predicted"])}
            lines.append(f"| {n} | {P} | {int(cf.sum())}/{P} | {'yes' if rank2 else 'NO'} | {frac(stats['effective_dimension'])} | "
                         f"{frac(stats['population_effective_dimension'])} | {'yes' if mir['all_equal_predicted'] else 'NO'} | {frac(mir['predicted'])} |")
            print(f"  n={n}: closed form {int(cf.sum())}/{P}, d_pop={frac(stats['population_effective_dimension'])}, mirror ok={mir['all_equal_predicted']}")
        lines.append("")

    lines += ["## What is exact and what changed", "",
              "* Lattice spacing π/2 and integer wavevectors make every phase a power of i; derivatives bring down integer wavevectors, so all first derivatives are Gaussian integers.",
              "* Gaussian envelopes are replaced by the beat envelopes of carrier + side-band packets (period 4 in every direction); the system lives on the torus (Z/4)^n.",
              "* Projective denominators are cleared: A~ = ρA, F~ = ρ²F, A~∧F~ = ρ³A∧F, F~∧F~ = ρ⁴F∧F are integer-valued.",
              "* Twists are signed permutations (the exact rotations of the lattice); the exact phase group is multiplication by powers of i.",
              "* Ranks use fraction-free Bareiss elimination; norms are kept squared; direction statistics and mirror/saturation averages are exact rationals from Gram determinants; averages over random mirrors and random twists become exact averages over the coordinate designs {±e_i}.",
              "* Not carried over: the 2π flux quantum (an integral) and anything requiring eigenvalues or square roots."]
    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, "F_exact_integer.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(RES_DIR, "F_exact_integer.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print("  wrote results/F_exact_integer.md and .json")
    return res


if __name__ == "__main__":
    run()
