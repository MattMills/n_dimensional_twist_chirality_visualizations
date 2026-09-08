# Exact integer reformulation — results

Every number below is an exact integer or rational; nothing was computed in floating point.

## F1. Two twisted packets on the torus (Z/4)^n

| n | points | ρ>0 | dominant 1 / 2 | exact ties | band V²≥½ | closed form exact | F~ = 2x∧y | phase-invariant | ranks of F~ | d_eff (band) | d_pop (band) | mean mirror cos | 1−6/n | saturation avg | 1−3/n+2/n² |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 64 | 64 | 44 / 20 | 24 | 54 | 64/64 | yes | yes | [2] | 1 | 1 | -1 | -1 | 0.2222… | 0.2222… |
| 4 | 256 | 256 | 170 / 86 | 84 | 220 | 256/256 | yes | yes | [2] | 1.9723… | 1.9811… | -0.5000… | -0.5000… | 0.3750… | 0.3750… |
| 5 | 1024 | 1024 | 680 / 344 | 336 | 880 | 1024/1024 | yes | yes | [2] | 2.7242… | 2.7400… | -0.2000… | -0.2000… | 0.4800… | 0.4800… |
| 6 | 4096 | 4096 | 2720 / 1376 | 1344 | 3520 | 4096/4096 | yes | yes | [2] | 2.6794… | 2.6945… | 0 | 0 | 0.5555… | 0.5555… |

In n = 3 the chirality A~∧F~ on the visibility band is a signed integer: 29 points positive, 25 negative, 0 zero; the integer screw chirality (K₁∧K₂)∧(μ₂−μ₁) of the carriers is 1.

## F2. N packets: highest non-zero chirality degree (exact) versus min(n, 2N − 1)

| n \ N | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| 4 | 3 (rank 2) | 4★ (rank 4) | 4★ (rank 4) | 4★ (rank 4) | 4★ (rank 4) |
| 5 | 3 (rank 2) | 5★ (rank 4) | 5★ (rank 4) | 5★ (rank 4) | 5★ (rank 4) |
| 6 | 3 (rank 2) | 5 (rank 4) | 6★ (rank 6) | 6★ (rank 6) | 6★ (rank 6) |
| 7 | 3 (rank 2) | 5 (rank 4) | 7★ (rank 6) | 7★ (rank 6) | 7★ (rank 6) |
| 8 | 3 (rank 2) | 5 (rank 4) | 7 (rank 6) | 8★ (rank 8) | 8★ (rank 8) |

Every entry equals min(n, 2N − 1): **yes**.  ★ = pseudoscalar.  Rank of F~ equals min(n − n mod 2, 2(N − 1)) at the best point.  Pairwise additivity F~ = Σ F~_ab exact at every tested point: **yes**.

Dominance cells and junction cubes on the full torus (unit hypercubes whose corners carry ≥ 3, ≥ 4 distinct dominant packets):

| n | N | cell sizes | ≥3-junction cubes | ≥4-junction cubes |
|---|---|---|---|---|
| 4 | 3 | [140, 72, 44] | 128 | 0 |
| 4 | 4 | [119, 66, 42, 29] | 176 | 80 |
| 5 | 3 | [560, 288, 176] | 512 | 0 |
| 5 | 4 | [476, 264, 168, 116] | 704 | 320 |

## F3. Two packets in high dimension, component-free (Gram determinants)

| n | points | closed form exact | rank F~ = 2 | d_eff | d_pop | mirror mean = 1−6/n | 1−6/n |
|---|---|---|---|---|---|---|---|
| 16 | 80 | 80/80 | yes | 6.8719… | 7.4237… | yes | 0.6250… |
| 32 | 80 | 80/80 | yes | 6.5443… | 7.0383… | yes | 0.8125… |
| 64 | 80 | 80/80 | yes | 7.1613… | 7.7671… | yes | 0.9062… |
| 128 | 50 | 50/50 | yes | 6.0155… | 6.7015… | yes | 0.9531… |
| 256 | 50 | 50/50 | yes | 6.8892… | 7.8303… | yes | 0.9765… |

## What is exact and what changed

* Lattice spacing π/2 and integer wavevectors make every phase a power of i; derivatives bring down integer wavevectors, so all first derivatives are Gaussian integers.
* Gaussian envelopes are replaced by the beat envelopes of carrier + side-band packets (period 4 in every direction); the system lives on the torus (Z/4)^n.
* Projective denominators are cleared: A~ = ρA, F~ = ρ²F, A~∧F~ = ρ³A∧F, F~∧F~ = ρ⁴F∧F are integer-valued.
* Twists are signed permutations (the exact rotations of the lattice); the exact phase group is multiplication by powers of i.
* Ranks use fraction-free Bareiss elimination; norms are kept squared; direction statistics and mirror/saturation averages are exact rationals from Gram determinants; averages over random mirrors and random twists become exact averages over the coordinate designs {±e_i}.
* Not carried over: the 2π flux quantum (an integral) and anything requiring eigenvalues or square roots.
