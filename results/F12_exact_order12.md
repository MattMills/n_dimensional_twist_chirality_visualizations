# Exact reformulation with 12th roots of unity — results

Phases are powers of ζ₁₂ = e^{iπ/6}; every real quantity is an element a + b√3 of ℤ[√3] (integer pairs), with rationals only in the final ratios. Decimals are exactly truncated expansions of elements of ℚ(√3); nothing was computed in floating point.

## F12-1. Two twisted packets: spread of chirality directions, order 12 vs order 4

| n | side-bands | order | points | band | closed form | F~=2x∧y | phase-inv. | ranks | d_eff | d_pop | mirror = 1−6/n | saturation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 3 | 4 | 64 | 54 | 64/64 | yes | yes | [2] | 1 | 1 | yes (-1) | yes |
| 3 | 3 | 12 | 1728 | 1320 | 1728/1728 | yes | yes | [2] | 1.0000 | 1.0000 | yes (-1.0000) | yes |
| 3 | 6 | 4 | 64 | 51 | 64/64 | yes | yes | [0, 2] | 1 | 1 | yes (-1) | yes |
| 3 | 6 | 12 | 1728 | 1172 | 1728/1728 | yes | yes | [0, 2] | 1.0000 | 1.0000 | yes (-1.0000) | yes |
| 4 | 3 | 4 | 600 | 518 | 600/600 | yes | yes | [2] | 1.8901… | 1.8966… | yes (-1.5000…) | yes |
| 4 | 3 | 12 | 600 | 463 | 600/600 | yes | yes | [2] | 1.6175… | 1.6214… | yes (-0.5000…) | yes |
| 4 | 6 | 4 | 600 | 406 | 600/600 | yes | yes | [2] | 2.5469… | 2.5622… | yes (-1.5000…) | yes |
| 4 | 6 | 12 | 600 | 398 | 600/600 | yes | yes | [2] | 2.7007… | 2.7185… | yes (-0.5000…) | yes |
| 5 | 3 | 4 | 600 | 517 | 600/600 | yes | yes | [2] | 2.4925… | 2.5069… | yes (-1.8000…) | yes |
| 5 | 3 | 12 | 600 | 454 | 600/600 | yes | yes | [2] | 2.4047… | 2.4178… | yes (-0.2000…) | yes |
| 5 | 6 | 4 | 600 | 432 | 600/600 | yes | yes | [2] | 3.2605… | 3.2892… | yes (-1.8000…) | yes |
| 5 | 6 | 12 | 600 | 454 | 600/600 | yes | yes | [2] | 3.3749… | 3.4061… | yes (-0.2000…) | yes |
| 6 | 3 | 4 | 600 | 507 | 600/600 | yes | yes | [2] | 2.5575… | 2.5730… | yes (0) | yes |
| 6 | 3 | 12 | 600 | 459 | 600/600 | yes | yes | [2] | 2.5629… | 2.5785… | yes (0.0000) | yes |
| 6 | 6 | 4 | 600 | 488 | 600/600 | yes | yes | [0, 2] | 4.4682… | 4.5288… | yes (0) | yes |
| 6 | 6 | 12 | 600 | 448 | 600/600 | yes | yes | [2] | 4.7863… | 4.8574… | yes (0.0000) | yes |

## F12-2. N packets (order 12): highest non-zero chirality degree versus min(n, 2N − 1)

| n \ N | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| 4 | 3 (rank 2) | 4★ (rank 4) | 4★ (rank 4) | 4★ (rank 4) | 4★ (rank 4) |
| 5 | 3 (rank 2) | 5★ (rank 4) | 5★ (rank 4) | 5★ (rank 4) | 5★ (rank 4) |
| 6 | 3 (rank 2) | 5 (rank 4) | 6★ (rank 6) | 6★ (rank 6) | 6★ (rank 6) |
| 7 | 3 (rank 2) | 5 (rank 4) | 7★ (rank 6) | 7★ (rank 6) | 7★ (rank 6) |
| 8 | 3 (rank 2) | 5 (rank 4) | 7 (rank 6) | 8★ (rank 8) | 8★ (rank 8) |

Every entry equals min(n, 2N − 1): **yes**. Pairwise additivity F~ = Σ F~_ab exact: **yes**.

Dominance cells and junction cubes on the full torus (ℤ/12)ⁿ:

| n | N | cell sizes | ≥3-junction cubes | ≥4-junction cubes |
|---|---|---|---|---|
| 3 | 3 | [715, 562, 451] | 128 | 0 |
| 3 | 4 | [646, 380, 321, 381] | 160 | 8 |
| 4 | 3 | [8580, 6744, 5412] | 1536 | 0 |
| 4 | 4 | [6839, 5554, 4562, 3781] | 2736 | 336 |

## F12-3. Two packets in high dimension (order 12), component-free

| n | points | closed form | rank 2 | d_eff | d_pop | mirror mean = 1−6/n |
|---|---|---|---|---|---|---|
| 16 | 60 | 60/60 | yes | 5.9521… | 6.4975… | yes (0.6250…) |
| 32 | 60 | 60/60 | yes | 6.7007… | 7.4175… | yes (0.8125…) |
| 64 | 60 | 60/60 | yes | 7.1322… | 7.9595… | yes (0.9062…) |
| 128 | 40 | 40/40 | yes | 4.3558… | 4.7659… | yes (0.9531…) |
| 256 | 40 | 40/40 | yes | 5.7889… | 6.5992… | yes (0.9765…) |

## Notes

* ℚ(ζ₁₂) = ℚ(i, √3): the real subfield is ℚ(√3), so exact real quantities are integer pairs a + b√3 (after scaling every phase to 2ζ, which is a harmless global factor). Only orders 3, 4 and 6 have a rational real subfield.
* Ranks use fraction-free Bareiss elimination in ℤ[√3] (exact division); statistics are elements of ℚ(√3) rendered as exactly truncated decimals.
* Order 4 rows recompute experiment F on matched point counts; order 12 rows use the same packets with three-times-wider beat envelopes and a twelve-element phase group.
