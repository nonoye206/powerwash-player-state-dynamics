# Step 4.1 Player-level State Sequences and Transition Matrix

State source: `pc1_pc5_gmm_diag_k5` labels. State IDs are numeric only; no state names are assigned here.

## Sequence Summary
- sessions: 105818
- players: 10834
- players with at least 2 sessions: 8031
- transitions: 94984
- minimal purchase anomaly sessions flagged: 54
- update-window sessions flagged: 9358

## State Summary
| state | sessions | session % | players | starts | ends | incoming | outgoing | self prob |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8645 | 8.17% | 8484 | 8441 | 1930 | 204 | 6715 | 0.0070 |
| 1 | 46709 | 44.14% | 6947 | 143 | 2910 | 46566 | 43799 | 0.5598 |
| 2 | 35784 | 33.82% | 6154 | 712 | 2399 | 35072 | 33385 | 0.4684 |
| 3 | 7158 | 6.76% | 3293 | 815 | 725 | 6343 | 6433 | 0.2504 |
| 4 | 7522 | 7.11% | 3988 | 723 | 2870 | 6799 | 4652 | 0.4615 |

## Transition Counts
|  | to_0 | to_1 | to_2 | to_3 | to_4 |
|---|---|---|---|---|---|
| from_0 | 47 | 4242 | 1555 | 465 | 406 |
| from_1 | 99 | 24519 | 14886 | 2204 | 2091 |
| from_2 | 15 | 14175 | 15639 | 1878 | 1678 |
| from_3 | 26 | 2448 | 1871 | 1611 | 477 |
| from_4 | 17 | 1182 | 1121 | 185 | 2147 |

## Transition Probabilities
|  | to_0 | to_1 | to_2 | to_3 | to_4 |
|---|---|---|---|---|---|
| from_0 | 0.007 | 0.6317 | 0.2316 | 0.0692 | 0.0605 |
| from_1 | 0.0023 | 0.5598 | 0.3399 | 0.0503 | 0.0477 |
| from_2 | 0.0004 | 0.4246 | 0.4684 | 0.0563 | 0.0503 |
| from_3 | 0.004 | 0.3805 | 0.2908 | 0.2504 | 0.0741 |
| from_4 | 0.0037 | 0.2541 | 0.241 | 0.0398 | 0.4615 |

## Notes
- Transition rows are row-normalized: each row sums over the next state conditional on the current state.
- Players with one session contribute to state counts and start/end counts, but not transitions.
- Date-window and purchase-anomaly flags are included for Step 4 sensitivity checks.