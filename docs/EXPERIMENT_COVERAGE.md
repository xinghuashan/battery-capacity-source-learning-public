# Final-paper coverage

| Paper component | Entry point or material | Release validation |
|---|---|---|
| 17 inputs, next-reference label, past-five history | `prepare_data`, `features`, `preprocessing` | All six extracted inputs and labels match the archived inputs exactly |
| Fixed three folds, two label schedules, seeds 0–4 | `config/protocol.json`, `train` | All six training jobs completed under the fixed protocol |
| Thirteen estimators and source-only regularization selection | `models`, `episodes`, `grouping`, `kernels` | 270 final models; 24 source-selection records and 11,430 new predictions checked |
| Primary 109-event and all-127 metrics | `analysis`, `evaluate` | Recomputed from all archived event predictions |
| Table 3 and Figure 2 | `analysis`, `figures` | Full-precision agreement; figure rendered |
| Event-level trajectories, Figure 3 | `analysis`, `figures` | All 109 primary events and all five M05 seeds retained |
| Per-cell comparisons, Figure 4 | `paired_analysis`, `figures` | Agreement with archived paired differences |
| Budget interaction, cell/pair deletion, Figure 5 | `paired_analysis`, `figures` | Agreement with archived values; no refitting |
| Residual alignment/energy, Table 4, Figure 6 | `correction_analysis`, `figures` | Identity checked; agreement with archived diagnostics |
| Amplitudes 0, 0.5 and 1 | `correction_analysis` | Specified saved-prediction diagnostics only; no amplitude selection |
| All five figure exports | `figures` | PNG, PDF and SVG generated; input-table hashes recorded |
| Historical implementation audits | Described in the manuscript | No claim of rerunning finite-difference or full audit suites |
| Earlier 150-design exploration and excluded external protocols | Historical development context | Outside this final-method release; no new external scores |

The saved-prediction pipeline and the raw-data training/prediction pipeline
have both been exercised. Raw-data numerical parity is established for the
recorded original environment; a fresh analysis-only environment was tested
separately. These checks reproduce the existing six-cell development study
and do not add independent external validation.
