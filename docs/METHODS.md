# Methods and implementation

| Paper label | Implementation | Defining choice |
|---|---|---|
| Ridge | `episodes.fit_anchor`, `models.fit_method` | Source-input scaler, cell-weighted Ridge, alpha = 1 |
| Ridge-CV | `models.fit_method` | Four source-query RMSEs select anchor alpha |
| KRR | `kernels`, `models.fit_method` | Residual RBF head, original 9-candidate grid |
| KRR-wide | Same | Residual RBF head, 24-candidate grid |
| B3 | `baselines.DirectResidual` | Two masked branches with jointly trained scalar heads |
| N01 | Same | B3 multiplied by a frozen source-context spread gate |
| M01 | `models.fit_method` | Weighted linear Ridge fitted to anchor residuals |
| M02 | `readout.FrozenHeads` | Fixed hidden features and gradient-trained raw head |
| M03 | `readout.Features`, `solve_readout` | Fixed initial features with normalized analytic head |
| M04 | Same | Learned features, source-in objective |
| M05 | Same | Learned features, leave-one-physical-source-cell query objective |
| M06 | `grouping.pseudo_assign`, same readout | Matched atomic pseudo-groups; physical-cell weighting |
| Raw-CV | `models.train_raw`, `select_raw` | Unnormalized features and source-selected head regularization |

Raw-CV is stored as `M05-raw-CV` in the original result CSVs. The command-line
name is `Raw-CV`; these labels refer to the same paper control.

Both branches use 17-16-16 SiLU hidden layers. Charge and history masks share
the cycle-index input. M03-M06 concatenate the two half-scaled hidden vectors
and two constant 0.5 coordinates, yielding 34 features. Their support energy is
the equal-physical-cell mean squared feature norm divided by 34. The analytic
head uses weighted labeled residuals in percentage points, an identity penalty,
and a differentiable Cholesky solve. The gradient includes support-energy
normalization and the analytic head. Final heads use all four source cells.

The scaler uses all support input rows without cell weights. Ridge and labeled
head weights are n / (number_of_physical_cells * labels_in_that_cell). M05 averages
four physical-cell query losses equally. M06 retains physical-cell query
weights rather than treating its pseudo-groups as independent physical cells.

Gradient-trained estimators use 300 full-batch AdamW updates, learning rate
0.001 and weight decay 0.001. Reported stochastic estimators use seeds 0-4.
There is no target-dependent epoch choice or fitted target calibration.

The original KRR regularization grid is (0.1, 1, 10), crossed with eta
(0.5, 1, 2). The wider regularization grid is (0.0001, 0.001, 0.01, 0.1, 1,
10, 100, 1000). Ridge-CV and Raw-CV use the wider grid. Selection minimizes the
exact mean of four source-query RMSEs; ties prefer greater regularization and,
for KRR, smaller eta. KRR length scale uses the median strictly positive pair
distance among support inputs, with squared distances divided by 17.

Raw-CV selects regularization with seed 0 using an outer held source and
three inner query episodes among the remaining three sources. Each inner
episode has two support cells. The selected value is then used in the final
four-source training for each of five seeds. This control uses more tuning
compute than M05 and is not a compute-matched single-factor ablation.

The full-label and sparse-label results differ. M05 has the lowest sparse
aggregate errors in this comparison, while N01 leads with full labels.
M05 does not improve every cell relative to M06. These patterns are retained
in the saved tables and figures.
