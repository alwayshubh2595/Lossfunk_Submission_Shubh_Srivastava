| 3 | “The stated confirmatory conclusions do not match the preregistered estimands.” | Mark \(C_R,C_P,C_I\) formally unevaluable; rename the Lorenz result \(C_{P\mid R2}\); label R1-vs-R2 interaction exploratory. |
| 4 | “The Lorenz story is one environment, five seeds, and driven by H50/outliers.” | Show paired seed points, geometric effects, exact tests, leave-one-out results, and the full horizon profile. |
| 5 | “Construct and external validity are weak: nonofficial SIGReg, no parity/matched-dimension arm, and failed bridge.” | Use exact implementation-specific naming, include a deviation ledger, state that the prior bridge was not reproduced, and frame the paper as a stress test generating hypotheses rather than a general SIGReg verdict. |

The paper is strongest when it treats the failed attribution as the result: viability must be established before drift effects can be causally compared.
tokens used
82,142
The results support a credible workshop paper, but only as a mixed/negative result about collapse-gated attribution—not as a general victory for protocol or a general failure of SIGReg.

## 1. Strongest honest headline

> In this finite-training, overcomplete latent benchmark, regularizer choice primarily determined whether rollout drift was measurable at all. Differential representation collapse precluded the preregistered no-regularizer–vs–SIGReg drift contrasts in most environments. Where both regularized alternatives remained viable—Lorenz—closed-loop training suppressed long-horizon amplification, while regularizer differences were concentrated under one-step training.

Concrete support:

- No regularizer met the composite collapse criterion in 46/46 core runs.
- SIGReg met it in 26/26 non-Lorenz main P0/P1 runs, versus 0/10 Lorenz runs. OU H-step teacher forcing was 4/5, not 5/5, collapsed.
- Moment matching collapsed in only 3/36 main P0/P1 runs, although the bridge adds another 4/10 failures.
- On Lorenz, all five seeds improved under BPTT for both moment and SIGReg.
- The scientifically fruitful message is methodological: collapse is a post-treatment intercurrent event—analogous to truncation by death—that can make an otherwise clean factorial drift study unidentified.

A suitable title would be:

> When Collapse Censors Rollout-Drift Attribution

Do not headline “protocol dominates” or “SIGReg fails.” Both exceed the evidence.

One formal correction is important: the reported Lorenz value −1.20, CI [−1.75, −0.65], is \(P1-P0\) within R2/SIGReg, not the preregistered \(C_P\), which was defined as averaged over R0 and R2. With R0 unavailable, preregistered \(C_P\) is also technically unevaluable. Rename it \(C_{P\mid R2}\). The preregistered “protocol dominates” criterion additionally required support in at least two dynamics families and \(C_R\) equivalence, neither of which occurred.

## 2. Alternative explanations

These must either be addressed by reanalysis or stated prominently.

- Lambda selection is the largest threat. SIGReg selected the upper grid boundary, λ=1, on OU and Lorenz; only that value was non-collapsed in those pilots. Both point-mass SIGReg grids were entirely collapsed. Once the non-collapse constraint failed, selecting minimum latent one-step MSE actively favored the most collapsed solution. The design document does not clearly preregister this fallback. Moreover, moment and EP losses have radically different scales, so the common numerical grid is not dose-matched. The conclusion must be “at the selected weights,” not “SIGReg intrinsically collapses.”

- The shortened training budget is not treatment-neutral merely because it was uniform. Regularizers and protocols can converge or collapse at different rates. The 20k→10k and 15k→8k reduction therefore leaves a treatment-by-training-time explanation open. Restrict claims to the observed training horizon. The bridge failure makes this caveat substantive rather than hypothetical.

- “Collapse” combines distinct phenomena. OU SIGReg retained probe \(R^2\approx0.67\)–0.87; pointmass-state SIGReg retained \(R^2\approx0.95\)–1.00. Several no-regularizer Lorenz runs also had high \(R^2\). These are often near-zero-scale or low-effective-dimensional codes whose information can be recovered after probe standardization—not complete information loss. Use “variance/dimensional collapse under the preregistered criterion,” not “the representation contains no state information.”

- The bridge did not reproduce. Stable-linear-det moment/BPTT failed the probe criterion in 4/5 seeds. Therefore the experiment cannot be presented as a direct replication or clean overturning of the earlier result. Changes in training length, lambda, metric, and implementation remain explanations.

- The dropped matched-dimension arm matters. A Gaussian target in d=16/32 for low-intrinsic-dimensional deterministic encodings may be difficult or structurally mismatched. The absent Gaussianity/stationarity diagnostics and M=1024 sensitivity leave this unresolved.

- Construct validity remains incomplete. This is an exact-integral sliced EP variant, not official quadrature SIGReg, and the preregistered official loss/gradient parity gate was not completed. Claims must remain implementation-specific.

- Two additional preregistration violations are visible in the code: final reporting uses validation rather than test data, and evaluation windows are randomly sampled per run rather than loaded from a fixed shared suite. This weakens both held-out interpretation and seed-paired inference.

- Random projections consume the same global RNG used for minibatch selection. Consequently, SIGReg conditions do not receive the same minibatch stream as moment/none conditions despite sharing a seed. This is another limitation of the paired causal interpretation.

## 3. Is the proposed claim defensible?

Only after revision.

Defensible:

> In this experiment, regularizer choice acted primarily as a representation-viability gate. Differential geometric collapse preempted the preregistered R0-vs-R2 drift attribution.

Overclaimed:

> The regularizer’s role is anti-collapse, not drift control.

The data do not identify “not drift control”: \(C_R\) is unavailable, and Lorenz descriptively shows a substantial regularizer effect under P0. Lambda sensitivity, loss-scale matching, and longer training were not performed.

Use:

> Its direct effect on drift remains unidentified because treatment-induced collapse removed the required comparisons.

## 4. Minimal extra analysis using existing data

Highest-value additions:

1. Collapse-mode decomposition.

   Report, per run, log covariance trace, effective dimension, rank, and probe \(R^2\). Separate:

   - near-zero-scale collapse;
   - dimensional/anisotropic collapse;
   - informational collapse, defined by probe failure.

   Include sensitivity tables for alternative effective-dimension and probe thresholds. This will resolve the OU objection directly.

2. Paired seed and horizon analysis on Lorenz.

   Use per-seed log effects, raw dots, leave-one-seed-out estimates, and exact sign-flip tests alongside the preregistered bootstrap:

   - SIGReg protocol effect: mean log ratio −1.20, geometric BPTT/P0 ratio 0.30; all five seeds improve, but exact two-sided sign-flip \(p=0.0625\).
   - Moment protocol effect: −3.09, ratio 0.046; exact \(p=0.0625\).
   - Exploratory SIGReg-vs-moment effect under P0: −1.82, or 6.2× geometrically—not 16×. The 16× figure is a ratio of skewed arithmetic means.
   - Under BPTT: +0.064, ratio 1.07.
   - Exploratory interaction: +1.89 log units, but exact sign-flip \(p=0.1875\). Call it a descriptive interaction pattern, not strong inferential evidence.

   The horizon profile is especially informative: under SIGReg, BPTT is 2.63× worse at H=1 but 0.158× at H=50. Under moment it is 1.06× at H=1 and 0.0176× at H=50. This shows a genuine short-horizon/long-horizon tradeoff rather than generic accuracy improvement.

3. Lambda-feasibility plot.

   Plot pilot λ against collapse status, probe \(R^2\), and validation MSE. Explicitly show boundary selections and all-collapsed grids. This cannot replace full lambda sensitivity, but it makes the limitation auditable.

4. Triangulate with amplification metrics.

   For viable Lorenz runs, show paired changes in H25/H50 perturbation amplification and absolute excess-amplification error. This supports the mechanism that BPTT suppresses unstable long-horizon dynamics rather than merely changing the probe metric.

A useful analytic observation is that at γ=0.5 and exactly zero latents, the EP loss equals

\[
1 + 1/\sqrt{3} - \sqrt{2} \approx 0.163,
\]

which matches the final regularizer loss of many collapsed SIGReg cells. Exact collapse is a zero-gradient stationary point. Present this as a candidate optimization explanation, not a proven cause.

## 5. Top reviewer objections

| Rank | Objection | Preemption |
|---|---|---|
| 1 | “SIGReg collapse is just lambda/loss-scale mis-tuning.” | Restrict claims to selected λ; show the full pilot frontier; disclose boundary/all-collapsed selection; avoid intrinsic superiority claims. |
| 2 | “Your collapse label is misleading because state remains decodable.” | Split variance, dimensional, and informational collapse; show continuous geometry and threshold sensitivity. |
| 3 | “The stated confirmatory conclusions do not match the preregistered estimands.” | Mark \(C_R,C_P,C_I\) formally unevaluable; rename the Lorenz result \(C_{P\mid R2}\); label R1-vs-R2 interaction exploratory. |
| 4 | “The Lorenz story is one environment, five seeds, and driven by H50/outliers.” | Show paired seed points, geometric effects, exact tests, leave-one-out results, and the full horizon profile. |
| 5 | “Construct and external validity are weak: nonofficial SIGReg, no parity/matched-dimension arm, and failed bridge.” | Use exact implementation-specific naming, include a deviation ledger, state that the prior bridge was not reproduced, and frame the paper as a stress test generating hypotheses rather than a general SIGReg verdict. |

The paper is strongest when it treats the failed attribution as the result: viability must be established before drift effects can be causally compared.
