# Literature Notes

## LeWorldModel

Reference:

```text
Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero.
LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.
arXiv:2603.19312, 2026.
https://arxiv.org/abs/2603.19312
```

Relevant points:

- LeWorldModel trains an end-to-end JEPA world model from pixels.
- Its loss has two main terms: next-embedding prediction and SIGReg.
- It does not use stop-gradient, EMA target encoders, pre-trained encoders, reconstruction, or reward supervision as its stabilizing mechanism.
- Its SIGReg objective enforces Gaussian-distributed latent embeddings through random one-dimensional projections and an Epps-Pulley normality statistic.
- The paper's default scale is small enough to be relevant to a single-GPU follow-up, but reproducing the full benchmark set is outside the 25 USD budget.

## LeJEPA / SIGReg

Reference:

```text
Randall Balestriero, Yann LeCun.
LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics.
arXiv:2511.08544, 2025.
https://arxiv.org/abs/2511.08544
```

Relevant points:

- LeJEPA motivates an isotropic Gaussian target distribution for JEPA embeddings.
- SIGReg is introduced as Sketched Isotropic Gaussian Regularization.
- The relevant correction for the prior experiment is that SIGReg is not just covariance matching.

## KerJEPA

Reference:

```text
Eric Zimmermann, Harley Wiltzer, Justin Szeto, David Alvarez-Melis, Lester Mackey.
KerJEPA: Kernel Discrepancies for Euclidean Self-Supervised Learning.
arXiv:2512.19605, 2025.
https://arxiv.org/abs/2512.19605
```

Relevant points:

- Clarifies that the LeJEPA Epps-Pulley regularizer can be understood as a sliced MMD with Gaussian prior and Gaussian kernel.
- Useful for explaining the math in the final paper.
- Supports using the closed-form Gaussian-kernel MMD-to-normal expression for implementation sanity checks.

## Fast LeWorldModel

Reference:

```text
Yuntian Gao, Xiangyu Xu.
Fast LeWorldModel.
arXiv:2606.26217, 2026.
https://arxiv.org/abs/2606.26217
```

Relevant points:

- Identifies autoregressive one-step LeWM rollout as susceptible to accumulated latent errors.
- Replaces autoregressive latent rollout with action-prefix prediction and dense multi-horizon supervision.
- This is highly relevant to the training-protocol hypothesis, but adding it to the main experiment would confound protocol with architecture/interface.

## DreamerV3

Reference:

```text
Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap.
Mastering Diverse Domains through World Models.
arXiv:2301.04104, 2023.
https://arxiv.org/abs/2301.04104
```

Relevant points:

- Standard reference for world models using imagination for behavior learning.
- Useful for framing why rollout quality matters downstream.

## DINO-WM And Reward-Free Planning With Latent Dynamics

References:

```text
Gaoyue Zhou, Hengkai Pan, Yann LeCun, Lerrel Pinto.
DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning.
arXiv:2411.04983, 2024.
https://arxiv.org/abs/2411.04983

Vlad Sobal, Wancong Zhang, Kynghyun Cho, Randall Balestriero, Tim G. J. Rudner, Yann LeCun.
Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models.
arXiv:2502.14819, 2025.
https://arxiv.org/abs/2502.14819
```

Relevant points:

- These papers motivate reward-free latent dynamics and test-time planning.
- They support the paper framing that latent rollout drift matters because planning optimizes over imagined trajectories.

## Teacher Forcing / Exposure Bias

Reference:

```text
Alex Lamb, Anirudh Goyal, Ying Zhang, Saizheng Zhang, Aaron Courville, Yoshua Bengio.
Professor Forcing: A New Algorithm for Training Recurrent Networks.
arXiv:1610.09038, 2016.
https://arxiv.org/abs/1610.09038
```

Relevant points:

- Establishes the teacher-forcing versus free-running mismatch in sequence models.
- Useful background for why 1-step supervised dynamics can behave poorly under rollout.

Reference:

```text
Riku Green, Zahraa S. Abdallah, Telmo M Silva Filho.
Exposure Bias as Epistemic Underidentification in Recursive Forecasting.
arXiv:2606.12990, 2026.
https://arxiv.org/abs/2606.12990
```

Relevant points:

- Recent framing of recursive rollout failure as more than simple distribution shift.
- Useful if the final results point toward self-induced latent states that are underidentified by one-step teacher forcing.

