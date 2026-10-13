---
layout: default
title: "Does Muon Operate at the Edge of Stability in LLM Pretraining?"
description: "A visual guide to how no-momentum Muon preserves a loss edge while separating loss balance from update reversal."
permalink: /blog/muon-eos-en/
lang: en
math: true
---

[Home](/)

# Does Muon Operate at the Edge of Stability in LLM Pretraining?

Yanzhe Chen · A visual research guide

**Our answer is “sublates”: Muon preserves a stochastic loss-neutral edge, while separating loss balance from update reversal.**

This post explains our [paper](https://arxiv.org/abs/2609.34915), joint work with Qifang Zhao, Xiaoxiao Xu, and Fanghui Liu. We focus on **no-momentum Muon**, from exact-polar theory to language-model experiments up to 1B parameters. The figures below tell the story, with the geometric intuition and technical definitions included directly in the text.

[Paper](https://arxiv.org/abs/2609.34915) · [Code](https://github.com/cyzebra/Muon-Sublates-the-Edge-of-Stability-in-LLM-Pretraining)

## 1. First, what does classical EoS look like?

![Classical EoS: curvature near the learning-rate boundary and loss oscillations](/assets/images/muon/classical-eos-litman2026.png)

*Figure 1. Full-batch GD on a three-layer MLP, CIFAR-10, learning rate 0.5. Curvature approaches $$2/\eta$$; loss fluctuates locally while improving over longer horizons. The inset reveals detrended oscillations. Reproduced unchanged from [Litman (2026), Figure 1](https://arxiv.org/html/2604.20446v1#S1.F1), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

**Read the two panels together:** the left tracks a curvature boundary; the right shows that reaching it need not stop learning. Approaching the edge does not require exact equality at every step.

In one scalar quadratic mode, GD makes the connection especially clear. Its update multiplier is $$1-\eta\lambda$$. At $$\eta\lambda=2$$, the multiplier is -1: loss is unchanged, consecutive updates reverse with equal magnitude, and the linear dynamics is marginally stable.

## 2. An intuitive starting point: why GD is a special geometry

A small parameter perturbation changes the gradient, and the optimizer then turns that gradient change into an update change. **There are two pieces: the loss geometry and the optimizer's transformation.**

For a smooth deterministic or fixed-batch transformation $$T$$, write

$$
F(W)=W-\eta T(\nabla L(W)).
$$

Where the derivatives exist, the chain rule gives

$$
DF(W)=I-\eta\,DT(G)\circ\nabla^2L(W).
$$

GD uses $$T=\operatorname{Id}$$: the transformation contributes no extra geometry, and the local operator reduces to $$I-\eta H$$. A nonlinear update transformation adds its own derivative. This helps explain why loss balance, temporal direction and perturbation stability need not remain coupled under a different optimizer. The local formula assumes differentiability; rank changes require additional care.

### What is no-momentum Muon?

For a minibatch matrix gradient with compact SVD, exact polar Muon uses

$$
\widehat G_k=U_r\Sigma_rV_r^\top,
\qquad P_k=\operatorname{Pol}(\widehat G_k)=U_rV_r^\top,
$$

$$
W_{k+1}=W_k-\eta P_k.
$$

It retains the active singular vectors and normalizes their singular values to one. This changes the spectral structure of the update, rather than uniformly rescaling the whole gradient. Our theoretical analysis isolates this exact, no-momentum update; the LLM experiments use the practical NS-5 approximation.

## 3. One comparison, separate diagnostics

![Combined GD and Muon comparison for loss balance, temporal sign, exact reversal and local perturbations](/assets/images/muon/diagram-split-edge.svg)

*Figure 2. A combined diagnostic reference. The GD column describes a scalar quadratic mode; the Muon column separates the stochastic loss edge, temporal orthogonality, exact reversal and local perturbations. The Jacobian perspective is explanatory: our LLM T1/T2 probes do not measure perturbation stability.*

This single comparison collects the roles that classical EoS tends to bring together. The definitions below explain how we measure the loss and direction columns.

### Loss balance: the exact conditional identity

Fix $$W_k$$, let $$G_k=\nabla L(W_k)\ne0$$, and resample a minibatch of size $$b$$. Define

$$
\rho_b(W_k)=\frac{\mathbb{E}_b\langle G_k,P_k\rangle_F}{\|G_k\|_*},
$$

$$
s_b^M(W_k;\eta)=\frac{2\mathbb{E}_b\left[L(W_k-\eta P_k)-L(W_k)+\eta\langle G_k,P_k\rangle_F\right]}{\eta^2\|G_k\|_*}.
$$

The nuclear norm $$\|G_k\|_*$$ is the sum of gradient singular values. Coherence $$\rho_b\in[-1,1]$$ measures retained first-order descent; $$s_b^M$$ measures the normalized finite-step loss remainder. An unbiased minibatch gradient need not induce an unbiased polar direction.

These definitions give

$$
\mathbb{E}_b\left[L(W_{k+1})-L(W_k)\mid W_k\right]
=\frac{\eta^2\|G_k\|_*}{2}\left(s_b^M-\frac{2\rho_b}{\eta}\right).
$$

**The bracket determines the loss-change sign.** T1 is its zero boundary. Full-batch exact-polar updates have $$\rho_{\mathrm{full}}=1$$, recovering the loss-neutral scale $$2/\eta$$. Effective Muon curvature is not GD's maximum Hessian eigenvalue; under additional second-order smoothness, its small-step limit is normalized directional curvature.

### Temporal direction: a different measurement

For consecutive nonzero directions,

$$
c_k^{\mathrm{pol}}=\frac{\langle P_k,P_{k+1}\rangle_F}{\|P_k\|_F\|P_{k+1}\|_F}.
$$

**Negative alignment does not determine which side of T1 the update occupies.** A matrix quadratic proves this directly: at the same learning rate and full coherence, the paper constructs two states with

| State | Effective loss curvature | Consecutive cosine |
| --- | --- | --- |
| A | $$5/(4\eta)$$ | $$-1$$ |
| B | $$5/(4\eta)$$ | $$1/3$$ |

Both decrease loss, but their direction signs differ. Curvature depends on a spectral trace; alignment depends on an eigenvalue sign count. The construction and assumptions appear in paper §2.3. A negative cosine alone also does not establish a two-step parameter return.

## 4. Does batch size change the picture?

![Controlled Transformer loss, conditional loss balance and temporal direction across batches](/assets/images/muon/figure1-controlled.png)

*Figure 3. Exact SVD-polar Transformer, fixed learning rate 0.007. Compare the same batch's loss, loss-boundary tracking and direction cosine across the three panels. From the top row of paper Figure 1.*

**Look at the right panel:** larger batches produce more negative alignment in these runs; small batches stay closer to orthogonality. Full-batch trajectories move toward coherent reversal without reaching it in the displayed global measurements.

![First loss-rise, T1 and T2 events in controlled MLP, CNN and Transformer](/assets/images/muon/figure3-event-order.png)

*Figure 4. Recorded first events at learning rate 0.007. Red is T1 and blue is T2. Compare their order within each batch configuration. From paper Figure 3; event detection uses the paper's recorded tolerances.*

**The order can swap.** In the displayed MLP, batch 64 has T2 before T1, while batch 4096 has T1 before T2. The two diagnostics do not mark one universal event.

## 5. What happens in LLM pretraining?

The LLM runs use no-momentum NS-5 approximation on Muon blocks and AdamW on auxiliary parameters. Their measurements complement, rather than replace, exact-polar theory.

| Experiment | Configuration | Evidence it supplies |
| --- | --- | --- |
| 22M checkpoint branches | Batch 16 for learning-rate interventions | Conditional loss signs can change while temporal cosines stay negative |
| 130M training runs | Batch 32; 4096-token sequences; about 5.23B tokens | Conditional loss-boundary tracking with weak negative global alignment |
| 1B training run | Batch 512; 4096-token sequences; about 20B tokens | Stronger partial cancellation with improving loss; T1 was not measured |

### 130M: loss balance with weak negative alignment

![130M loss, conditional loss boundary and consecutive direction cosine](/assets/images/muon/figure1-130m.png)

*Figure 5. Peak learning rate 0.02. The conditional effective curvature repeatedly approaches the corrected boundary, while global directions stay far from complete reversal. From the bottom row of paper Figure 1.*

Across the four peak-rate runs, full-horizon cosine medians range from approximately -0.048 to -0.032. Loss neutrality comes from finite-reference **Muon-only checkpoint probes**; validation improvement belongs to complete **Muon-plus-AdamW training**. A near-zero probe increment does not mean the full run has stopped learning.

![130M module medians, block trajectories and probe-batch coherence in a grouped diagnostic figure](/assets/images/muon/figure-130m-diagnostics.png)

*Figure 6. Three complementary views of the 130M baseline at peak rate 0.02: (a) module-by-layer median cosine, (b) alignment trajectories, and (c) reference-gradient coherence for different probe batches. Panels reproduce paper Figures 6, 19(a) and 5(a), respectively; curves and color scales are unchanged.*

**Read across the panels:** the heatmap locates stronger negative alignment; the trajectories show how alignment evolves; the coherence curves show how sampling changes the loss-side signal. Middle-layer key/value matrices differ from many query matrices, and larger probe batches generally strengthen reference coherence.

### Is near orthogonality just minibatch randomness?

![Frozen-state direction comparisons and temporal cosines for two Tiny Transformer seeds](/assets/images/muon/figure11-frozen-vs-time.png)

*Figure 7. Orange: independent minibatch directions at a fixed state. Blue: actual consecutive-step directions. Orange bars are 95% delete-one-batch jackknife intervals. From paper Figure 11; Tiny Transformer, batch eight, two seeds.*

Frozen-state averages are small and positive; stable-stage temporal averages are negative. This supports sampling dispersion as a source of weak alignment and state changes as a source of negative bias. Since both Muon and AdamW change the state, the feedback cannot be assigned to either optimizer alone.

### 1B: stronger partial cancellation

![1B training and validation loss, global cosine and layer-level cosines](/assets/images/muon/figure8-1b.png)

*Figure 8. Loss keeps improving despite more negative global alignment; layer trajectories also differ. From paper Figure 8.*

The 1B run remains far from exact reversal. Its larger batch is consistent with the controlled trend, but a 130M-to-1B comparison changes multiple factors and cannot isolate batch size. **No T1 probes were collected for 1B.**

## 6. One checkpoint, three learning rates

We branch from rate 0.04 to 0.02, 0.04 or 0.08, keeping each model's batch size and auxiliary AdamW schedule fixed. The shared states are step 4000 for 22M and step 7000 for 130M.

The table pairs one-step conditional probes at step 4000 with temporal means during steps 5000–6399. These are different horizons; the cosine means have no reported intervals.

| Muon rate | Conditional loss increment | 95% probe interval | Later mean cosine |
| --- | --- | --- | --- |
| 0.02 | -0.01967 | [-0.01999, -0.01936] | -0.064 |
| 0.04 | -0.00026 | [-0.00138, 0.00086] | -0.100 |
| 0.08 | +0.15219 | [0.14650, 0.15789] | -0.117 |

**Loss crosses from decrease to increase; direction remains negative.** The intervals measure uncertainty from eight paired probes at one state, not across-seed training uncertainty. Cosine means summarize 280 recorded pairs per branch over the later constant-rate window. The 130M learning-rate branches have no conditional T1 probes.

![Validation loss and temporal cosine after 22M and 130M learning-rate changes](/assets/images/muon/figure7-lr-perturbation.png)

*Figure 9. Rate changes initially shift temporal alignment, which then tends back toward its pre-change value. Grey regions mark decay. From paper Figure 7. These validation curves are distinct from the one-step conditional probe values in the table above.*

Some larger-rate branches improve later validation loss. That motivates investigating schedules; it does not establish “raise the rate whenever cosine is near zero” as a rule.

## 7. Can these diagnostics help design a schedule?

![A proposed research workflow using loss-side and trajectory-side diagnostics](/assets/images/muon/diagram-joint-schedule.svg)

*Figure 10. A research workflow motivated by the separation, not a validated adaptive scheduler. Compare candidate learning rates or batches using both diagnostics, validation loss and token budgets.*

T1 measures one-step loss cost. Temporal alignment measures directional reinforcement or cancellation. Their joint response may inform WSD learning-rate choices and a data-driven critical batch size. Converting that idea into an optimized scheduler remains future work.

**Does Muon operate at the edge of stability? It retains an edge in a split form.** A stochastic loss-neutral boundary survives, but it does not impose a universal reversal signature. This is the sense in which Muon *sublates* EoS: it preserves part of the classical picture while changing its structure.

Momentum adds temporal filtering and optimizer state. The no-momentum results here should not be transferred unchanged to that setting.

---

Yanzhe Chen, Qifang Zhao, Xiaoxiao Xu, and Fanghui Liu. [*Muon Sublates the Edge of Stability in LLM Pretraining*](https://arxiv.org/abs/2609.34915), arXiv v1.

Figures 3–9 reproduce or combine the paper's measured results; the learning-rate table uses paper Table 2 and Appendix C.3. Figures 2 and 10 are explanatory diagrams created for this blog; Figure 1 is credited separately. Full assumptions, proofs and protocols are in the [paper](https://arxiv.org/abs/2609.34915) and [code](https://github.com/cyzebra/Muon-Sublates-the-Edge-of-Stability-in-LLM-Pretraining). Background: [Cohen et al. (2021)](https://openreview.net/forum?id=jh-rTtvkGeM) and [Jordan et al. on Muon](https://kellerjordan.github.io/posts/muon/).
