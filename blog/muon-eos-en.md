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

This post explains our [paper](https://arxiv.org/abs/2609.34915), joint work with Qifang Zhao, Xiaoxiao Xu, and Fanghui Liu. We focus on **no-momentum Muon**, from exact-polar theory to language-model experiments up to 1B parameters. The figures below tell the story; longer definitions are tucked into expandable notes.

[Paper](https://arxiv.org/abs/2609.34915) · [Code](https://github.com/cyzebra/Muon-Sublates-the-Edge-of-Stability-in-LLM-Pretraining)

## 1. First, what does classical EoS look like?

![Classical EoS: curvature near the learning-rate boundary and loss oscillations](/assets/images/muon/classical-eos-litman2026.png)

*Figure 1. Full-batch GD on a three-layer MLP, CIFAR-10, learning rate 0.5. Curvature approaches $$2/\eta$$; loss fluctuates locally while improving over longer horizons. The inset reveals detrended oscillations. Reproduced unchanged from [Litman (2026), Figure 1](https://arxiv.org/html/2604.20446v1#S1.F1), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

**Read the two panels together:** the left tracks a curvature boundary; the right shows that reaching it need not stop learning. Approaching the edge does not require exact equality at every step.

In one scalar quadratic mode, GD makes the connection especially clear. Its update multiplier is $$1-\eta\lambda$$. At $$\eta\lambda=2$$, the multiplier is -1: loss is unchanged, consecutive updates reverse with equal magnitude, and the linear dynamics is marginally stable.

## 2. What splits when we switch to Muon?

![GD and Muon compared through loss balance, temporal direction and perturbation stability](/assets/images/muon/diagram-split-edge.svg)

*Figure 2. An explanatory map, not an experimental plot. The GD column concerns one scalar quadratic mode. Under Muon, loss balance, direction and perturbation stability require separate questions. T2 means temporal orthogonality; it does not mean exact reversal.*

| Setting | Loss-neutral boundary | What happens to direction? |
| --- | --- | --- |
| GD: one scalar quadratic mode | $$\lambda=2/\eta$$ | Equal-magnitude reversal at the same boundary; sign reversal starts earlier, at $$\eta\lambda>1$$ |
| Full-batch exact-polar Muon | $$s^M=2/\eta$$ | Loss curvature does not determine the cosine sign |
| Stochastic exact-polar Muon | $$s_b^M=2\rho_b/\eta$$ | Boundary depends on state and batch coherence; temporal alignment is a separate diagnostic |

The quantity on the left changes across optimizers: Muon's effective loss curvature is not GD's largest Hessian eigenvalue. The shared scale $$2/\eta$$ in the full-batch loss identity does not make every dynamical signature share one threshold.

![An illustrative singular-value spectrum before and after the exact polar map](/assets/images/muon/diagram-polar-spectrum.svg)

*Figure 3. A conceptual spectrum example. Exact polar normalization retains the active singular vectors and sets their singular values to one. The illustrated values are not measured training data.*

This spectral transformation changes the update geometry. It is more than a uniform rescaling of the gradient.

## 3. Two signals to read independently

We use **T1** for conditional loss neutrality and **T2** for the zero-sign boundary of consecutive polar directions.

| Signal | What it measures | How to interpret it |
| --- | --- | --- |
| $$s_b^M-2\rho_b/\eta$$ | Expected one-step loss change at a fixed state | Negative: loss-decrease side; zero: neutrality; positive: loss-increase side |
| $$c^{\mathrm{pol}}$$ | Alignment of consecutive nonzero directions | Positive: acute; near zero: near orthogonal; negative: obtuse |
| $$c^{\mathrm{pol}}=0$$ | T2's sign boundary | Orthogonality, not a stability test |
| $$c^{\mathrm{pol}}=-1$$ | Exact directional reversal | Stronger than a negative cosine; a parameter return also requires matching update magnitudes and compatible update rules |

The loss-neutral edge is

$$
\boxed{s_b^M=\frac{2\rho_b}{\eta}}.
$$

Here, $$\rho_b$$ measures how much first-order descent the stochastic polar direction retains against the population or fixed full-data gradient. Effective curvature $$s_b^M$$ measures the normalized finite-step remainder along the actual update.

**The key distinction:** a negative temporal cosine tells us how two directions align. It cannot, by itself, tell us which side of T1 an update occupies.

<details markdown="1">
<summary>Technical details: coherence, curvature and the exact loss identity</summary>

Fix $$W_k$$, let $$G_k=\nabla L(W_k)\ne0$$, and resample a minibatch of size $$b$$. Exact-polar no-momentum Muon uses $$P_k=\operatorname{Pol}(\widehat G_k)$$ and $$W_{k+1}=W_k-\eta P_k$$. Define

$$
\rho_b(W_k)=\frac{\mathbb{E}_b\langle G_k,P_k\rangle_F}{\|G_k\|_*},
$$

$$
s_b^M(W_k;\eta)=\frac{2\mathbb{E}_b\left[L(W_k-\eta P_k)-L(W_k)+\eta\langle G_k,P_k\rangle_F\right]}{\eta^2\|G_k\|_*}.
$$

The nuclear norm $$\|G_k\|_*$$ is the sum of gradient singular values. Coherence lies in $$[-1,1]$$; an unbiased minibatch gradient need not induce an unbiased polar direction. These definitions give the exact conditional identity

$$
\mathbb{E}_b\left[L(W_{k+1})-L(W_k)\mid W_k\right]
=\frac{\eta^2\|G_k\|_*}{2}\left(s_b^M-\frac{2\rho_b}{\eta}\right).
$$

Full-batch exact-polar updates have $$\rho_{\mathrm{full}}=1$$. Under additional second-order smoothness, the small-step limit of $$s_b^M$$ is normalized directional Hessian curvature. Its finite-step definition is not simply a maximum Hessian eigenvalue.

For consecutive nonzero directions,

$$
c_k^{\mathrm{pol}}=\frac{\langle P_k,P_{k+1}\rangle_F}{\|P_k\|_F\|P_{k+1}\|_F}.
$$

A matrix quadratic proves the separation directly. At the same learning rate and full coherence, the paper constructs two states with:

| State | Effective loss curvature | Consecutive cosine |
| --- | --- | --- |
| A | $$5/(4\eta)$$ | $$-1$$ |
| B | $$5/(4\eta)$$ | $$1/3$$ |

Both decrease loss, but their direction signs differ. Curvature depends on a spectral trace; alignment depends on an eigenvalue sign count. The construction and assumptions appear in paper §2.3.

</details>

<details markdown="1">
<summary>Technical perspective: why GD is a special geometry</summary>

For a differentiable gradient transformation $$T$$, write an update map as $$F(W)=W-\eta T(\nabla L(W))$$. Wherever the derivatives exist, the chain rule gives the local parameter-perturbation operator

$$
DF(W)=I-\eta\,DT(G)\circ\nabla^2L(W).
$$

GD has $$T=\operatorname{Id}$$, so this reduces to $$I-\eta H$$. A nonlinear polar transformation adds update geometry through its derivative. This local linearization explains why perturbation stability is an additional question; our LLM T1/T2 measurements do not measure that operator's stability. Rank changes or nondifferentiable points require additional care.

</details>

## 4. Does batch size change the picture?

![Controlled Transformer loss, conditional loss balance and temporal direction across batches](/assets/images/muon/figure1-controlled.png)

*Figure 4. Exact SVD-polar Transformer, fixed learning rate 0.007. Compare the same batch's loss, loss-boundary tracking and direction cosine across the three panels. From the top row of paper Figure 1.*

**Look at the right panel:** larger batches produce more negative alignment in these runs; small batches stay closer to orthogonality. Full-batch trajectories move toward coherent reversal without reaching it in the displayed global measurements.

![First loss-rise, T1 and T2 events in controlled MLP, CNN and Transformer](/assets/images/muon/figure3-event-order.png)

*Figure 5. Recorded first events at learning rate 0.007. Red is T1 and blue is T2. Compare their order within each batch configuration. From paper Figure 3; event detection uses the paper's recorded tolerances.*

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

*Figure 6. Peak learning rate 0.02. The conditional effective curvature repeatedly approaches the corrected boundary, while global directions stay far from complete reversal. From the bottom row of paper Figure 1.*

Across the four peak-rate runs, full-horizon cosine medians range from approximately -0.048 to -0.032. Loss neutrality comes from finite-reference **Muon-only checkpoint probes**; validation improvement belongs to complete **Muon-plus-AdamW training**. A near-zero probe increment does not mean the full run has stopped learning.

![130M module-by-layer median direction cosine heatmap](/assets/images/muon/figure6-module-heatmap.png)

*Figure 7. Rows are layers; columns are matrix modules. Redder cells have more negative alignment. Middle-layer key/value matrices differ from many query matrices, so a global cosine hides heterogeneity. From paper Figure 6, constant-rate stage, peak rate 0.02.*

### Is near orthogonality just minibatch randomness?

![Frozen-state direction comparisons and temporal cosines for two Tiny Transformer seeds](/assets/images/muon/figure11-frozen-vs-time.png)

*Figure 8. Orange: independent minibatch directions at a fixed state. Blue: actual consecutive-step directions. Orange bars are 95% delete-one-batch jackknife intervals. From paper Figure 11; Tiny Transformer, batch eight, two seeds.*

Frozen-state averages are small and positive; stable-stage temporal averages are negative. This supports sampling dispersion as a source of weak alignment and state changes as a source of negative bias. Since both Muon and AdamW change the state, the feedback cannot be assigned to either optimizer alone.

### 1B: stronger partial cancellation

![1B training and validation loss, global cosine and layer-level cosines](/assets/images/muon/figure8-1b.png)

*Figure 9. Loss keeps improving despite more negative global alignment; layer trajectories also differ. From paper Figure 8.*

The 1B run remains far from exact reversal. Its larger batch is consistent with the controlled trend, but a 130M-to-1B comparison changes multiple factors and cannot isolate batch size. **No T1 probes were collected for 1B.**

## 6. One checkpoint, three learning rates

We branch from rate 0.04 to 0.02, 0.04 or 0.08, keeping each model's batch size and auxiliary AdamW schedule fixed. The shared states are step 4000 for 22M and step 7000 for 130M.

![Measured conditional loss responses and later temporal cosine means in 22M learning-rate branches](/assets/images/muon/chart-lr-probe-response.svg)

*Figure 10. Replotted from paper Table 2 and Appendix C.3, without new measurements. Left: one-step conditional probes at step 4000, with 95% intervals. Right: temporal means during steps 5000–6399. These are different horizons; the cosine means have no reported intervals.*

| Muon rate | Conditional loss increment | 95% probe interval | Later mean cosine |
| --- | --- | --- | --- |
| 0.02 | -0.01967 | [-0.01999, -0.01936] | -0.064 |
| 0.04 | -0.00026 | [-0.00138, 0.00086] | -0.100 |
| 0.08 | +0.15219 | [0.14650, 0.15789] | -0.117 |

**Loss crosses from decrease to increase; direction remains negative.** The intervals measure uncertainty from eight paired probes at one state, not across-seed training uncertainty. Cosine means summarize 280 recorded pairs per branch over the later constant-rate window. The 130M learning-rate branches have no conditional T1 probes.

![Validation loss and temporal cosine after 22M and 130M learning-rate changes](/assets/images/muon/figure7-lr-perturbation.png)

*Figure 11. Rate changes initially shift temporal alignment, which then tends back toward its pre-change value. Grey regions mark decay. From paper Figure 7. These validation curves are distinct from the one-step conditional probes in Figure 10.*

Some larger-rate branches improve later validation loss. That motivates investigating schedules; it does not establish “raise the rate whenever cosine is near zero” as a rule.

## 7. Can these diagnostics help design a schedule?

![A proposed research workflow using loss-side and trajectory-side diagnostics](/assets/images/muon/diagram-joint-schedule.svg)

*Figure 12. A research workflow motivated by the separation, not a validated adaptive scheduler. Compare candidate learning rates or batches using both diagnostics, validation loss and token budgets.*

T1 measures one-step loss cost. Temporal alignment measures directional reinforcement or cancellation. Their joint response may inform WSD learning-rate choices and a data-driven critical batch size. Converting that idea into an optimized scheduler remains future work.

**Does Muon operate at the edge of stability? It retains an edge in a split form.** A stochastic loss-neutral boundary survives, but it does not impose a universal reversal signature. This is the sense in which Muon *sublates* EoS: it preserves part of the classical picture while changing its structure.

Momentum adds temporal filtering and optimizer state. The no-momentum results here should not be transferred unchanged to that setting.

---

Yanzhe Chen, Qifang Zhao, Xiaoxiao Xu, and Fanghui Liu. [*Muon Sublates the Edge of Stability in LLM Pretraining*](https://arxiv.org/abs/2609.34915), arXiv v1.

Figures 4–9 and 11 reproduce the paper's measured results. Figure 10 visualizes reported numbers. Figures 2, 3 and 12 are explanatory diagrams created for this blog; Figure 1 is credited separately. Full assumptions, proofs and protocols are in the [paper](https://arxiv.org/abs/2609.34915) and [code](https://github.com/cyzebra/Muon-Sublates-the-Edge-of-Stability-in-LLM-Pretraining). Background: [Cohen et al. (2021)](https://openreview.net/forum?id=jh-rTtvkGeM) and [Jordan et al. on Muon](https://kellerjordan.github.io/posts/muon/).
