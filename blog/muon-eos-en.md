---
layout: default
title: "Does Muon Operate at the Edge of Stability in LLM Pretraining?"
description: "Our answer is sublation: no-momentum Muon preserves a stochastic loss edge while separating loss balance from update reversal, motivating joint learning-rate and batch-size diagnostics."
permalink: /blog/muon-eos-en/
lang: en
math: true
---

[Home](/) · [中文版](/blog/muon-eos-zh/)

# Does Muon Operate at the Edge of Stability in LLM Pretraining?

Yanzhe Chen · Research blog

**Does replacing gradient descent with Muon preserve the familiar edge of stability in language-model pretraining? Our answer is captured by one word in our paper's title: sublates.**

In the no-momentum setting we study, Muon preserves a stochastic, coherence-corrected loss-neutral edge while separating loss balance from update reversal. Understanding whether training is “at the edge” therefore requires looking at loss and temporal direction separately.

This post explains our paper [*Muon Sublates the Edge of Stability in LLM Pretraining*](https://arxiv.org/abs/2609.34915), joint work with Qifang Zhao, Xiaoxiao Xu, and Fanghui Liu. We combine exact-polar theory, controlled MLP/CNN/Transformer experiments, and language-model experiments at 22M, 130M, and 1B parameters. These experiments provide complementary evidence: 130M runs measure loss balance and direction, the 1B run extends the directional observations, and learning-rate interventions test their separate responses.

The story follows three questions: **What survives of classical EoS? Does loss balance still imply reversal? Can the distinction inform pretraining schedules?**

[Paper](https://arxiv.org/abs/2609.34915) · [Code](https://github.com/cyzebra/Muon-Sublates-the-Edge-of-Stability-in-LLM-Pretraining)

## 1. What does “at the edge of stability” mean?

Large-learning-rate training often makes progress without decreasing loss at every step. Short-term fluctuations coexist with improvement over longer horizons. The classical edge of stability (EoS) connects this behavior to a learning-rate-dependent curvature boundary. For background, see [Cohen et al. (ICLR 2021)](https://openreview.net/forum?id=jh-rTtvkGeM).

![Classical EoS: curvature approaches the learning-rate boundary while loss oscillates and improves](/assets/images/muon/classical-eos-litman2026.png)

*Figure 1. A classical EoS example: effective curvature (blue) and maximum Hessian eigenvalue (green) approach $$2/\eta$$ (purple dashed); loss and its five-step mean show continued improvement, with detrended oscillations in the inset. Three-layer MLP on CIFAR-10, full-batch GD, learning rate 0.5. Reproduced without modification from Elon Litman (2026), [The Origin of Edge of Stability, Figure 1](https://arxiv.org/html/2604.20446v1#S1.F1), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

The green curve exceeds the dashed line slightly: approaching the edge does not require exact equality at every step. Short-term oscillations and long-term learning coexist.

This is the phenomenon we want to investigate under Muon: training approaches loss balance while continuing to learn. To understand what survives of the classical picture, we first need to see why loss balance and update reversal are linked in GD.

## 2. Why do loss balance and reversal meet in GD?

Consider gradient descent on

$$
f(x)=\frac{\lambda}{2}x^2,
\qquad
x_{k+1}=(1-\eta\lambda)x_k,
$$

where $$\lambda>0$$ is curvature and $$\eta$$ is the learning rate. At a point with nonzero loss,

$$
\frac{f(x_{k+1})}{f(x_k)}=(1-\eta\lambda)^2.
$$

Loss decreases for $$0<\eta\lambda<2$$, is unchanged at $$\eta\lambda=2$$, and increases beyond it. At the boundary,

$$
x_{k+1}=-x_k,
\qquad
x_{k+2}=x_k.
$$

The parameter and consecutive updates reverse with equal magnitude. One-step loss neutrality, equal-magnitude reversal, and marginal linear stability therefore coincide in this mode.

Direction reversal starts earlier, at $$\eta\lambda>1$$. Between 1 and 2, the sign alternates while the magnitude shrinks and loss still decreases. The special property at 2 is equal-magnitude reversal.

This calculation illustrates the classical coupling; it does not assert exact two-step cycles in arbitrary neural networks. Our question is whether Muon retains the mechanism that forces these signals together.

## 3. What changes when we replace GD with Muon?

Focus on one matrix parameter block and remove momentum. For a minibatch gradient with compact SVD,

$$
\widehat G_k=U_r\Sigma_rV_r^\top,
$$

the exact polar update is

$$
P_k=\operatorname{Pol}(\widehat G_k)=U_rV_r^\top,
\qquad
W_{k+1}=W_k-\eta P_k.
$$

Polar normalization sets the active singular values to one while retaining the associated singular vectors. It changes the spectral structure of the update, rather than merely multiplying the entire gradient by a scalar.

We consequently ask two separate questions:

- What is the conditional expected loss change of the current update?
- How does the next update align with the current one?

T1 describes loss neutrality. A consecutive-direction cosine describes temporal alignment, with T2 denoting its zero-sign boundary.

## 4. Does a loss-neutral edge survive?

Fix the current parameters $$W_k$$ and resample a minibatch of size $$b$$. Let $$L$$ be the population or fixed full-data objective, with $$G_k=\nabla L(W_k)\ne0$$. The stochastic polar direction need not align with this gradient as well as its full-batch counterpart.

We measure retained first-order descent through coherence:

$$
\rho_b(W_k)=
\frac{\mathbb{E}_b\langle G_k,P_k\rangle_F}{\|G_k\|_*}.
$$

The denominator is the nuclear norm, the sum of gradient singular values; the numerator is the expected Frobenius inner product at the fixed state. Coherence lies in $$[-1,1]$$. An unbiased minibatch gradient need not produce an unbiased polar direction, because the polar map is nonlinear.

We also define a normalized finite-step loss remainder:

$$
s_b^M(W_k;\eta)=
\frac{2\mathbb{E}_b\left[
L(W_k-\eta P_k)-L(W_k)+\eta\langle G_k,P_k\rangle_F
\right]}{\eta^2\|G_k\|_*}.
$$

This quantity captures the loss change left after accounting for the first-order prediction. Under additional second-order smoothness, its small-step limit is normalized directional Hessian curvature. At finite step size, it is not simply the largest Hessian eigenvalue.

Together, these definitions give the exact conditional identity

$$
\mathbb{E}_b\left[L(W_{k+1})-L(W_k)\mid W_k\right]
=\frac{\eta^2\|G_k\|_*}{2}
\left(s_b^M-\frac{2\rho_b}{\eta}\right).
$$

The bracket determines the sign: below the boundary, conditional expected loss decreases; above it, loss increases. The T1 boundary is

$$
\boxed{s_b^M=\frac{2\rho_b}{\eta}}.
$$

Full-batch exact-polar updates have $$\rho_{\mathrm{full}}=1$$ and recover $$2/\eta$$. The left-hand side remains Muon's effective loss curvature, not GD's largest Hessian eigenvalue transplanted into a different optimizer.

This answers the first part of our question: a precise loss-neutral edge survives, corrected for stochastic coherence. It is the preserved part of sublation. We still need to ask whether the same edge forces update reversal.

## 5. Does reaching loss neutrality imply reversal?

For consecutive nonzero polar directions, define

$$
c_k^{\mathrm{pol}}=
\frac{\langle P_k,P_{k+1}\rangle_F}
{\|P_k\|_F\|P_{k+1}\|_F}.
$$

Positive values mean acute alignment, values near zero mean near orthogonality, and negative values mean obtuse alignment. Only $$-1$$ means exact reversal. T2 is the sign boundary $$c_k^{\mathrm{pol}}=0$$; it is not itself a stability criterion.

A matrix quadratic in the paper demonstrates why loss curvature cannot determine this sign. At the same learning rate and full coherence, two states have

| State in the matrix quadratic | Effective loss curvature | Consecutive-direction cosine |
| --- | --- | --- |
| A | $$5/(4\eta)$$ | $$-1$$ |
| B | $$5/(4\eta)$$ | $$1/3$$ |

Both are on the loss-decrease side, yet one reverses exactly and the other has positive alignment. Curvature depends on a spectral trace, while alignment depends on an eigenvalue sign count. See §2.3 of the paper for the construction.

Controlled experiments also show that the first T1 and T2 events can occur in either order. A negative cosine alone therefore identifies neither a loss-increase step nor a two-step return. The latter requires stronger conditions, including exact reversal and equal update magnitudes; practical LLM training includes approximation, scaling, and other parameter updates.

![Loss, loss balance, and direction alignment in a controlled Transformer](/assets/images/muon/figure1-controlled.png)

*Figure 2. Controlled Transformer experiments use exact SVD-polar updates at learning rate 0.007. Columns show loss, effective curvature versus the coherence-corrected boundary, and consecutive-direction cosine. Batch size influences both loss and trajectory diagnostics. Adapted from the top row of paper Figure 1; the rows are separated here for readability.*

## 6. Does the separation persist in LLM pretraining?

Our theory isolates exact-polar no-momentum updates. The LLM experiments use practical no-momentum NS-5 approximation on Muon blocks and AdamW on auxiliary parameters. Keeping this distinction explicit matters when interpreting the measurements.

### 130M: loss balance with weak negative alignment

The 130M Llama-like model uses batch size 32, sequence length 4096, and about 5.23B training tokens. Four warmup–stable–decay (WSD) runs use peak Muon learning rates 0.02, 0.04, 0.08, and 0.12.

Conditional effective curvature repeatedly lies near the coherence-corrected boundary. Full-horizon direction-cosine medians range from approximately -0.048 to -0.032, far from exact reversal at -1. Meanwhile, validation loss of complete training improves over longer horizons.

Loss neutrality is measured with finite-reference, Muon-only probes at fixed checkpoints. Validation improvement belongs to complete Muon-plus-AdamW training. A near-zero conditional probe increment does not mean that the entire training run has stopped learning.

![Loss, conditional loss balance, and direction cosine in 130M pretraining](/assets/images/muon/figure1-130m.png)

*Figure 3. The 130M baseline uses peak learning rate 0.02. Conditional boundary tracking coexists with weak negative global alignment. Adapted from the bottom row of paper Figure 1.*

### Near orthogonality: sampling and state changes

High-dimensional random vectors are often nearly orthogonal, so weak negative alignment alone does not identify its cause. In a Tiny Transformer, we compare directions sampled at a frozen parameter state with actual consecutive-step directions.

Frozen-state mean cosines are small and positive; stable-stage temporal means are negative. These experiments support minibatch dispersion as a source of weak alignment and state changes as a source of negative bias. Since both Muon and auxiliary AdamW change the state, this feedback cannot be assigned to either optimizer alone. The experiments support this interpretation rather than a complete decomposition across every model and setting.

### 1B: stronger partial cancellation

The 1B run uses batch size 512, sequence length 4096, and about 20B tokens. Its global alignment is more negative than in the 130M runs, but still far from exact reversal, while loss continues to improve.

This is consistent with the larger-batch trend in controlled experiments. However, the cross-scale comparison also changes the model and training configuration, so it does not isolate a causal batch-size effect.

The 1B run measures loss and direction, but not T1, due to compute limits. It supplies larger-scale directional evidence, not a measurement of 1B loss-boundary tracking.

![Loss, global cosine, and layer-level cosines in 1B pretraining](/assets/images/muon/figure8-1b.png)

*Figure 4. Loss improves despite substantial partial cancellation. Layer-level curves show that a global cosine also hides heterogeneous behavior. From paper Figure 8.*

## 7. Do loss and direction respond together to learning-rate changes?

Branching from a shared checkpoint provides a more direct test than comparing unrelated runs. We change the Muon learning rate from 0.04 to 0.02, 0.04, or 0.08 at step 4000 for 22M and step 7000 for 130M. Each model's training batch size and auxiliary AdamW schedule remain unchanged across its branches.

At the shared 22M state, paired conditional probes show that halving the rate places the step on the loss-decrease side of T1, whereas doubling it places the step on the loss-increase side. Temporal cosines remain negative in all three branches before decay. Conditional loss increments move back toward zero at later checkpoints.

Paper Table 2 reports the following responses at step 4000 with probe batch 16. Each increment evaluates a Muon-only virtual update on the same finite reference objective:

| Muon learning rate | Mean one-step conditional loss increment | 95% probe interval |
| --- | --- | --- |
| 0.02 | -0.01967 | [-0.01999, -0.01936] |
| 0.04 | -0.00026 | [-0.00138, 0.00086] |
| 0.08 | +0.15219 | [0.14650, 0.15789] |

These intervals describe probe uncertainty at one state, not across-seed training uncertainty. The 130M learning-rate branches have no conditional T1 probes; they supply complementary evidence about validation loss and directional responses.

The test is simple: **the alignment sign can remain negative while the conditional loss-change sign differs.**

![Validation loss and direction cosine after learning-rate interventions](/assets/images/muon/figure7-lr-perturbation.png)

*Figure 5. Raising the learning rate initially makes alignment more negative; lowering it makes alignment less negative. Both tend to move back toward their pre-intervention values. Grey shading marks decay. These panels show validation loss and temporal cosine; the one-step conditional loss-sign evidence comes from paired probes reported in the paper, not directly from the validation curves. From paper Figure 7.*

In these configurations, a larger peak rate can also improve later validation loss. This observation does not establish a rule that one should keep raising the learning rate whenever the cosine is close to zero.

## 8. Can this help design learning-rate and batch-size schedules?

Loss and trajectory diagnostics should be monitored together. T1 describes conditional one-step loss cost; temporal alignment describes how consecutive directions reinforce or partially cancel one another. Negative alignment alone gives too little information to decide that a learning rate is excessive.

The separation motivates future WSD schedules that check the loss-side margin before increasing the rate. It also suggests examining critical batch size through the joint response of coherence and alignment, rather than gradient noise alone. Our paper does not yet introduce or validate an optimized adaptive scheduler.

Momentum is another open direction. It adds temporal filtering and optimizer state, so the no-momentum thresholds and directional patterns should not be assumed to transfer unchanged.

Returning to our opening question, **does Muon operate at the edge of stability in LLM pretraining?** It retains an edge in a split form: a stochastic loss-neutral boundary survives, but reaching it does not require a universal reversal signature. We call this a **split edge of stability**.

Borrowing the term from Hegelian philosophy, this is what *sublates* means in our title: preserving part of an earlier picture while changing its structure. Experiments from controlled networks to language models support measuring loss and direction separately, providing a starting point for future joint learning-rate and batch-size schedules.

---

Yanzhe Chen, Qifang Zhao, Xiaoxiao Xu, and Fanghui Liu. *Muon Sublates the Edge of Stability in LLM Pretraining*. [arXiv:2609.34915](https://arxiv.org/abs/2609.34915).

This post follows the paper's arXiv v1 version. Research-result figures are reproduced from the paper; the classical EoS image is credited in Figure 1, and experimental measurements were produced by its authors. See the paper and [code repository](https://github.com/cyzebra/Muon-Sublates-the-Edge-of-Stability-in-LLM-Pretraining) for definitions, proofs, probe protocols, and training configurations. For background on Muon, see [the introduction by Keller Jordan and collaborators](https://kellerjordan.github.io/posts/muon/).
