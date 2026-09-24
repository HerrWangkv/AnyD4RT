5
2
0
2

c
e
D
0
3

]

G
L
.
s
c
[

1
v
8
3
1
4
2
.
2
1
5
2
:
v
i
X
r
a

GARDO: Reinforcing Diffusion Models without Reward Hacking

Haoran He1,2 Yuxiao Ye1 Jie Liu3 Jiajun Liang2 Zhiyong Wang4
Ziyang Yuan2 Xintao Wang2 Hangyu Mao2 Pengfei Wan2 Ling Pan1
1Hong Kong University of Science and Technology 2Kuaishou Technology 3CUHK MMLab 4The University of Edinburgh
haoran.he@connect.ust.hk

Abstract

1. Introduction

Fine-tuning diffusion models via online reinforcement
learning (RL) has shown great potential for enhancing
text-to-image alignment. However, since precisely specify-
ing a ground-truth objective for visual tasks remains chal-
lenging, the models are often optimized using a proxy re-
ward that only partially captures the true goal. This mis-
match often leads to reward hacking, where proxy scores
increase while real image quality deteriorates and gen-
eration diversity collapses. While common solutions add
regularization against the reference policy to prevent re-
ward hacking, they compromise sample efficiency and im-
pede the exploration of novel, high-reward regions, as
the reference policy is usually sub-optimal. To address
the competing demands of sample efficiency, effective ex-
ploration, and mitigation of reward hacking, we propose
Gated and Adaptive Regularization with Diversity-aware
Optimization (GARDO), a versatile framework compatible
with various RL algorithms. Our key insight is that reg-
ularization need not be applied universally; instead, it is
highly effective to selectively penalize a subset of samples
that exhibit high uncertainty. To address the exploration
challenge, GARDO introduces an adaptive regularization
mechanism wherein the reference model is periodically up-
dated to match the capabilities of the online policy, ensuring
a relevant regularization target. To address the mode col-
lapse issue in RL, GARDO amplifies the rewards for high-
quality samples that also exhibit high diversity, encouraging
mode coverage without destabilizing the optimization pro-
cess. Extensive experiments across diverse proxy rewards
and hold-out unseen metrics consistently show that GARDO
mitigates reward hacking and enhances generation diver-
sity without sacrificing sample efficiency or exploration,
highlighting its effectiveness and robustness. Our project
is available at https://tinnerhrhe.github.io/
gardo_project

Denoising diffusion and flow models trained on large-scale
datasets have found great success in text-to-image genera-
tion tasks, exhibiting unprecedented capabilities in visual
quality [25, 42, 59, 62]. While supervised pre-training pro-
vides these models a strong prior, it is often insufficient
for ensuring alignment with human preferences [38, 50].
Reinforcement Learning (RL) has emerged as a predom-
inant paradigm to address this gap, either employing on-
line policy-gradient approaches [4, 10, 15, 30, 35, 64] or
direct reward backpropagation [7, 63]. These methods typ-
ically assume a reward model r(x) that captures human
preferences, and the training objective is to maximize the
rewards over generated samples, i.e., Ex∼π[r(x)]. Conse-
quently, the accuracy of r(x) plays a critical role in fine-
tuning performance. However, a key challenge arises from
the nature of reward specification in vision. In contrast to
the language domain, where rewards are often verifiable
(e.g., the correctness of mathematical solutions or com-
piled code), rewards for visual tasks are typically more com-
plex. Such rewards typically fall into two categories, both
of which are imperfect proxies for genuine human prefer-
ence: 1) Model-based rewards such as ImageReward [63],
UnifiedReward [55], and HPSv3 [34] are trained on fi-
nite human preference datasets to approximate a ground-
truth “genuine” reward, indicating that they are accurate
only within their training data distribution; 2) Rule-based
rewards such as object detection and text rendering (i.e.,
OCR) [5] are limited to evaluating specific attributes, and
thus fail to capture the overall qualities of generated sam-
ples. These limitations expose a significant distribution
shift [57] vulnerability: fine-tuning methods can rapidly
over-optimize the proxy reward by generating images that
fall outside the trusted support of the reward model. This
often yields spurious reward signals and thereby leads to
reward hacking [49], where proxy rewards increase while
the actual quality of the images degrades.

To address this challenge, prior work has employed KL-
based regularization during fine-tuning to mitigate over-
optimization on these spurious reward signals [10, 30, 51].

 
 
 
 
 
 
However, as the online policy improves, the divergence
from the static reference model can cause the KL penalty
to dominate the RL loss. This often leads to diminishing
policy updates, thereby impeding sample efficiency. More-
over, by design, the regularization constrains the online pol-
icy to remain in proximity to the reference model, which is
often suboptimal. This can stifle effective exploration and
prevent the discovery of emerging behaviors that are absent
from the reference model.

Can we prevent reward hacking without compromising
sample efficiency and effective exploration? In this pa-
per, we propose Gated and Adaptive Regularization with
Diversity-aware Optimization (GARDO), a novel frame-
work designed to overcome these limitations. Our gated
regularization mechanism is motivated by the core insight:
KL penalty is not universally required; Theoretically,
only samples assigned with spurious rewards need regular-
ization to prevent hacking. Accordingly, our gated mech-
anism applies the KL penalty selectively, targeting only a
small subset (e.g., ≈10%) of samples within each batch that
exhibit the highest reward uncertainty [2]. We quantify this
uncertainty by measuring the disagreement among an en-
semble of reward functions, which serves as an effective
proxy for the trustworthiness of the reward signal. High
disagreement signifies that a generated sample likely falls
into an out-of-distribution region where the reward models
are extrapolating unreliably. To further accelerate training
and sustain exploration, GARDO incorporates an adaptive
regularization target. We periodically update the reference
model with a recent snapshot of the online policy, facili-
tating continual improvement while retaining the stabiliz-
ing benefits of KL regularization. Finally, to enhance mode
coverage, we introduce a diversity-aware optimization strat-
egy. This is achieved by carefully amplifying the advantage
term for high-quality samples that also exhibit high diver-
sity. This amplification is carefully calibrated to ensure
it neither dominates nor reverses the sign of the original
advantage, which encourages broader exploration without
destabilizing the optimization.

Our contributions are summarized as follows:

(i) We
provide a systematic analysis of the reward hacking phe-
nomenon in RL-based fine-tuning for text-to-image align-
identifying the core limitations of existing regu-
ment,
larization techniques.
(ii) We make a novel innovation
on traditional KL regularization. First, a gated regular-
ization method selectively applies penalties only to high-
uncertainty samples, thereby allowing the majority of sam-
ples to be optimized freely towards high-reward regions.
Second, an adaptive regularization target facilitates contin-
ual improvement and sustained exploration by dynamically
(iii) We propose a robust
updating the reference anchor.
and generalist diversity-aware approach to improve genera-
tion diversity and mode coverage for diffusion RL. (iv) Ex-

tensive experiments on multiple tasks across different prox-
ies and unseen metrics demonstrate both the efficiency and
effectiveness of our proposed method. Our work demon-
strates that it is possible to successfully balance the com-
peting demands of sample efficiency, exploration, and di-
versity, all while robustly mitigating reward hacking during
the fine-tuning of image generative models.

2. Related Works

Fine-Tuning Diffusion Models via Rewards. Recent re-
search has increasingly focused on fine-tuning pre-trained
diffusion models using reward signals derived from human
feedback. A reward model (RM) is trained on a finite
dataset, learning to assign a scalar score that approximates
true human preference [31, 56, 60, 63]. Typical methods
include policy-gradient RL [3, 11, 36], direct preference
optimization (DPO) [29, 31, 33, 52, 65, 68, 69], and di-
rect reward backpropagation [6, 40, 63]. Recently, Flow-
GRPO [30] and DanceGRPO [64] have adapted GRPO
for finetuning cutting-edge flow matching models, showing
strong performance and inspiring subsequent research [12,
19, 27, 53, 54, 67, 71].
Reward Hacking. Fine-tuning with either model-based or
rule-based reward models for alignment of visual genera-
tion is highly susceptible to reward hacking [26, 49]. This
issue arises because the reward model is an imperfect proxy
for the true human preference. As a generative model op-
timizes against this proxy, it often discovers adversarial so-
lutions that exploit the reward model’s flaws, such as favor-
ing extreme saturation or visual noise [6, 31], to achieve
a high score, despite failing to satisfy human intent. To
address this, standard methods like DPOK [10] and Flow-
GRPO [30] employ KL regularization to constrain the pol-
icy update. While this can prevent the most egregious forms
of reward hacking, it often sacrifices convergence speed and
hinders effective exploration. Another line of work, exem-
plified by RewardDance [60], focuses on improving the re-
ward model itself through scaling; however, this does not
eliminate the vulnerability to out-of-distribution samples.
Concurrently, Pref-GRPO [54] utilizes a pairwise prefer-
ence model to provide a more robust reward signal and
avoid illusory advantages. This approach, however, is not
universally applicable as it is limited to preference-based
reward models and incurs substantial computational costs
for pairwise comparisons.

3. Preliminaries

3.1. Denoising as an MDP

Both diffusion models and flow models map the source dis-
tribution, often a standard Gaussian distribution, to a true
data distribution p0. As shown in previous works [17, 22,
48], these models can utilize an SDE-based sampler dur-

Inspired by
ing inference to restore from diffused data.
DDPO [4], we formulate the multi-step denoising process in
flow and diffusion-based models as a Markov Decision Pro-
cess (MDP), defined by a tuple (S, A, R, P, ρ0). At each
denoising step t, the model receives a state st ≜ (c, t, xt),
and predicts the action at ≜ xt−1 based on the policy
π(at|st) ≜ pθ(xt−1|xt, c). A non-zero reward R(st, at) ≜
r(x0, c) is given only at the final step, where R = 0 if t ̸= 0.
The transition function P is deterministic. At the begin-
ning of each episode, the initial state sT is sampled from
the initial state distribution ρ0. The goal is to learn a policy
(cid:2)R(s0, a0)] by maximizing the
π∗ = arg maxπ Eρ0,at∼π(st)
expected cumulative reward R.

3.2. RL on Diffusion and Flow Models

Reinforcement Learning (RL) [4, 10] has been widely used
for enhancing sample quality by steering outputs towards a
desired reward function. Unlike earlier approaches based
on PPO [45] or REINFORCE [58], recent methods [30, 64]
have demonstrated greater success using GRPO [46]. We
take GRPO as the base RL algorithm throughout our paper.
GRPO rollouts {xi
i=1 samples conditioned on the same
input c, and estimates the advantage within each group:

0}G

Ai

t =

R(xi

0) − mean({R(xi
i=1)

std({R(xi

0)}G

0)}G

i=1)

, ∀t.

(1)

Figure 1. Setting OCR as the proxy reward, vanilla RL methods
like Flow-GRPO exploit the OCR reward at the cost of losing real
image quality, leading to reward hacking. It generates unrealis-
tic, noisy images with blurry backgrounds and visual artifacts. In
contrast, our method maintains better image quality and diversity.
Prompt:“A storefront with ‘GARDO’ written on it”.

Definition 1 (Reward hacking [26]). Suppose ˜R is a proxy
reward used during RL fine-tuning, and R denotes the true
reward. ˜R is a hackable proxy with respect to πref s.t.

J(π, R) < J(πref , R) for some π ∈ arg max

J(π, ˜R),

π

where J(π, R) = Eπ[(cid:80)
πref is the reference policy.

(3)
t R(st, at)] is the RL objective, and

Then the following surrogate objective optimizes πθ:

J(θ) = Ex0∼πθold

(cid:2) 1
G

G
(cid:88)

i=1

1
T

(cid:88)T

t=0

(cid:0) min (cid:0)ri

tAi
t,

(2)

clip(ri

t, 1 − ϵ, 1 + ϵ)Ai
t

(cid:1) − βDKL(πθ|πref )(cid:1)(cid:3),

with ri
t = πθ(at|st)/πθold (at|st) the importance sampling
ratio, πθold the behavior policy to sample data, ϵ the clipping
range of rt, and DKL the KL regularization term.

4. Method

We first present the reward hacking issues in image gener-
ation as a motivated example. Subsequently, we provide a
theoretical analysis, and propose practical methods to ad-
dress the problem. Finally, we provide a didactic example
for empirical validation to illustrate our method clearly.

4.1. Reward Hacking in RL-Based Fine-tuning

We investigate RL in the context of fine-tuning text-to-
image models. Here, the reward function is often a neu-
ral network trained on finite preference data or a rule-based
metric, which serves as just a proxy for the true reward.
Misalignment between the two objectives can lead to re-
ward hacking: a learned policy performs well according to
the proxy reward but not according to the true reward [39].

As demonstrated in Figure 1, standard methods like
Flow-GRPO [30] maximize the proxy score (i.e., OCR) at
the expense of perceptual quality, failing to align with the
true human preferences. Previous methods [51] address this
issue by incorporating a KL divergence regularization term,

Jβ(πθ) = Eπθ

(cid:2) (cid:88)

t

˜R(st, at)(cid:3) − βDKL(πθ|πref ),

(4)

where β controls the regularization strength. This kind
of approach mitigates reward hacking (or reward over-
optimization) by explicitly penalizing significant deviations
of the online policy, π, from the reference policy, πref .
However, this strategy introduces two significant limita-
tions. First, because the reference policy πref is typically
suboptimal, the regularization loss can impede learning,
leading to poor sample efficiency. Second, the constant
penalty constrains πθ to exploit the “safe” region around
πref , preventing it from discovering novel behaviors or so-
lutions that are absent from the reference model.

Ideally, for fine-tuning image generative models via RL,
our goal is to reconcile the following competing objec-
tives: (i) optimizing πθ with high sample efficiency, (ii)
avoiding reward hacking and over-optimization, (iii) en-
couraging exploration to high-reward, novel modes which

Flow-GRPO+GARDO(Ours)Figure 2. Overview of GARDO. GARDO introduces an uncertainty-driven, gated KL mechanism to control the proportion of regulariza-
tion, avoiding unnecessary penalties. Our proposed diversity-aware advantage shaping effectively encourages exploration of novel states.

may be missing from πref , and (iv) preserving genera-
tion diversity. Conventional KL regularization is effec-
tive in accomplishing (ii) and (iv), but often at the ex-
pense of (i) and (iii). This inherent trade-off motivates
our work. Based on our theoretical findings, we propose
Gated and Adaptive Regularization with Diversity-aware
Optimization (GARDO), a novel framework that is simple,
effective, and capable of satisfying all these objectives si-
multaneously. Overview of GARDO is illustrated in Fig. 2,
and we detail our method in the subsequent sections.

4.2. Gated and Adaptive Regularization

First, we investigate the reasons of reward hacking. As dis-
cussed in previous works [16, 50, 61], the optimal solution
to the KL-regularized objective in Eq (4) can be written as

p∗(x) =

1
Z

πref (x) exp

(cid:32) ˜R(x)
β

(cid:33)

,

(5)

where Z is an intractable normalization constant [41]. Here,
p∗(x) is largely defined by both the reference model πref (x)
and the proxy reward ˜R(x).

Proposition 1 The probability ratio between any two sam-
ples, x1 and x2, under the optimal solution distribution de-
fined in Eq. (5) is given by the following closed form,

p∗(x1)
p∗(x2)

=

πref (x1)
πref (x2)

exp

(cid:32) ˜R(x1) − ˜R(x2)
β

(cid:33)

.

(6)

If πref (x1) = πref (x2), then their probability difference is
defined solely by the proxy reward, i.e., p∗(x1)/p∗(x2) =
exp(( ˜R(x1) − ˜R(x2))/β). Reward hacking arises when the
proxy reward is misaligned with the true reward, R. For
example, if ˜R(x1) > ˜R(x2) while R(x1) < R(x2), the
model will incorrectly assign a higher sampling probability

to the lower-quality sample, x1. However, in other cases,
such as ‘x1 > x2’ holds for both ˜R and R, the proxy is
reliable and does not lead to reward hacking. Inspired by
this finding, we have the following key insight:

Takeaway 1

Regularization is not universally required. It is required
only for samples with spurious proxy reward ˜R.

The application of KL regularization should be condi-
tional on the alignment between proxy and true rewards.
Universally penalizing all samples introduces unnecessary
regularization signals, thereby impeding convergence.

Gated KL Mechanism. KL is gated for selective samples.
We propose an uncertainty-driven approach to select sam-
ples for regularization during the training process [1]. Only
samples that exhibit high uncertainty U will be penalized,
where U reflects the trustworthiness of the proxy reward [8].
Our approach for quantifying uncertainty is inspired by
prior work in reinforcement learning [2], where the uncer-
tainty is often estimated using the disagreement within an
ensemble learned K bootstrapped Q-functions Qk [37], i.e.,
U(st, at) := Std(Qk(st, at)). In the denoising MDP of im-
age generation, considering the proxy ˜R is a function of the
final state x0, we can simplify the uncertainty quantifica-
tion as U(st) := Std(V k(st)). However, directly learn-
ing ensemble K value functions from scratch is computa-
tionally prohibitive for large-scale generative models, given
their vast state space and the complexity of the image gen-
eration process. To circumvent this challenge, we adopt a
more practical approach that approximates the value func-
tion using readily available, pre-trained reward models [28].
Instead of leveraging the deviation among ensemble value
functions for estimating uncertainty, we propose a new un-

𝐽𝐽PolicyModel𝝅𝜽Prompt: A red book and a yellow vaseRewardFunctionDiversity AwareAdvantage Shaping𝑑1𝑑2𝑑3𝑑𝐺×𝐴1𝐴2𝐴3𝐴𝐺ℒ𝐺𝐴𝑅𝐷𝑂𝒰1𝑥𝑖………෠𝑅11෠𝑅11෨𝑅1෨𝑅2…෨𝑅𝐺UncertaintyEstimation𝒰2𝒰𝐺……OldPolicyModel𝝅𝜽Dinov3AdvantageComputation…………𝐴1𝐴2𝐴𝐺…𝑑1𝑑2𝑑𝐺…DiversityEstimation𝑒1𝑒2𝑒𝐺…Figure 3. We train a diffusion model with 3-layer MLP on Gaussian mixtures (pre-trained distribution), with the goal to capture multimodal
high-reward clusters as shown in the reward landscape. The vanilla RL method (DDPO [4]) with a large KL coefficient β is overly
constrained and fails to increase rewards. Conversely, a small β incurs severe mode collapse. Our proposed diversity-aware optimization,
when applied alone, successfully captures the multimodal modes, including the central cluster with the lowest probability density in the
reference policy πref . Our full GARDO framework simultaneously achieves maximum reward and discovers all high-reward clusters.

certainty quantification approach:

U(xi) := w( ˜R(xi)) − mean({w( ˆRn(xi))}K

n=1),

(7)

(cid:80)

j̸=i

I(yi > yj) denotes the win rate
where w(yi) = 1
B
within a batch of size B, and { ˆRn}K
n=1 are the auxiliary re-
ward models. Under this formulation, a high uncertainty
score arises when w( ˜R) ≫ mean({w( ˆRn)}K
n=1), effec-
tively flagging samples with anomalously high proxy ˜R
compared to the ensemble. We choose light-weight Aes-
thetic [44] and ImageReward [63] as ˆR throughout our pa-
per (i.e., K=2), thereby the computation cost is negligible.
Note that only ˜R is optimized during fine-tuning, while ˆR
only serves as a metric for estimating uncertainty. Surpris-
ingly, a key empirical finding of our work is that this selec-
tive penalization is highly efficient. We find that applying
the KL penalty to only a small subset of samples (e.g., ap-
proximately 10 %) with the highest uncertainty is sufficient
to prevent reward hacking.
Adaptive KL Regularization. While the gated KL mech-
anism penalizes only high-uncertainty samples, we observe
that the sample efficiency still slows as training progresses.
As indicated in Proposition 1, πref also plays a pivotal role
in determining the optimal distribution. If ˜R(x1) = ˜R(x2),
a failure mode arises if πref(x1) ≪ πref(x2), resulting
p∗(x1) ≪ p∗(x2), which is not an expected behavior. We
remark that a static reference model becomes increasingly
sub-optimal, particularly at later training stages, which can
affect the optimization a lot. To mitigate this limitation and
facilitate sustained improvement, we propose an adaptive
regularization objective. We periodically hard-resets the
reference model πref to the current policy at specific epochs,

allowing it to remain updated. In particular, the reference
model is updated whenever the KL divergence DKL sur-
passes a pre-defined threshold ϵKL, or, failing that, after a
maximum of m gradient steps. Our proposed adaptive KL
mechanism ensures the regularization target remains rele-
vant, preventing the KL penalty from dominating the RL
loss and halting exploration, thereby achieving sustained
policy improvement.

Takeaway 2

A static reference model inevitably becomes a con-
straint on RL optimization. Dynamically updating the
reference model facilitates prolonged improvement.

4.3. Diversity-Aware Optimization

As noted by Liu et al. [30], a significant consequence of
reward hacking is the reduced diversity. This issue is ex-
acerbated by the intrinsically mode-seeking nature of rein-
forcement learning, which often struggles to capture mul-
timodal distributions [23]. Enhancing sample diversity is
therefore critical not only for preventing mode collapse but
also for broadening the policy’s exploration space. While
our gated and adaptive regularization scheme effectively
mitigates reward hacking without sacrificing sample effi-
ciency, it does not explicitly promote generation diversity.
To address this limitation, we introduce a diversity-aware
optimization strategy that amplifies rewards for high-quality
samples that also exhibit high diversity.
Diversity-Aware Advantage Shaping. The core idea is to
reshape the advantage function by incorporating a diversity-
based signal during policy optimization. Specifically, for a

×".$densityMode collapseMode coverageMaximum rewards(8)

5. Experiments

0}G

group of generated samples, {xi
i=1 ∼ pθ(x0|c), we first
map each sample from the pixel space into a semantic fea-
ture space, obtaining feature embeddings ei for each clean
image xi
0. We employ DINOv3 [47] for the feature extrac-
tion, which serves as a powerful vision foundation model.
A sample’s diversity is then quantified by its isolation in the
feature space. We define a diversity score, di, as the cosine
distance to its nearest neighbor within the group, {ei}G
i=1.
This diversity score is subsequently used to reshape the ad-
vantages, as illustrated as follows:

ei = Dinov3(xi

0), di = min({1 −

ei · ej
|ei||ej|

}j̸=i),

Ashaped

i

= Ai · di if Ai > 0 else Ai,

i, j ∼ [1, G].

There are several key design principles for our proposed
diversity-aware advantage reshaping: (1) We use a multi-
plicative re-weighting of the advantage term rather than an
additive diversity bonus. This design circumvents the need
for delicate hyperparameter tuning to balance the scales of
the proxy reward and the diversity score, which could other-
wise cause one signal to dominate the other. (2) The advan-
tage shaping is applied only when a sample’s advantage is
positive. This is a critical constraint that ensures the model
is rewarded only for generating samples that are both high-
quality and novel. It explicitly prevents the model from gen-
erating aberrant or low-quality images simply to increase
its diversity score. We summarize our method in the pseu-
docode in Alg. 1.

Takeaway 3

Multiplicative advantage reshaping exclusively within
positive samples enables robust diversity improvement.

Empirical Validation. As illustrated in Fig. 3, we pro-
vide a didactic example to validate the superior efficacy
of our method. We observe that only GARDO success-
fully captures all high-reward modes, reaching the maxi-
mum reward. The most notable outcome is GARDO’s dis-
covery of the central cluster, a mode with only 0.1× prob-
ability density assigned by the reference model compared
with other modes. This “mode recovery” capacity under-
scores GARDO’s potential for robust exploration, showing
it can incentivize emerging behaviors that lie far from the
pre-trained distribution.
An Interesting Finding. Beyond the above techniques, we
find that simply removing the standard deviation in advan-
tage normalization also helps mitigate reward hacking. In
image generation tasks, reward models often assign overly
similar rewards R(xi
0, c) to comparable images within the
same group, causing an extremely small standard deviation,
i.e., Std → 0. This dangerously amplifies small, and often
meaningless, reward differences in Eq. (1), making training

sensitive to reward noise and leading to over-optimization.
While a concurrent work, Pref-GRPO [54], proposes to mit-
igate this issue by using a preference model to convert re-
wards into pairwise win-rates, this method is computation-
ally expensive and lacks generality, as it relies on exhaustive
pairwise comparisons and is restricted to preference-based
reward models. In contrast, we propose a simpler, more ef-
ficient, and general solution: directly removing standard de-
viation (Std) from advantage normalization [18, 32]. This
imposes a natural constraint when rewards are similar, pre-
venting harmful amplification.

Setup. Following Flow-GRPO [30], we choose SD3.5-
Medium [9] as the base model and GRPO as the base
RL algorithm for empirical validation. To demonstrate the
versatility of our framework across different base models
and RL algorithms, we also provide additional results on
Flux.1-dev [25] and on a distinctly different online RL al-
gorithm, DiffusionNFT [70], in Appendix B. Throughout
all experiments, images are generated at 512×512 resolu-
tion. We fine-tune the reference model with LoRA [21]
(α = 64, r = 32). We set group size G = 24 for estimating
the diversity-aware advantages.
Benchmarks. We employ multiple tasks with diverse met-
rics to evaluate the performance of our method for prevent-
ing reward hacking without sacrificing sample efficiency.
We employ GenEval [13] and Text Render (OCR) [5] as
the proxy tasks. GenEval evaluates the model’s generation
ability on complex compositional prompts, including 6 dif-
ferent dimensions like object counting, spatial relations, and
attribute binding. OCR measures the text accuracy of gener-
ated images. For these two tasks, we use the corresponding
training and test sets from Flow-GRPO. We employ unseen
metrics including Aesthetic [44], PickScore [24], ImageRe-
ward [63], ClipScore [20], and HPSv3 [34] for assessing the
o.o.d. generalization performance. For evaluating the diver-
sity of the generated images, we use the mean of pairwise
cosine distance across a group of images for quantification,
i.e., Div = meani,j∈[1,G],i̸=j(1− ei·ej
|ei||ej | ). More implemen-
tation details are provided in Appendix A.

5.1. Results Analysis

Removing Std from Advantage Normalization is Useful.
As shown in Table 1, eliminating standard deviation nor-
malization alleviates reward hacking and improves perfor-
mance on unseen rewards compared to the baseline, while
largely preserving sample efficiency and high proxy re-
wards. However, its performance on unseen metrics still
falls short of the reference model. This observation clearly
demonstrates that, while this technique is valuable, it is in-
sufficient on its own to fully resolve the reward hacking
problem as defined in Definition 1.

Table 1. Results of GARDO and diverse baselines across both proxy rewards and diverse o.o.d. rewards. The proxy task is marked by the
orange color.

Method

#Step

Trained Tasks

Unseen Tasks

SD3.5-M [42]

+GRPO (β=0)
+GRPO (β=0) w/o std norm
+GRPO (β=0.01) w/o std norm
+ GARDO (Ours) (β=0.04) w/o div
+GARDO (Ours) (β=0.04)
+GARDO (Ours) (β=0.04)

+GRPO (β=0)
+GRPO (β=0) w/o std norm
+GRPO (β=0.01) w/o std norm
+ GARDO (Ours) (β=0.04) w/o div
+GARDO (Ours) (β=0.04)

-

600
600
600
600
600
1400

2000
2000
2000
2000
2000

0.58

22.40

GenEval OCR Aesthetic PickScore
5.07
0.63
OCR Task
4.67
4.88
5.08
5.03
5.07
5.02

21.82
22.05
22.45
22.41
22.41
22.31

0.52
0.57
0.64
0.63
0.65
0.60

0.93
0.92
0.86
0.91
0.92
0.96

0.95
0.94
0.81
0.95
0.95

GenEval Task
4.80
4.91
5.15
5.01
5.09

0.60
0.61
0.64
0.63
0.68

21.92
22.06
22.5
22.1
22.34

ImgRwd ClipScore HPSv3 Diversity
0.83

21.84

28.2

9.70

0.61
0.68
0.90
0.87
0.92
0.87

0.73
0.79
0.97
0.90
0.95

27.9
28.0
28.6
28.7
28.7
28.7

28.4
28.5
28.7
28.6
29.4

8.11
8.32
9.89
9.22
9.75
9.51

6.73
6.91
10.17
8.61
9.27

18.15
19.37
21.32
19.89
21.60
21.03

15.6
15.91
21.73
19.98
24.95

Figure 4. Learning curves and o.o.d. generalization results across different methods. GARDO not only matches the sample efficiency of
the KL-free baseline, but also mitigates reward hacking effectively, as evidenced by the superior performance on unseen metrics.

Efficiency vs. Reward Hacking. The results in Table 1
highlight a critical trade-off inherent in RL-based fine-
tuning. The GRPO (β = 0) baseline, while achieving
a high proxy reward, suffers from severe reward hacking,
as evidenced by its poor performance on unseen metrics
such as Aesthectic, HPSv3, and diversity. Adding a KL
penalty mitigates this over-optimization but at a significant
cost to sample efficiency, resulting in a proxy reward that
is over 10 points lower given the same computational bud-
get. In contrast, our proposed method successfully recon-
ciles this trade-off. As illustrated in Fig. 4, it achieves a
proxy reward comparable to the strongest KL-free base-
line while simultaneously preserving high unseen rewards.
Notably, GARDO’s generalization performance not only
matches but in some cases surpasses that of the original ref-
erence model. Quantitative results on GenEval tasks are
provided in Fig. 6. After training over the same steps, we
find that only GARDO successfully follows the instruction
and generates high-quality images, while vanilla GRPO ob-

viously hacks the proxy reward, generating noisy images
with blurred backgrounds and Gibbs artifacts.

Figure 5. Our diversity-aware advantage shaping effectively im-
proves the generation diversity.

Diversity-Aware Advantage Shaping Improves Diver-
sity. The results in Table 1 demonstrate that our proposed
diversity-aware optimization leads to a remarkable increase
in sample diversity scores, i.e., 19.98 → 24.95 for GenEval.
Visualization is provided in Fig. 5 to further validate this

4x efficiency3x efficiency(a) GenEval Task(b) OCR TaskDiversity ComparisonGARDO (Ours)GARDO w/o divPrompt: A lighthouse stands by the shoreFigure 6. Qualitative images generated by GARDO and vanilla GRPO across different prompts. Only GARDO generates images correctly
aligned with the prompt, while maintaining a satisfactory perceptual quality and diversity.

efficacy. This increased diversity broadens the policy’s ex-
ploration space, enabling it to discover novel states beyond
the initial data distribution. This, in turn, prevents conver-
gence to a narrow set of solutions (i.e., mode collapse). Ul-
timately, the enhanced exploration translates into improved
final performance on both proxy and unseen metrics.
Gated and Adaptive KL Enhances Sample Efficiency
without Reward Hacking. As shown in Table 1, ‘GARDO
w/o div’, which comprises only the gated and adaptive KL
regularization, is sufficient to overcome the sample effi-
ciency bottleneck of standard regularization, matching the
convergence speed of the KL-free baseline while still miti-
gating reward hacking. It achieves a 0.91 OCR score given
600 steps without compromising the unseen rewards. This
success is accomplished through two key components: (1)
The adaptive regularization dynamically updates the refer-
ence model to prevent excessive regularization from a sub-
optimal anchor, and (2) the gated KL identifies and applies a
relatively high regularization specifically to “illusory” sam-
ples with high reward uncertainty, thus avoiding unneces-
sary penalties. The dynamics of KL percentage and KL
loss are provided in Fig. 8a, where only around 10% of the
samples are penalized. Fig. 7 shows the examples that are
identified as highly uncertain and thus penalized.
Comparison with Multiple Reward Training. While
we leverage an ensemble of off-the-shelf reward models
for uncertainty estimation, the policy itself is optimized
against only a single proxy reward. This differentiates
our approach from multi-objective reinforcement learning
(MORL) methods, which seek to balance a weighted com-
bination of multiple, often competing, reward signals. To
demonstrate the superiority of our approach, we compare

Figure 7. Samples that are identified with high uncertainty over the
training process. While these samples reach high proxy rewards,
they get very low unseen rewards.

GARDO with the policy trained by multiple rewards, i.e.,
0.8×OCR(Proxy)+0.1×Aesthetic+0.1×Imagereward.
This baseline is unregularized to maximize the sample ef-
ficiency. From the results shown in Fig. 8b, we observe
that RL trained with multiple rewards exhibits significantly
lower sample efficiency with respect to the primary OCR
proxy reward. This finding is aligned with previous multi-
objective RL literature [66], highlighting the challenges of
optimizing for conflicting or misaligned rewards.

5.2. Emerging Behavior

By eliminating universal KL regularization for all samples
and periodically resetting the reference model according to
the learning dynamic, along with the diversity-aware opti-
mization, we can effectively unlock the emerging behavior
that is missing from the base model [14]. We validate this
using a challenging counting task: the model is trained on

a photo of a chair left of a zebraa photo of three suitcasesa photo of a broccoli and a vasea photo of a red doga photo of a happy girl at Disneylanda photo of a cat and a dog setting on the grassSD3.5-M+GRPO (𝛽=0)+GARDO(Ours)A cinematic movie poster featuring bold, dramatic typography with the title "No Time to Die”, …A crumpled plane ticket stub lying on a wooden table, clearly showing the text "Flight BA123”, …Proxy reward: 1.0Aesthetic: 3.2ImageReward: 0.12Proxy reward: 1.0Aesthetic: 3.0ImageReward: -0.256. Conclusion and Limitations

In this paper, we propose a novel and effective approach
to address the challenge of reward hacking in fine-tuning
diffusion models. We introduce a gated and adaptive
regularization mechanism for more fine-grained control,
and a diversity-aware strategy to encourage mode cover-
age, which significantly enhances sample efficiency and
emerging behaviors. Nevertheless, a primary limitation of
our method is its dependency on auxiliary reward models
for uncertainty estimation. Consequently, the scalability
of our approach to resource-intensive video generative
models remains an open question for future investigation.

References

[1] Gaon An, Seungyong Moon, Jang-Hyun Kim, and Hyun Oh
Song. Uncertainty-based offline reinforcement learning with
diversified q-ensemble. Advances in neural information pro-
cessing systems, 34:7436–7447, 2021. 4

[2] Chenjia Bai, Lingxiao Wang, Zhuoran Yang, Zhi-Hong
Deng, Animesh Garg, Peng Liu, and Zhaoran Wang. Pes-
simistic bootstrapping for uncertainty-driven offline rein-
forcement learning. In International Conference on Learning
Representations, 2022. 2, 4

[3] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and
Sergey Levine. Training diffusion models with reinforce-
ment learning. arXiv preprint arXiv:2305.13301, 2023. 2
[4] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and
Sergey Levine. Training diffusion models with reinforce-
ment learning. In ICLR, 2024. 1, 3, 5

[5] Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng
Chen, and Furu Wei. Textdiffuser: Diffusion models as text
painters. Advances in Neural Information Processing Sys-
tems, 36:9353–9387, 2023. 1, 6

[6] Kevin Clark, Paul Vicol, Kevin Swersky, and David J Fleet.
Directly fine-tuning diffusion models on differentiable re-
wards. arXiv preprint arXiv:2309.17400, 2023. 2

[7] Kevin Clark, Paul Vicol, Kevin Swersky, and David J. Fleet.
Directly fine-tuning diffusion models on differentiable re-
wards. In The Twelfth International Conference on Learning
Representations, 2024. 1

[8] William R Clements, Bastien Van Delft, Benoˆıt-Marie
Robaglia, Reda Bahi Slaoui, and S´ebastien Toth. Estimating
risk and uncertainty in deep reinforcement learning. arXiv
preprint arXiv:1905.09638, 2019. 4

[9] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim
Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik
Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling recti-
fied flow transformers for high-resolution image synthesis.
In Forty-first international conference on machine learning,
2024. 6

[10] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu,
Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Moham-
mad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok:
Reinforcement learning for fine-tuning text-to-image diffu-
sion models. Advances in Neural Information Processing
Systems, 36:79858–79885, 2023. 1, 2, 3

(a) KL dynamics

(b) Compare w/ multiple rewards

Figure 8. (a): Dynamics of both KL loss and gated KL percentage
throughout training. (b): Reward curves of GARDO and multi-
reward GRPO on the OCR task.

datasets containing 1-9 objects and then tested on its abil-
ity to generate 10-11 objects, where the base model con-
sistently fails. The results in Table 2 show that GARDO
significantly improves the counting accuracy. This is partic-
ularly evident in the difficult task of generating 10 objects,
where the base model exhibits near-zero accuracy. We pro-
vide the visualization results of GARDO counting 11 ob-
jects in Fig. 9.

Table 2. Counting accuracy across vanilla GRPO with different
KL coefficient and our method.

Method

#Step Trained Tasks

SD3.5-M [42]
GRPO (β = 0.04)
GRPO (β = 0.01)
GRPO (β = 0)
GARDO (Ours)

-
2000
2000
2000
2000

0.28
0.41
0.56
0.77
0.77

Unseen Tasks

Counting 10 Counting 11

0.01
0.09
0.27
0.28
0.38

0.01
0.07
0.15
0.15
0.18

Figure 9. Visualization of GARDO counting 11 objects.

0200400600Steps0.00.10.20.30.40.50.60.70.8KL loss024681012KL Percentage (%)KL LossKL Percentage024681012KL Percentage (%)KL LossKL Percentage05001000150020002500Steps0.600.650.700.750.800.850.900.95Proxy RewardGARDO (Ours)GRPO w/ Multiple Rewardsa photo of eleven forksa photo of eleven cupsa photo of eleven parking metersa photo of eleven traffic lightsa photo of eleven refrigeratorsGARDO (Ours)a photo of eleven surfboards[11] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu,
Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Moham-
mad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Re-
inforcement learning for fine-tuning text-to-image diffusion
models. Advances in Neural Information Processing Sys-
tems, 36, 2024. 2

[12] Xiaolong Fu, Lichen Ma, Zipeng Guo, Gaojing Zhou,
Chongxiao Wang, ShiPing Dong, Shizhe Zhou, Ximan Liu,
Jingling Fu, Tan Lit Sin, et al. Dynamic-treerpo: Breaking
the independent trajectory bottleneck with structured sam-
pling. arXiv preprint arXiv:2509.23352, 2025. 2

[13] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt.
Geneval: An object-focused framework for evaluating text-
to-image alignment. Advances in Neural Information Pro-
cessing Systems, 36:52132–52152, 2023. 6

[14] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song,
Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi
Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning
capability in llms via reinforcement learning. Nature, 645
(8081):633–638, 2025. 8

[15] Shashank Gupta, Chaitanya Ahuja, Tsung-Yu Lin,
Sreya Dutta Roy, Harrie Oosterhuis, Maarten de Rijke,
and Satya Narayan Shukla. A simple and effective re-
learning method for text-to-image diffusion
inforcement
fine-tuning. arXiv preprint arXiv:2503.00897, 2025. 1
[16] Anthony GX-Chen, Jatin Prakash, Jeff Guo, Rob Fer-
gus, and Rajesh Ranganath. Kl-regularized reinforcement
arXiv preprint
learning is designed to mode collapse.
arXiv:2510.20817, 2025. 4

[17] Haoran He, Jiajun Liang, Xintao Wang, Pengfei Wan, Di
Zhang, Kun Gai, and Ling Pan. Scaling image and video
generation via test-time evolutionary search. arXiv preprint
arXiv:2505.17618, 2025. 2

[18] Haoran He, Yuxiao Ye, Qingpeng Cai, Chen Hu, Binxing
Jiao, Daxin Jiang, and Ling Pan. Random policy valuation
is enough for llm reasoning with verifiable rewards. arXiv
preprint arXiv:2509.24981, 2025. 6

[19] Xiaoxuan He, Siming Fu, Yuke Zhao, Wanli Li, Jian Yang,
Dacheng Yin, Fengyun Rao, and Bo Zhang. Tempflow-grpo:
When timing matters for grpo in flow models. arXiv preprint
arXiv:2508.04324, 2025. 2

[20] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras,
and Yejin Choi. Clipscore: A reference-free evaluation met-
ric for image captioning. arXiv preprint arXiv:2104.08718,
2021. 6, 14

[21] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-
Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al.
Lora: Low-rank adaptation of large language models. ICLR,
1(2):3, 2022. 6

[22] Jaihoon Kim, Taehoon Yoon, Jisung Hwang, and Minhyuk
Sung.
Inference-time scaling for flow models via stochas-
tic generation and rollover budget forcing. arXiv preprint
arXiv:2503.19385, 2025. 2

[23] Sunwoo Kim, Minkyu Kim, and Dongmin Park. Test-
time alignment of diffusion models without reward over-
optimization. In The Thirteenth International Conference on
Learning Representations, 2025. 5

[24] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Ma-
tiana, Joe Penna, and Omer Levy. Pick-a-pic: An open
dataset of user preferences for text-to-image generation. Ad-
vances in neural information processing systems, 36:36652–
36663, 2023. 6, 14

[25] Black Forest Labs. Flux. https://github.com/

black-forest-labs/flux, 2024. 1, 6, 14

[26] Cassidy Laidlaw, Shivam Singhal, and Anca Dragan. Corre-
lated proxies: A new definition and improved mitigation for
reward hacking. In The Thirteenth International Conference
on Learning Representations, 2025. 2, 3

[27] Junzhe Li, Yutao Cui, Tao Huang, Yinping Ma, Chun Fan,
Miles Yang, and Zhao Zhong. Mixgrpo: Unlocking flow-
based grpo efficiency with mixed ode-sde. arXiv preprint
arXiv:2507.21802, 2025. 2

[28] Xiner Li, Yulai Zhao, Chenyu Wang, Gabriele Scalia, Gok-
cen Eraslan, Surag Nair, Tommaso Biancalani, Shuiwang Ji,
Aviv Regev, Sergey Levine, et al. Derivative-free guidance
in continuous and discrete diffusion models with soft value-
based decoding. arXiv preprint arXiv:2408.08252, 2024. 4
[29] Zhanhao Liang, Yuhui Yuan, Shuyang Gu, Bohan Chen,
Tiankai Hang, Ji Li, and Liang Zheng. Step-aware prefer-
ence optimization: Aligning preference with denoising per-
formance at each step. arXiv preprint arXiv:2406.04314,
2024. 2

[30] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng
Liu, Xintao Wang, Pengfei Wan, Di ZHANG, and Wanli
Ouyang. Flow-GRPO: Training flow matching models via
online RL. In The Thirty-ninth Annual Conference on Neu-
ral Information Processing Systems, 2025. 1, 2, 3, 5, 6, 13

[31] Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun
Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Menghan Xia,
Xintao Wang, et al. Improving video generation with human
feedback. arXiv preprint arXiv:2501.13918, 2025. 2
[32] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu
Pang, Chao Du, Wee Sun Lee, and Min Lin. Understand-
ing r1-zero-like training: A critical perspective. In Second
Conference on Language Modeling, 2025. 6

[33] Yihong Luo, Tianyang Hu, and Jing Tang. Reinforcing diffu-
sion models by direct group preference optimization. arXiv
preprint arXiv:2510.08425, 2025. 2

[34] Yuhang Ma, Xiaoshi Wu, Keqiang Sun, and Hongsheng Li.
Hpsv3: Towards wide-spectrum human preference score. In-
ternational Conference on Computer Vision, 2025. 1, 6, 14

[35] Zichen Miao, Jiang Wang, Ze Wang, Zhengyuan Yang, Li-
juan Wang, Qiang Qiu, and Zicheng Liu. Training diffusion
models towards diverse image generation with reinforcement
learning. In 2024 IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition (CVPR), pages 10844–10853,
2024. 1

[36] Zichen Miao, Jiang Wang, Ze Wang, Zhengyuan Yang, Li-
juan Wang, Qiang Qiu, and Zicheng Liu. Training diffusion
models towards diverse image generation with reinforcement
In Proceedings of the IEEE/CVF Conference on
learning.
Computer Vision and Pattern Recognition, pages 10844–
10853, 2024. 2

[37] Ian Osband, Charles Blundell, Alexander Pritzel, and Ben-
jamin Van Roy. Deep exploration via bootstrapped dqn. Ad-
vances in neural information processing systems, 29, 2016.
4

[38] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Car-
roll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini
Agarwal, Katarina Slama, Alex Ray, et al. Training language
models to follow instructions with human feedback. Ad-
vances in neural information processing systems, 35:27730–
27744, 2022. 1

[39] Alexander Pan, Kush Bhatia, and Jacob Steinhardt. The ef-
fects of reward misspecification: Mapping and mitigating
misaligned models. In International Conference on Learning
Representations, 2022. 3

[40] Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, and
Katerina Fragkiadaki. Aligning text-to-image diffusion
arXiv preprint
models with reward backpropagation.
arXiv:2310.03739, 2023. 2

[41] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christo-
pher D Manning, Stefano Ermon, and Chelsea Finn. Direct
preference optimization: Your language model is secretly a
reward model. Advances in Neural Information Processing
Systems, 36:53728–53741, 2023. 4

[42] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Bj¨orn Ommer. High-resolution image
In Proceedings of
synthesis with latent diffusion models.
the IEEE/CVF conference on computer vision and pattern
recognition, pages 10684–10695, 2022. 1, 7, 9, 14

[43] Chitwan Saharia, William Chan, Saurabh Saxena, Lala
Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour,
Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans,
et al. Photorealistic text-to-image diffusion models with deep
language understanding. Advances in neural information
processing systems, 35:36479–36494, 2022. 14

[44] Christoph Schuhmann, Romain Beaumont, Richard Vencu,
Cade Gordon, Ross Wightman, Mehdi Cherti, Theo
Coombes, Aarush Katta, Clayton Mullis, Mitchell Worts-
man, et al. Laion-5b: An open large-scale dataset for training
next generation image-text models. Advances in neural in-
formation processing systems, 35:25278–25294, 2022. 5, 6,
14

[45] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Rad-
ford, and Oleg Klimov. Proximal policy optimization algo-
rithms. arXiv preprint arXiv:1707.06347, 2017. 3

[46] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao
Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li,
Yang Wu, et al. Deepseekmath: Pushing the limits of math-
ematical reasoning in open language models. arXiv preprint
arXiv:2402.03300, 2024. 3

[47] Oriane Sim´eoni, Huy V Vo, Maximilian Seitzer, Federico
Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov,
Marc Szafraniec, Seungeun Yi, Micha¨el Ramamonjisoa,
et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025. 6,
14

[48] Saurabh Singh and Ian Fischer. Stochastic sampling from
deterministic flow models. arXiv preprint arXiv:2410.02217,
2024. 2

[49] Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov, and
David Krueger. Defining and characterizing reward gam-
ing. Advances in Neural Information Processing Systems,
35:9460–9471, 2022. 1, 2

[50] Masatoshi Uehara, Yulai Zhao, Tommaso Biancalani, and
Sergey Levine. Understanding reinforcement learning-based
fine-tuning of diffusion models: A tutorial and review. arXiv
preprint arXiv:2407.13734, 2024. 1, 4

[51] Masatoshi Uehara, Yulai Zhao, Kevin Black, Ehsan Haji-
ramezanali, Gabriele Scalia, Nathaniel Lee Diamant, Alex M
Tseng, Tommaso Biancalani, and Sergey Levine.
Fine-
tuning of continuous-time diffusion models as entropy-
regularized control. arXiv preprint arXiv:2402.15194, 2024.
1, 3

[52] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou,
Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming
Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model align-
ment using direct preference optimization. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 8228–8238, 2024. 2

[53] Feng Wang and Zihao Yu. Coefficients-preserving sam-
pling for reinforcement learning with flow matching. arXiv
preprint arXiv:2509.05952, 2025. 2

[54] Yibin Wang, Zhimin Li, Yuhang Zang, Yujie Zhou, Jiazi
Bu, Chunyu Wang, Qinglin Lu, Cheng Jin, and Jiaqi Wang.
Pref-grpo: Pairwise preference reward-based grpo for sta-
arXiv preprint
ble text-to-image reinforcement learning.
arXiv:2508.20751, 2025. 2, 6

[55] Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi
Wang. Unified reward model for multimodal understanding
and generation. arXiv preprint arXiv:2503.05236, 2025. 1

[56] Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi
Wang. Unified reward model for multimodal understanding
and generation. arXiv preprint arXiv:2503.05236, 2025. 2

[57] Olivia Wiles, Sven Gowal, Florian Stimberg, Sylvestre
Alvise-Rebuffi, Ira Ktena, Krishnamurthy Dvijotham, and
Taylan Cemgil. A fine-grained analysis on distribution shift.
arXiv preprint arXiv:2110.11328, 2021. 1

[58] Ronald J Williams. Simple statistical gradient-following al-
gorithms for connectionist reinforcement learning. Machine
learning, 8(3):229–256, 1992. 3

[59] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan
Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei
Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi
Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, De-
qing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai
Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng
Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao
Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu,
Yuxuan Cai, and Zenan Liu. Qwen-image technical report,
2025. 1

[60] Jie Wu, Yu Gao, Zilyu Ye, Ming Li, Liang Li, Hanzhong
Guo, Jie Liu, Zeyue Xue, Xiaoxia Hou, Wei Liu, et al.
Rewarddance: Reward scaling in visual generation. arXiv
preprint arXiv:2509.08826, 2025. 2

[61] Luhuan Wu, Brian Trippe, Christian Naesseth, David Blei,
and John P Cunningham. Practical and asymptotically exact

conditional sampling in diffusion models. Advances in Neu-
ral Information Processing Systems, 36:31372–31403, 2023.
4

[62] Enze Xie, Junsong Chen, Yuyang Zhao, Jincheng Yu, Ligeng
Zhu, Chengyue Wu, Yujun Lin, Zhekai Zhang, Muyang Li,
Junyu Chen, et al. Sana 1.5: Efficient scaling of training-time
and inference-time compute in linear diffusion transformer.
arXiv preprint arXiv:2501.18427, 2025. 1

[63] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai
Imagere-
Li, Ming Ding, Jie Tang, and Yuxiao Dong.
ward: Learning and evaluating human preferences for text-
to-image generation. Advances in Neural Information Pro-
cessing Systems, 36:15903–15935, 2023. 1, 2, 5, 6, 14
[64] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting
Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo,
Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual
generation. arXiv preprint arXiv:2505.07818, 2025. 1, 2, 3
[65] Kai Yang, Jian Tao, Jiafei Lyu, Chunjiang Ge, Jiaxin Chen,
Weihan Shen, Xiaolong Zhu, and Xiu Li. Using human
feedback to fine-tune diffusion models without any reward
In Proceedings of the IEEE/CVF Conference on
model.
Computer Vision and Pattern Recognition, pages 8941–
8951, 2024. 2

[66] Runzhe Yang, Xingyuan Sun, and Karthik Narasimhan.
A generalized algorithm for multi-objective reinforcement
learning and policy adaptation. Advances in neural infor-
mation processing systems, 32, 2019. 8
[67] Benjamin Yu, Jackie Liu, and Justin Cui.

Smart-grpo:
Smartly sampling noise for efficient rl of flow-matching
models. arXiv preprint arXiv:2510.02654, 2025. 2

[68] Huizhuo Yuan, Zixiang Chen, Kaixuan Ji, and Quanquan Gu.
Self-play fine-tuning of diffusion models for text-to-image
generation. arXiv preprint arXiv:2402.10210, 2024. 2
[69] Jiacheng Zhang, Jie Wu, Weifeng Chen, Yatai Ji, Xuefeng
Xiao, Weilin Huang, and Kai Han. Onlinevpo: Align video
diffusion model with online video-centric preference opti-
mization. arXiv preprint arXiv:2412.15159, 2024. 2
[70] Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang,
Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon,
Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffu-
sion reinforcement with forward process. arXiv preprint
arXiv:2509.16117, 2025. 6, 14

[71] Yujie Zhou, Pengyang Ling, Jiazi Bu, Yibin Wang, Yuhang
Zang, Jiaqi Wang, Li Niu, and Guangtao Zhai. G2rpo: Gran-
ular grpo for precise reward in flow models. arXiv preprint
arXiv:2510.01982, 2025. 2

A. Implementation Details

A.1. Details of GARDO

We choose Flow-GRPO [30] as the base RL algorithm to
implement GARDO. We provide the training details and re-
lated hyperparameters as follows:
• Following Flow-GRPO [30], we set

the sampling

timestep T = 10 and an evaluation timestep T = 40.
• We set the training batch size as 6, and set the group size

G = 24.

• Learning rate is 3e−4, and the clip range is 1e−4.
• We use Lora with α = 32 and r = 64.
• Given a group of images generated with the same prompt,
the initial noises are set to be the same to ensure GRPO’s
assumptions.

GARDO’s adaptive KL. We set the KL threshold ϵKL =
1e − 4. The maximum gradient steps m between two con-
secutive reset operations is set to be 100. This means reset
happens if the KL loss surpasses ϵKL or there have been m
gradient steps after the last recent reset.

GARDO’s gated KL We set the initial KL percentage
k = 0.1. This means the top 10% samples with the highest
uncertainty U are penalized at the beginning of the train-
ing stage. k is dynamically updated throughout the training
process. We maintain a cache window to determine k. The
window size is 20, which stores the maximum and mini-
mum value of uncertainty U over 20 epochs. If the mean
value of the uncertainty within the current batch is higher
than the maximum value cached in the window, then we set
k ← k × 1.1; if the mean value of the uncertainty within the
current batch is lower than the minimum value cached in
the window, k ← k × 0.9; otherwise, k remains unchanged.
We summarize our method in the pseudocode in Alg. 1.

A.2. Computation Specification

We train our model using 8 NVIDIA A800 GPUs. It usually
takes around 49s for a gradient step, including both sam-
pling and training.

A.3. Evaluation Metrics

Proxy Tasks.
In this paper, we include two proxy tasks to
train the diffusion models by RL, i.e., GenEval and OCR
tasks. We ensure the training hyperparameters are the same
across different methods for fair comparison. Below, we
introduce the proxy tasks used in the paper.
• The GenEval framework following Flow-GRPO’s exper-
imental protocol. The training dataset is sourced from
the Flow-GRPO dataset. GenEval includes six difficult
compositional image generation tasks. We use its official
evaluation pipeline, which detects object co-occurrence,

5:
6:
7:
8:
9:

10:
11:

12:
13:

14:

15:
16:
17:

18:
19:
20:
21:
22:

23:
24:
25:
26:
27:
28:

29:

Algorithm 1 Overview of GARDO
Require: initial policy model πθ; proxy reward ˜R; aux-
iliary reward models ˆR1, ˆR2; Dinov3 model fϕ; KL
loss threshold ϵKL; maximum reset steps m; prompt
dataset C; total sampling steps ˜T ; number of samples
per prompt G; Gated KL percentage k.

1: Set reference policy πref ← πθ
2: Uncertainty window WU ← ∅ with a window size W
3: for training iteration n = 1 to N do
4:

Sample batch prompts Cb ∼ C
Update old policy model: πθold ← πθ
Ltotal ← 0, ¯Ubatch ← 0
for each prompt c ∈ Cb do

steps.

Initial the same noise ϵ ∼ N (0, I)
Generate G images {x0}G

i=1 after ˜T sampling

Obtains advantages AG
Reshape AG

i=1 from ˜R via Eq. 1.

i=1 and get diversity-aware advan-

tages by Eq. 8.

Obtains uncertainty estimation U G
gate
Determine
Percentile({U i}G
i=1, 1 − k).

threshold:

i=1 via Eq. 7.
=
ϵU

Ltotal + 1
G

(cid:80)G

Calculate KL-regularized loss Ltotal ←
RL + I(U i > ϵU ) · Li
¯Ubatch ← ¯Ubatch + (mean({U i}))

i=1[Li

KL]

end for
Update policy model via gradient ascent: θ ← θ +

η∇θLtotal

if n mod m is 0 or LKL > ϵKL then

Reset πref ← πθ

end if
if |WU | ≥ W then

if ¯Ubatch > max(WU ) then
k ← min(k × 1.1, 1.0)

end if
if ¯Ubatch < min(WU ) then

k ← k × 0.9

end if

end if
Update window WU : Append ¯Ubatch, remove old-

est if |WU | > W .

30: end for

spatial positioning, object count, and color attributes for
fine-grained assessment.

• For the OCR task, we use the training dataset and test
dataset from Flow-GRPO. This task measures text render-
ing accuracy with the reward r = max(1 − N e/Nref , 0),
where N e is the minimum edit distance between the ren-
dered text and the target text and Nref is the number of
characters required to render.

Table 3. Results on GenEval tasks for DiffusionNFT algorithm. GARDO successfully surpasses baselines in terms of both sample
efficiency in proxy reward and generalization on o.o.d. rewards.

Method

#Step

Trained Tasks

Unseen Tasks

SD3.5-M [42]
+DiffusionNFT (β=0)
+DiffusionNFT (β=0.04)
+DiffusionNFT (β=0.04)
+GARDO (Ours) (β=0.04)

-
400
400
1200
400

GenEval OCR Aesthetic PickScore
5.07
0.63
4.23
0.94
4.87
0.72
4.55
0.90
5.23
0.95

22.40
21.85
22.54
21.96
22.59

0.58
0.68
0.54
0.51
0.64

ImgRwd ClipScore HPSv3 Diversity
0.83
0.62
1.14
0.89
1.10

21.84
11.78
13.87
13.70
14.57

9.70
5.66
11.51
10.04
12.17

28.2
28.8
29.1
29.0
29.2

Unseen Tasks We adopt DrawBench [43] for evaluation,
which consists of 200 prompts spanning 11 different cat-
egories, serving as an effective test set for comprehensive
evaluation of the T2I models. For each prompt, we generate
four images for evaluation for a fair and convincing compar-
ison. We employ Aesthetic [44], PickScore [24], ImageRe-
ward [63], ClipScore [20], and HPSv3 [34] for extensive
evaluation of the o.o.d. generalization ability, and detect-
ing the degree of reward hacking. For the diversity score,
we employ Dinov3 [47] to extract feature embeddings ei of
each image. Then we use the mean of pairwise cosine dis-
tance across a group of images for diversity quantification:

Div = meani,j∈[1,G],i̸=j(1 −

ei · ej
|ei||ej|

),

(9)

where a group of images is generated, given the same
prompt.

B. Additional Results

B.1. Results on DiffusionNFT [70]

GARDO is built upon an existing regularized-RL objec-
tive, which can be compatible with various RL algorithms.
To demonstrate GARDO’s versatility, we apply it to a re-
cently released RL algorithm, DiffusionNFT [70], which
can be very different from GRPO. DiffusionNFT directly
optimizes velocity without relying on the computation of
log likelihood, bridging the gap between SFT pre-training
and RL post-training.
From the results shown in Ta-
ble 3, we have the following observations: (1) Similar to
Flow-GRPO, vanilla DiffusionNFT still suffers from re-
ward hacking, as evidenced by the reduced performance
on unseen tasks like Aesthetic, picksocre, and HPSv3. (2)
Adding KL regularization can effectively mitigate reward
hacking, enabling more robust optimization. However, it
significantly compromises the sample efficiency. Given the
same training steps (i.e., 400 steps), KL-regularized Diffu-
sionNFT only achieves a 0.72 accuracy on the proxy task,
i.e., GenEval. (3) GARDO performs best in a balance be-
tween sample efficiency and the mitigation of reward hack-
ing. GARDO can obtain the highest score (i.e., 0.95) on
GenEval given 400 steps without reducing unseen rewards
compared to the reference model. It even boosts the unseen

reward, like Aesthetic, and remains the highest diversity
compared with baselines. This clearly demonstrates that
GARDO achieves the highest sample efficiency while effec-
tively preventing hacking, showing superior performance in
terms of both proxy rewards and unseen rewards.

B.2. Results on Flux.1-dev

GARDO can also generalize to different base models. We
choose Flux.1-dev [25] as the base model, which con-
tains 12B parameters and is known as the SOTA model
for text-to-image generation. Here, we leverage HPSv2
as the proxy reward for optimization, using open-sourced
HPDv3 (https://huggingface.co/datasets/
MizzenAI/HPDv3) as the training dataset. GARDO still
achieves the highest sample efficiency compared with the
KL-regularized Flow-GRPO method. We provide the re-
ward curves in Fig. 10a and qualitative demos in Fig. 11 and
Fig. 12. As shown in Fig. 10b, GARDO outperforms both
Flow-GRPO and base models in generalizing to unseen re-
wards, indicating GARDO’s great potential to optimize any
proxy rewards without reward hacking.

(a)

(b)

Figure 10. (a): Learning curves of GARDO on Flux.1-dev with
HPSv2 as the proxy reward. (b): Results on both proxy reward
and o.o.d. unseen rewards. GARDO performs best in the trade-
off.
B.3. More Qualitative Results

We provide the generated images along the training process
in Fig. 13 and Fig. 14. As the training step increases, we ob-
serve that Flow-GRPO obviously hacks the reward ( or ex-
ploits the flaws), yielding reduced perceptual visual quality.
However, GARDO remains a high visual quality through-
out the training process, without compromising optimiza-
tion performance on the proxy reward.

0100200300400500Steps0.220.240.260.280.300.320.34Proxy Reward (HPSv2)GARDO (Ours)GRPO (=0.01)GRPO (=0.001)HPSv2(Proxy)OCRAestheticPickScoreImgRwdClipScoreHPSv3Diversity0.330.525.7523.041.1126.6713.1517.17GARDO (Ours)Flow-GRPOFlux.1-devFigure 11. Qualitative results on Flux.1-dev across GARDO and baselines. While both GRPO and GARDO significantly improve the
visual quality, GRPO tends to hack the HPSv2 reward, generating unnecessary or even undesired details.

A yellow book and a red vaseA blue bird and a brown bearA white car and a red sheepA panda making latte art A sheep to the right of a wine glass +GARDO(Ours)Flow-GRPOFlux.-1devA storefront with NeurIPS written on itFigure 12. Qualitative results on Flux.1-dev across GARDO and baselines. While both GRPO and GARDO significantly improve the
visual quality, GRPO tends to hack the HPSv2 reward, generating unnecessary or even undesired details.

A panda making latte art A sheep to the right of a wine glass +GARDO(Ours)Flow-GRPOFlux.-1devA shark in the desertFigure 13. Generated images along the training process.

Figure 14. Generated images along the training process.

GARDO    (Ours)Flow-GRPOTraining StepPrompt: a photo of a broccoli above a bottleGARDO    (Ours)Flow-GRPOTraining StepPrompt: a photo of an orange motorcycle and a pink donut