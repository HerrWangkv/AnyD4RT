6
2
0
2

n
u
J

0
3

]

V
C
.
s
c
[

3
v
1
7
2
6
1
.
3
0
6
2
:
v
i
X
r
a

VIGOR: VIdeo Geometry-Oriented Reward for
Temporal Generative Alignment

Tengjiao Yin1 , Jinglei Shi1†

, Heng Guo2 , and Xi Wang3

1 VCIP & TMCC & DISSec, College of Computer Science, Nankai University
2 Beijing University of Posts and Telecommunications
3 LIX, École Polytechnique, IP Paris

https://vigor-geometry-reward.com/

Abstract. Video diffusion models lack explicit geometric supervision
during training, leading to inconsistency artifacts such as object defor-
mation, spatial drift, and depth violations in generated videos. To address
this limitation, we propose a geometry-based reward model that leverages
pretrained geometric foundation models to evaluate multi-view consis-
tency through cross-frame reprojection error. Unlike previous geometric
metrics that measure inconsistency in pixel space, where pixel intensity
may introduce additional noise, our approach conducts error computa-
tion in a pointwise fashion, yielding a more physically grounded and ro-
bust error metric. Furthermore, we introduce a geometry-aware sampling
strategy that filters out low-texture and non-semantic regions, focusing
evaluation on geometrically meaningful areas with reliable correspon-
dences to improve robustness. We apply this reward model to align video
diffusion models through two complementary pathways: post-training of
a bidirectional model via SFT or Reinforcement Learning and inference-
time optimization of a Causal Video Model (e.g., Streaming video genera-
tor) via test-time scaling with our reward as a path verifier. Experimental
results validate the effectiveness of our design, demonstrating that our
geometry-based reward provides superior robustness compared to other
variants. By enabling efficient inference-time scaling, our method offers
a practical solution for enhancing open-source video models without re-
quiring extensive computational resources for retraining.

Keywords: Video Diffusion Model · Geometry-based Reward · Multi-
view Consistency · Causal Inference

1

Introduction

Video diffusion models have achieved remarkable progress in recent years [4, 14,
25, 37, 41, 49] demonstrating increasingly photorealistic and temporally coherent
video synthesis. Yet a fundamental limitation persists across all of these systems:

† Corresponding author.

 
 
 
 
 
 
2

T. Yin et al.

Object Deformation

Depth Violation

Improved Geometry (Ours)

Spatial Drift

Flickering

Improved Geometry (Ours)

Fig. 1: Examples of geometry artifacts in generated videos and our results.

a lack of explicit geometric supervision during training. This results in emerg-
ing geometric artifacts (as shown in Fig. 1): object deformation, spatial drift,
flickering, depth violations, physically implausible perspective changes, etc.

While closed-source models [4,14,37] tend to exhibit stronger geometric real-
ism than their open-source counterparts, this advantage is largely attributable to
training at an unprecedented scale: by fitting massive corpora of natural video,
these models allow implicit geometric priors to emerge as a byproduct of sheer
data volume. Some recent works have attempted a more principled alternative
by incorporating explicit geometric supervision such as conditioning on depth
maps [10,21,48] or camera poses [17,44,45]. However, such approaches are funda-
mentally constrained by data accessibility: geometric ground truth of this kind is
rarely paired with the Internet-scale corpora on which modern video generators
are trained, rendering explicit supervision impractical for open-source research
and infeasible for customized downstream applications.

In light of these limitations, an increasingly adopted strategy for improv-
ing generative diffusion models is to align or tilt them toward an informative
reward (either a verifiable reward [7] or a learned reward model [13, 47]). In-
stead of relying solely on pretraining objectives, an external reward can help
capture desired structural or perceptual properties to steer generation. Exist-
ing approaches primarily focus on post-hoc alignment [31, 33], such as Super-
vised Fine-Tuning (SFT) or Reinforcement Learning (RL), where model param-
eters are tilted to reward-favoured distribution. Alternatively, without additional

VIdeo Geometry-Oriented Reward for Temporal Generative Alignment

3

training, reward-based Test-Time Scaling (TTS) [34,51] can improve generation
by using rewards at inference time to guide search, re-ranking, or iterative re-
finement among candidate rollouts. Such a paradigm also applies to geometric
aspects: recent works [1, 11, 26, 31] have demonstrated that reward-based post-
training (e.g., RL) can effectively improve video generation quality, revealing
that the success of reward-based alignment hinges on two key components: (a) a
well-designed reward function that faithfully captures the target property (e.g.,
geometric realism), and (b) a procedure that effectively leverages such reward
for alignment, either through parameter-updating post-hoc optimization (e.g.,
SFT/RL) or inference-time optimization (e.g., TTS).

In this work, unlike prior approaches that rely on handcrafted, rule-based re-
wards (e.g., [11,12,26,31,33]), we propose a geometry-based reward derived from
a pretrained geometric foundation model VGGT [42]. We compute the pointwise
reprojection error as a reward signal and additionally leverage the strong rep-
resentation of VGGT to identify geometrically meaningful regions for guiding
sampling. Based on this reward, we curate a diverse prompt suite spanning a
wide range of scene types, viewing angles, and camera trajectories, and generate
paired videos with contrastive levels of geometric consistency. These pairs form
our GB3DV-25k dataset, which provides scalable preference data for geometry-
aware alignment. We then apply our reward across multiple downstream settings
and model families to demonstrate its effectiveness: (a) post-hoc alignment: SFT
and preference-based RL of a bidirectional diffusion model (using our dataset);
and (b) test-time scaling: TTS applied to both a bidirectional diffusion model
and a causal autoregressive video model, the latter of which, to the best of our
knowledge, has not been previously explored. Within the TTS setting for the
causal model, we also explore several reward-guided search strategies enabled by
the causal rollout mechanism, showing that autoregressive generation supports
more structured inference-time exploration, yielding improved scaling behavior.

Our contributions can be summarized as follows:

– We propose a geometry-based reward that measures multi-view consistency
via cross-frame pointwise reprojection error using a pretrained geometric
foundation model, together with a geometry-aware sampling strategy that
focuses evaluation on reliable, semantically meaningful regions.

– Leveraging this reward, we curate GB3DV-25k, a preference dataset of
25,600 geometry-ranked video pairs spanning diverse scenes and camera mo-
tions, and show that SFT and RL improve the geometric consistency of
bidirectional video diffusion models.

– We study reward-guided TTS for video generation, which pioneers TTS for
streaming (causal autoregressive) video models. We propose a general TTS
framework and instantiate three efficient search strategies, i.e. Search on
Start, Search on Path, and Beam Search, that use our reward as a path
verifier to improve geometric realism without retraining.

4

T. Yin et al.

2 Related Works

2.1 Video Diffusion Models

Recent video diffusion models (VDMs) are predominantly built upon Diffusion
Transformer (DiT) architectures [36] operating in compressed spatiotemporal
latent spaces. They follow a paradigm that first encode video through a 3D
variational autoencoder (VAE) that applies joint spatial and temporal com-
pression, then denoise the resulting latent representation with a transformer
backbone trained under flow matching [30] or rectified flow objectives [32]. Ac-
cording to the token processing approach, video generation models can be fur-
ther catogerized into bidirectional and causal autoregressive ones. Bidirectional
models [4, 12, 14, 24, 25, 37, 41, 49] attend to all space-time tokens jointly via
full 3D spatiotemporal attention [4, 15, 18, 49], yielding high visual fidelity and
strong long-range temporal coherence. However, this design requires denoising
all frames simultaneously, leading to attention cost that scales quadratically
with the total token count and latency proportional to video length. Causal au-
toregressive models [6, 22, 50, 53] instead generate frames sequentially with KV
caching, trading full attention for low-latency streaming. Our test-time scaling
strategy is applied on top of causal models in this family, exploiting their se-
quential generation structure to enable structured inference-time search without
any retraining.

2.2 Reward-Tilted Generation

Reward-based methods steer generative models by transferring the reward in-
formation into the generative distribution through three paradigms. The first
is gradient-based reward fine-tuning, where model weights are updated to max-
imize a reward signal, either via supervised fine-tuning (SFT) on high-reward
samples [27] or via direct backpropagation through differentiable rewards [9,31].
The second is particle-based scaling methods, which search over a discrete set of
generation candidates, via Best-of-N sampling (BoN) [40] or Test-Time Scaling
(TTS) [34, 51], and select the highest-reward output without updating model
parameters. The third is reinforcement learning, where policy optimization ob-
jectives such as RLHF [8, 35], DPO [38], and GRPO [39] are used to align the
model with a reward signal. Recent works have explored distilling geometric pri-
ors into video diffusion models via RL-based alignment. Epipolar-DPO [26] uses
Sampson epipolar distance as a geometry-aware DPO preference signal, while
concurrent work VideoGPA [11] extends this to scene-level by reconstructing
videos with VGGT [42] and measuring pixel-level discrepancy via re-rendering.
Additional related work is discussed in the supplementary material.

3 Preliminaries

3.1 Video Generation via Flow Matching

Modern video diffusion models are increasingly built upon rectified flow [32],
a framework that defines a straight-line transport between a Gaussian noise

VIdeo Geometry-Oriented Reward for Temporal Generative Alignment

5

distribution p1 = N (0, I) and the data distribution p0. Given a clean video
sample x0 ∼ p0 and noise ϵ ∼ N (0, I), the noisy latent at time t ∈ [0, 1] is
defined by the linear interpolation: xt = (1 − t) x0 + t ϵ.

∥vθ(xt, t) − (x0 − ϵ)∥2(cid:105)
(cid:104)

A neural network vθ(xt, t) is trained to predict the velocity field that trans-
ports xt back toward the data manifold by minimizing the flow-matching objec-
tive: LFM = Et,x0,ϵ
. At inference time, a clean sample
is recovered by numerically integrating the learned velocity field from t = 1
to t = 0 via an ODE solver. This formulation underlies several state-of-the-
art open-source video generators and serves as the backbone for the preference
alignment procedure described in Sec. 5.

3.2 Camera Re-Projection

We briefly review the standard pinhole camera model that underpins our ge-
ometric reward computation. Each frame i in a video sequence is associated
with a camera intrinsic matrix Ki ∈ R3×3 and an extrinsic transformation
[Ri | ti] ∈ R3×4, where Ri and ti denote the rotation and translation from
world coordinates to the camera frame. In our work, these camera parameters
and per-frame depth maps are estimated in a feed-forward manner by geometric
foundation models [28,29,42,43], which jointly predicts camera intrinsics, extrin-
sics, and dense depth from a sequence of images without recursive optimization.
Given a 2D pixel location (uk, vk) in frame i with associated depth dk, the

corresponding 3D point in world coordinates is obtained via back-projection:

Xw

k = R−1

i

(cid:16)

dkK−1
i

(cid:2)uk, vk, 1(cid:3)T

− ti

(cid:17)

,

(1)

k = (Xk, Yk, Zk)T denotes the world-space coordinate. To reproject
where Xw
this 3D point into a target frame j with camera parameters (Kj, Rj, tj), we
transform it into the target camera space and apply the perspective projection:

Xc

k = RjXw

k + tj,

(cid:104)

(cid:105)T

k , ˆv(j)
ˆu(j)

k , 1

=

1
Z c
k

Kj Xc
k,

(2)

k, Y c

k , Z c

k = (X c

where Xc
k is the corre-
sponding depth in frame j. The reprojected pixel ˆp(j)
k )T serves as
a prediction of where point k should appear in frame j, and its deviation from
the tracker-predicted correspondence constitutes our reward signal (Sec. 4.2).

k)T is the camera-space coordinate and Z c
k , ˆv(j)

k = (ˆu(j)

4 Geometry-Based Reward

To align video generative models with geometric preference, we introduce Geometry-
Based Reward, which takes advantage of an existing geometric foundation model
(VGGT [42]) to conduct dense reconstruction and evaluate three-dimensional
consistency by point-wise reprojection error. While dense reconstruction pro-
vides a geometry-grounded representation of the video, simply computing pixel-
space error via warping frame to frame would introduce significant interference.

6

T. Yin et al.

Fig. 2: Overview of our framework, consisting of two components: (a) Geometric-
Based Reward Model: a geometry-aware sampling (GAS) module leverages global
attention of VGGT to identify salient patches and computes cross-frame pointwise
reprojection error; (b) Geometric Preference Alignment: the model is aligned via
SFT [27] and DPO [31] on a bidirectional model, or test-time scaling (TTS) with our
reward as a path verifier on a causal model [53].

To remedy the issue, as shown in Fig 2-(left), we propose a geometry-aware sam-
pling strategy that filters out distractors and computes reprojection error in a
pointwise manner to capture three-dimensional inconsistencies.

4.1 Geometry-Aware Sampling

Previous studies [3,5,16,20] show that the global attention layers of VGGT exca-
vate global information across input frames. For geometric context, we find that
the shallow global-attention layers constantly emphasize geometrically mean-
ingful areas (Fig. 3, middle row), as they process more geometric properties of
the scene. We leverage this to select geometrically critical areas and filter out
irrelevant regions (e.g., sky or ground).
Given a video sequence I = {Ii}N

i=0, we pass it through alternating-attention
layers [42] and extract the shallow layer to compute the geometric-attention
score. Anchoring frame i as the query and the others j as keys, we compute the
scaled dot-product attention of the query tokens Qi to key tokens Kj:

Ai =

1
N − 1

N −1
(cid:88)

j̸=i

softmax

(cid:32)

(cid:33)

,

QiKT
j√
d

(3)

where d is the attention dimension. These token-level scores are averaged across
heads and summed over all frames j ̸= i, then up-sampled to full resolution

VGGTGeometric-Based Reward ModelInput VideoGeometric-Aware SamplingPointwise Reprojection ErrorFramewiseAggregateGeo-Attn. MapAttn. Tokenlayer 1Geo-Attn. Scoreframe j → frame iSource Cam.Target Cam.Pixelwise (Previous)Reconstruct 3D and ReprojectGeometric Preference AlignmentPost-hoc Alignment with Geometric PreferenceTest-time Scaling with Geometric PreferenceWarpMSEPointwise (Ours)L2 DistanceReproj.Corr.SFTDPOBidirectionalLoRABidirectionalLoRASearch on StartTemporal AxisSeed AxisBeamSearchSearch on Pathcandidate seedselected seedverifierdiscarded pathchosen pathpruningVIdeo Geometry-Oriented Reward for Temporal Generative Alignment

7

“Wide shot of the Brandenburg Gate under an overcast sky. Camera dollies in ...”

t
u
p
n
I

o
e
d
i
V

n
o
i
t
n
e
t
t
A

s
p
a
M

d
e
l
p
m
a
S

s
t
n
i
o
P

Fig. 3: Visualization of Geometry-Aware Sampling, which shows that the global
attention of VGGT naturally captures the background geometry. We select top-τ per-
centage of attention-emphasized patches and sample at the center of each patch.

(H, W ) via bilinear interpolation and normalized to [0, 1] to produce a geometric
attention heatmap Mi.

We partition each frame into non-overlapping p×p patches, with each patch’s
value being the mean of Mi within it, and select the top τ % by attention value.
For each selected patch, we take its center pixel as a sampling point (Fig. 3,
bottom row), yielding 2D point locations P (i)
k=1 per frame i,
where K (i) is the number of retained patches. By default we set p = 4 and
τ = 20% in all experiments.

k = {(u(i)

k )}K(i)

k , v(i)

4.2 Pointwise Reprojection Error

After identifying geometrically critical regions in each frame, we can establish
point correspondences across frames and compute pointwise reprojection errors
to quantify geometric consistency.

Point Tracking and Correspondence. For each reference frame i with sam-
pled points Pi, we employ a tracking module to establish correspondences by
querying the tracker with points Pi against all other frames j. This yields tracked
k = (u(j)
positions p(j)
k )T for each point k across all frames j ∈ {0, . . . , N −1},
along with confidence scores c(j)

k , v(j)

k ∈ [0, 1].

Unprojection to 3D Space. For each query point pk in reference frame i, we
retrieve its depth value dk = Di[vk, uk] from the VGGT-predicted depth map
Di, together with the corresponding camera parameters (Ki, Ri, ti). The 3D
world coordinate Xw
k is then recovered via back-projection as defined in Eq. 1.

8

T. Yin et al.

Reprojection and Error Computation. To validate geometric consistency,
we reproject each 3D point Xw
k into target frame j using Eq. 2, yielding the
geometry-grounded estimate ˆp(i→j)
. We apply validity filtering of sampled points
to ensure robust error computation. The final reprojection error metric is the
mean L2 distance over all valid point-frame pairs:

k

Ereproj =

1
|V|

(cid:88)

∥ˆp(i→j)
k

− p(j)

k ∥2,

(k,i,j)∈V

(4)

where V denotes the set of valid point-frame pairs. A lower reprojection error
indicates stronger geometric consistency between the tracker’s correspondences
and the foundation model’s predicted geometry.

5 Geometry-Guided Preference Alignment

Once a reliable reward model is available, it can be used to influence video gen-
eration in several ways: (a) online optimization, where the model is updated
using reinforcement learning with reward feedback, such as RLHF [8, 35] and
GRPO [39]; (b) offline optimization, which includes post-hoc preference learn-
ing methods such as DPO [38] or supervised fine-tuning [27] on offline generated
data; and (c) test-time optimization, where the reward guides search, ranking,
or iterative refinement of candidate generations without updating model pa-
rameters. Due to limitations in model size and computational speed, most video
generation methods adopt offline optimization [11,26], particularly SFT or DPO,
to incorporate reward knowledge into the generation process.

5.1 Post-hoc Preference Alignment

Preference Data Construction. For offline optimization, a dataset must first
be constructed using the reward signal to selectively curate or annotate samples.
We construct our GB3DV-25k dataset as follows: Given a fixed set of random
seeds {si}N
i=0 and condition vector c, we generate a video set via video diffusion
x0 = Gθ(zi, c), zi ∼ N (0, I) where zi is the Gaussian noise generated with seed
si. We then leverage the proposed geometry-based reward as evaluator rgeo to
assess the videos. To magnify the geometric difference signal across videos, we
pick the best and worst samples for each prompt and construct the (xw
0) pair
as our preference data.

0 , xl

Supervised Fine-Tuning. The most direct way to leverage a reward signal
is to perform supervised fine-tuning (SFT) in the reward-selected dataset D∗.
In this setting, we adopt LoRA [19], a parameter-efficient fine-tuning (PEFT)
approach, on the video generator and train the model vθ using the same diffu-
sion/flow matching objective:

LFM = Et,x0∈D∗,ϵ

∥vθ(xt, t) − (x0 − ϵ)∥2(cid:105)
(cid:104)

.

(5)

VIdeo Geometry-Oriented Reward for Temporal Generative Alignment

9

DPO with Geometric Preference. Direct Preference Optimization (DPO)
is an offline reinforcement learning method that aligns the policy using pairwise
preferences under the Bradley–Terry model. It leverages preference comparisons
to directly optimize the policy without requiring a globally normalized reward
function. Given preference pairs (xw
0), the geometric preference DPO opti-
mizes the parameters θ via:

0 , xl

max
θ

E

(xw

0 ,xl

0)∼D

(cid:20)

(cid:18)

log σ

β log

pθ(xw
pref(xw

0 | c)
0 | c)

− β log

pθ(xl
pref(xl

0 | c)
0 | c)

(cid:19)(cid:21)

,

(6)

where β is the coefficient that controls the deviation from the reference policy
pref, and D denotes the preference dataset containing xw
0 ranked by rgeo.
In our work, we adopt the formulation of Flow-DPO [31], which reformulates

0 ≻ xl

the objective for rectified flow models:

L = −E[log sigmoid (−

βt
2

−

(cid:16)

∥vw − vθ (xw
(cid:16)(cid:13)
(cid:13)vl − vθ

(cid:0)xl

t , t)∥2 − ∥vw − vref (xw
t, t(cid:1)(cid:13)
2
(cid:13)

(cid:13)vl − vref

t , t)∥2
t, t(cid:1)(cid:13)
(cid:13)

− (cid:13)

(cid:0)xl

2(cid:17)(cid:17)(cid:17)(cid:105)

(7)

,

where vref and vθ are the velocities predicted by the model, βt = β(1 − t2) is
the training weight that depends on the noise level, and x∗
0 + tϵ is
the noisy latent of the preference pair.

t = (1 − t)x∗

To prevent mode collapse, we introduce two auxiliary loss terms: a first-
order term that penalizes static motion and a second-order term that encourages
overall smoothness, applied with opposite signs:

Laux = −E(cid:2)Vart(ˆx0)(cid:3) + γ · E(cid:2)∥∆2

t ˆx0∥2(cid:3),

(8)

where ˆx0 is the reconstructed clean sample, ∆2
t ˆx0 denotes the second-order tem-
poral difference of ˆx0, and γ > 0 is the weighting coefficient. The final loss of
DPO is: Ltotal = L + λLaux, where λ controls the strength of the penalty.

5.2 TTS with Geometric Preference

In contrast to most post-hoc alignment methods that require constructing a
dataset beforehand, Test-Time Scaling (TTS) optimizes the reward signal di-
rectly at inference time. It aims to identify an optimal sample using a verifier V
together with a search procedure f :

f : V × Gθ × {RN ×H×W ×C × Rd}N → RN ×H×W ×C,

(9)

where Gθ is the pretrained generator and V evaluates the quality of candidates.
For bidirectional models, which denoise all spatial-temporal tokens jointly,
TTS reduces to a Best-of-N protocol similar to most image generation TTS sce-
narios [34, 51]: N candidates are generated in parallel, and the highest-rewarded
sample is returned. Causal autoregressive models, by contrast, generate frames

10

T. Yin et al.

sequentially, where each frame xt is conditioned on the previously generated
prefix x<t. This temporal Markov structure exposes a richer search space: one
can intervene at each step to prune low-quality paths and explore diverse trajec-
tories, going beyond the simple Best-of-N selection that only retains a narrow
search space.

To exploit this richer search space, we instantiate TTS on a causal autore-
gressive model [53] and reformulate the problem as searching for a generation
path in a space spanned by the seed axis and temporal axis, with the goal of
finding the highest-rewarded sample. Applying brute-force search guarantees an
optimal solution, but it yields intractable O(SN ) complexity for a video clip of
N frames with a seed search range of size S. To address this, we propose three
search algorithms (shown in Fig. 2, right bottom) that all achieve a reasonable
complexity but with different dynamics:

Search on Start (SoS). The algorithm operates along the seed axis. Given a
fixed set of S candidate seeds {s1, s2, . . . , sS}, the algorithm performs a complete
forward pass for each seed independently and evaluates the resulting video clip
using the reward model rgeo. The seed yielding the highest reward is selected,
and its corresponding output clip is returned:

sbest = arg max

si

rgeo[Gθ(zi, c)] ,

i = 1, . . . , S.

(10)

Search on Path (SoP). The algorithm proceeds along the temporal axis, dy-
namically selecting from a set of seed candidates S at each time step t. When
generating the next frame, it iterates over S and selects the optimal seed st
best
that yields the highest reward. The reward model evaluates frames within a slid-
ing context window W spanning the preceding w frames, including the current
one. The seed path that achieves the highest cumulative reward is then returned:

st
best = arg max

st
i

rgeo

(cid:2)W ∪ (cid:8)Gθ(zt

i , c)(cid:9)(cid:3) ,

i = 1, . . . , S.

(11)

Beam Search (BS). The algorithm generalizes both SoS and SoP by maintain-
ing K candidate paths throughout generation. At each time step, it evaluates
K × S child nodes spawned from all current paths and retains only the top-K
nodes, pruning the rest. The path achieving the highest cumulative reward is
returned. The objective follows the same formulation as Eq. 11, with the dis-
tinction that K optimal seeds st
best are retained at every step rather than one.
It’s noteworthy that the previous two search plans can be viewed as special
configurations of BS: SoS corresponds to (K, 1) and SoP corresponds to (1, S),
with complexities of O(KN ) and O(SN ). The beam search algorithm supports
flexible (K, S) configurations, yielding a general complexity of O(KSN ).

VIdeo Geometry-Oriented Reward for Temporal Generative Alignment

11

6 Experiments

6.1

Implementation Details

Dataset Curation. Preference alignment requires training pairs with notable
differences in geometric consistency. To ensure sufficient diversity, we generate
10 samples per prompt using bidirectional CausVid [50]. To construct the prompt
suite, we source scene references from two representative datasets: RealEstate10k [52]
for indoor scenes and GLDv2 [46] for outdoor scenes. We employ Qwen3-VL [2]
to generate detailed scene captions, explicitly instructing the model to describe
both static and dynamic objects, diverse camera movements, and varying shoot-
ing angles. The resulting prompt suite comprises 2,560 entries, from which
CausVid generates a total of 25,600 video clips, constituting our curated GB3DV-
25k dataset.

Evaluation Metrics. For geometric consistency, we adopt two complementary
protocols: 3D reconstruction quality for dense evaluation and multi-view con-
sistency metrics for sparse evaluation. For 3D reconstruction quality, we apply
VGGT [42] to uniformly sampled frames to obtain depth maps and camera poses,
reproject the recovered 3D point cloud into each target frame, and measure re-
projection fidelity via PSNR, SSIM, and LPIPS. For multi-view consistency,
we compute three scores on the evaluated videos: epipolar consistency (EPI),
reprojection-pixelwise (RPX), and reprojection-pointwise (RPT). For overall
video quality, we use VBench [23], a comprehensive benchmark covering sub-
ject consistency (SC), background consistency (BC), motion smoothness (MS),
dynamic degree (DD), aesthetic quality (AQ), and imaging quality (IQ). All
VBench scores reported in our tables are normalized using the empirical mini-
mum and maximum values specified in the VBench paper [23].

Baselines. We compare our methods against the following baselines:
Base Model. We use the Causvid [50] for TTS of bidirectional experiments
(Sec. 6.2), Causal-Forcing [53] for TTS of streaming video experiments (Sec. 6.3),
and Wan2.1-T2V-1.3B [41] for post-hoc alignment experiments (Sec. 6.4).
Epipolar. Built directly on the official code of [26], assessing geometric consis-
tency via epipolar constraints quantified by the Sampson distance.
Reproj-Pix. A faithful reproduction of VideoGPA [11], which employs VGGT
to estimate camera geometry and quantifies the reward via pixelwise intensity
differences between the source and warped frames.

6.2 TTS of Bidirectional Video Generation

Evaluation Setups. We evaluate the effectiveness of test-time scaling on bidi-
rectional video generation by adopting the best-of-N sampling protocol. During
evaluation, each reward model independently selects the highest-scoring video
from N generated candidates. For generality, the experiments are conducted on
the full-scale GB3DV-25k dataset and N is set to 10.

12

T. Yin et al.

Table 1: Evaluation of Test-Time Scaling via Best-of-N Sampling.

Method

Geometric Consistency Evaluation

Overall Video Quality Evaluation (VBench %)

PSNR ↑

SSIM ↑ LPIPS ↓ EPI ↓ RPX ↓ RPT ↓

SC ↑ BC ↑ MS ↑ AQ ↑

IQ ↑ Total ↑

Baseline [50]
Epipolar [26]
Reproj-Pix [11]
Reproj-Pts (Ours)

19.68
22.45
21.07

22.66

0.6381
0.7557
0.7267

0.7665

0.3604
0.2432
0.3036

0.2330

5.553
-
4.549

3.442

0.9738
0.9546
-

0.9539

4.706

2.815
3.473
-

94.16
96.22
95.21

96.39

92.36

97.16

94.02
92.97
93.95

98.14
97.48
97.96

57.28
58.20
58.04

58.23

75.68
75.91

76.14
76.09

83.33
84.50
83.97

84.52

Evaluation Results. Quantitative results are shown in Tab. 1. We omit each
method’s score on its own selection metric (marked "-"), as such self-referential
entries are trivially optimal and obscure a fair comparison. Our Reproj-Pts re-
ward achieves the best results on all three 3D reconstruction metrics and the
best EPI and RPX among multi-view metrics, showing that pointwise repro-
jection error is a more robust and physically grounded signal than pixel-space
alternatives. The consistent gain over Reproj-Pix [11] confirms the benefit of
decoupling geometric error from pixel intensity. For overall quality, our method
attains the highest total VBench score (84.52%) while leading on subject con-
sistency and aesthetic quality, indicating that optimizing geometric consistency
does not sacrifice perceptual quality.

6.3 TTS of Streaming Video Generation

Evaluation Setups. We apply test-time scaling to streaming video generation
and evaluate the three proposed search algorithms: Search on Start (SoS), Search
on Path (SoP), and Beam Search (BS). We conduct a budget scaling study
spanning budgets from 1 to 16 (i.e. budget of 4 means 4 seeds for SoS and SoP,
2 child nodes and top-2 for BS) on a 16-clip subset of GB3DV-25k to investigate
the scaling behavior of different searching schemes of TTS.

Evaluation Results. Budget scaling results are shown in Fig. 4. PSNR, SSIM,
LPIPS, and RPT are for 3D consistency metrics, and SC, BC, MS, AQ, and
IQ are the perceptual quality dimensions of VBench. We observe two findings.
(1) All variants exhibit scaling behavior: as the budget increases, all three
methods improve consistently on both geometric and perceptual metrics, con-
firming that our geometry-based reward offers a meaningful search signal. (2)
Budget configuration shapes the type of improvement: SoS allocates
K > 1 parallel paths for greater diversity but lacks temporal refinement, achiev-
ing the lowest RPT yet weaker results elsewhere; SoP maintains S > 1 seeds per
step for stable fine-grained gains, yielding the best VBench trend; Beam Search
combines both (K > 1, S > 1) and attains the strongest 3D reconstruction.
Qualitatively (Fig. 5), the baseline shows geometric artifacts and incorrect per-
spective in later frames, while all three strategies produce geometrically coherent
videos throughout. Note that SoP shares the baseline’s initial seed and thus an
identical first frame, whereas SoS and Beam Search optimize the seed before
generation and explore a larger space from the first frame.

VIdeo Geometry-Oriented Reward for Temporal Generative Alignment

13

Fig. 4: Budget Evaluation of TTS. All three methods—SoS, SoP, and Beam
Search—demonstrate scaling tendencies as the search budget increases. Beam Search
prevails in 3D metrics, while SoP achieves the best overall performance.

6.4 Post-hoc Alignment of Bidirectional Video Generation

Evaluation Setups. We apply post-hoc alignment to a bidirectional DiT [41]
using our geometry-based reward, evaluating two complementary strategies: su-
pervised fine-tuning (SFT) and Direct Preference Optimization (DPO). For
both, we adopt Low-Rank Adaptation (LoRA) [19] with rank r = 64 and α = 128
applied to the q, k, v, and o projection modules of the DiT. For the auxiliary
loss terms, we configure λ = 0.1 and γ = 0.01.

Results of post-hoc alignment. Quantitative results are shown in Tab. 2.
SFT already yields notable improvements over the baseline across most geomet-
ric metrics, as fine-tuning on high-reward samples tilts the generation distribu-
tion toward geometrically preferred modes. DPO further improves upon SFT by
explicitly contrasting winning and losing samples, achieving stronger geometric
consistency across 3D reconstruction and multi-view metrics. Among DPO vari-
ants, our Reproj-Pts reward attains the best SSIM (0.7977), LPIPS (0.1789),
and EPI (2.127), outperforming the Epipolar baseline on the metrics most re-
flective of dense geometric coherence. For overall video quality, both SFT and
DPO consistently improve over the baseline on subject and background consis-
tency, while our DPO (Reproj-Pts) achieves the best SC (97.05%), BC (95.25%),
and IQ (76.63%), demonstrating that geometric preference alignment enhances
perceptual quality alongside geometric fidelity.

369121515.015.516.016.5PSNR 36912150.420.440.460.48SSIM 36912150.480.500.520.54LPIPS 369121546810RPT 36912158687888990SC 369121587888990BC 369121592.092.593.093.5MS 3691215515253545556AQ 369121574.074.575.075.576.0IQ 369121578.078.579.079.580.080.581.0VBench Total SoS (Search on Start)SoP (Search on Path)Beam Search (Top-K)Baseline (budget = 1)14

T. Yin et al.

Baseline [53]

Baseline + Search on Path

Baseline + Search on Start

Baseline + Beam Search

Fig. 5: Qualitative Results of Test-Time Scaling. The baseline exhibits geometric
artifacts (highlighted by ×), wrong perspective relation as shown in the last two frames.
Our approach, whether optimizing over the initial seed (SoS), selecting frame-by-frame
along the temporal axis (SoP), or applying beam search (BS), consistently produces
geometrically coherent videos with no visible artifacts (highlighted by ✓).

Table 2: Evaluation of Post-hoc Alignment.

Method

Geometric Consistency Evaluation

Overall Video Quality Evaluation (VBench %)

PSNR ↑

SSIM ↑ LPIPS ↓ EPI ↓ RPX ↓ RPT ↓

SC ↑ BC ↑ MS ↑ DD ↑ AQ ↑

IQ ↑

Baseline [41]
+ SFT (Reproj-Pts)
+ SFT + DPO (Epipolar)
+ SFT + DPO (Reproj-Pts)

22.45
23.52

23.57
23.54

0.7548
0.7927
0.7973

0.7977

0.2243
0.1842
0.1818

0.1789

2.832
2.337
2.187

2.127

0.998
1.003
1.018
1.022

1.783
2.257

1.385
1.424

95.98
96.97
96.98

94.43
95.15
95.16

97.05

95.25

97.61

97.96
97.88
97.77

27.34
15.62
25.39
25.78

57.74
57.73

57.76
57.71

76.30
76.58
76.52

76.63

6.5 Ablation Studies

Regularization in Post-hoc Alignment. The geometry reward inherently
favors static frames, leading to dynamic degree (DD) collapse under DPO. The
auxiliary loss in Eq. 8, with static penalty λ and smoothness weight γ, coun-
teracts this collapse. Tab. 3 reports deltas w.r.t. λ = 0, γ = 0, isolating the DD
recovery attributable to Eq. 8 from the DPO objective itself. A moderate λ re-
stores DD with negligible quality cost, whereas an overly large γ harms motion
smoothness.

VIdeo Geometry-Oriented Reward for Temporal Generative Alignment

15

Table 3: Ablation of Regularization Terms in Eq. 8.

λ

SC ↑

BC ↑ MS ↑ DD ↑ AQ ↑

IQ ↑

γ

SC ↑

BC ↑ MS ↑ DD ↑ AQ ↑

IQ ↑

+0.02 −0.04 +0.00 +0.78 −0.02 +0.02
0.1
+0.05 −0.08 +0.02 +0.78 −0.06 −0.01
0.5
1.0
−0.00 −0.02 +0.00 +0.39 −0.15 −0.00
10.0 −0.02 +0.17 +0.08 +1.56 −0.01 −0.18

−0.03 +0.04 −0.01 −0.39 +0.01 +0.04
0.1
−0.04 +0.05 −0.02 +0.00 +0.04 +0.05
0.5
1.0
+0.01 +0.03 −0.03 +0.00 −0.03 +0.02
10.0 −0.05 −0.10 −0.12 +0.00 −0.12 +0.02

Table 4: Ablation of Geometry-Aware Sampling (GAS).

Setting

SC ↑

BC ↑ MS ↑ DD ↑ AQ ↑

IQ ↑

Setting

SC ↑

BC ↑ MS ↑ DD ↑ AQ ↑

IQ ↑

Uniform
−0.06 −0.00 +0.08 −4.30 −0.11 +0.02
Deep Attn −0.12 +0.17 +0.05 −2.34 −0.26 −0.19
−0.02 −0.02 +0.01 −1.17 −0.02 +0.02
p=8
−0.18 −0.32 −0.06 +1.17 +0.25 +0.11
τ =5%
−0.08 −0.03 −0.01 −0.78 +0.03 −0.07
τ =40%

Saliency −0.17 +0.04 +0.08 −4.69 +0.02 −0.16
−0.27 −0.37 −0.06 +2.34 +0.30 +0.11
p=2
p=16
−0.07 +0.03 −0.03 −0.39 +0.15 −0.04
τ =10% −0.22 −0.26 −0.04 +1.56 +0.19 +0.08
τ =80% −0.08 −0.03 −0.01 −0.78 +0.03 −0.07

Geometry-Aware Sampling (GAS). Tab. 4 ablates the sampling strategy,
patch size p, and ratio τ , with deltas relative to our default (shallow atten-
tion, p = 4, τ = 20%). Attention-based selection clearly surpasses uniform and
saliency sampling, and the default configuration of p and τ best balances geo-
metric coverage against perceptual quality.

7 Conclusions

In this paper, we introduced VIGOR, a Video Geometry-Oriented Reward frame-
work that addresses the lack of explicit geometric supervision in current video dif-
fusion models. Leveraging pretrained geometric foundation models, we proposed
a physically grounded reward based on pointwise cross-frame reprojection error,
coupled with a geometry-aware sampling strategy for robustness in low-texture
and non-semantic regions. Extensive experiments show that VIGOR mitigates
common temporal artifacts such as object deformation, spatial drift, and depth
violations. We further validated its versatility across post-training alignment
and inference-time optimization for both bidirectional and causal architectures,
providing a practical and scalable solution for geometrically consistent video
generation.

Acknowledgements

This work was supported by the National Natural Science Foundation of China
(Grant No. 62302240), the Beijing Major Science and Technology Project (Grant
No. Z251100007125021), and by Hi! PARIS through the Hi! PARIS Chair 2024
held at École Polytechnique (LIX). Additional computational resources were
provided by the Supercomputing Center of Nankai University and the IDRIS
High-Performance Computing facilities (under allocation 2026-AD011014300R3,
courtesy of GENCI).

16

T. Yin et al.

References

1. Asim, M., Wewer, C., Wimmer, T., Schiele, B., Lenssen, J.E.: MEt3R: Measuring
multi-view consistency in generated images. In: IEEE Conf. Comput. Vis. Pattern
Recog. (CVPR). pp. 6034–6044 (2025)

2. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., et al.: Qwen3-vl technical report.

Tech. rep., Alibaba Group (2025), arXiv:2511.21631

3. Bratulić, J., Mittal, S., Brox, T., Rupprecht, C.: On geometric understanding and

learned data priors in vggt (2025), arXiv:2512.11508

4. Brooks, T., Peebles, B., Holmes, C., DePue, W., et al.: Video generation models
as world simulators. Tech. rep., OpenAI (2024), https://openai.com/research/
video-generation-models-as-world-simulators, accessed: 2026-06-26

5. Cao, Y., Wu, F., Chen, D.Z., Zhong, Y., Hong, L., Xu, D.: Vggt-det: Mining vggt
internal priors for sensor-geometry-free multi-view indoor 3d object detection. In:
IEEE Conf. Comput. Vis. Pattern Recog. (CVPR) (2026)

6. Chen, B., Monso, D.M., Du, Y., Simchowitz, M., Tedrake, R., Sitzmann, V.: Diffu-
sion forcing: Next-token prediction meets full-sequence diffusion. In: Adv. Neural
Inform. Process. Syst. (NeurIPS) (2024)

7. Chen, J., Huang, Y., Lv, T., Cui, L., Chen, Q., Wei, F.: Textdiffuser: Diffusion
models as text painters. In: Adv. Neural Inform. Process. Syst. (NeurIPS) (2023)
8. Christiano, P., Leike, J., Brown, T.B., Martic, M., Legg, S., Amodei, D.: Deep
reinforcement learning from human preferences. In: Adv. Neural Inform. Process.
Syst. (NeurIPS) (2017)

9. Clark, K., Vicol, P., Swersky, K., Fleet, D.J.: Directly fine-tuning diffusion models
on differentiable rewards. In: Int. Conf. Learn. Represent. (ICLR). vol. 2024, pp.
4793–4822 (2024)

10. Dai, Y., Jiang, F., Wang, C., Xu, M., Qi, Y.: Fantasyworld: Geometry-consistent
world modeling via unified video and 3d prediction (2025), arXiv:2509.21657
11. Du, H., Ye, J., Cong, X., Li, R., Ni, J., et al.: Videogpa: Distilling geometry priors

for 3d-consistent video generation (2026), arXiv:2601.23286

12. Gao, Y., Guo, H., Hoang, T., Huang, W., Jiang, L., et al.: Seedance 1.0: Ex-
ploring the boundaries of video generation models. Tech. rep., ByteDance (2025),
arXiv:2506.09113

13. Ghosh, D., Hajishirzi, H., Schmidt, L.: Geneval: An object-focused framework
for evaluating text-to-image alignment. In: Adv. Neural Inform. Process. Syst.
(NeurIPS) (2023)

14. Google DeepMind: Veo 3 technical report. Tech. rep., Google DeepMind (2025),
https://storage.googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.
pdf, accessed: 2026-06-26

15. Gupta, A., Yu, L., Sohn, K., Gu, X., Hahn, M., Li, F.F., Essa, I., Jiang, L., Lezama,
J.: Photorealistic video generation with diffusion models. In: Eur. Conf. Comput.
Vis. (ECCV) (2023)

16. Han, J., Hong, S., Jung, J., Jang, W., An, H., et al.: Emergent outlier view rejection
in visual geometry grounded transformers. In: IEEE Conf. Comput. Vis. Pattern
Recog. (CVPR). pp. 427–437 (2026)

17. He, H., et al.: Cameractrl ii: Dynamic scene exploration via camera-controlled
video diffusion models. In: IEEE Int. Conf. Comput. Vis. (ICCV). pp. 13416–13426
(2025)

18. Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M., Fleet, D.J.: Video

diffusion models. In: Adv. Neural Inform. Process. Syst. (NeurIPS) (2022)

VIdeo Geometry-Oriented Reward for Temporal Generative Alignment

17

19. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen,
W.: Lora: Low-rank adaptation of large language models. In: Int. Conf. Learn.
Represent. (ICLR) (2021)

20. Hu, Y., Cheng, C., Yu, S., Guo, X., Wang, H.: VGGT4D: Mining motion cues in
visual geometry transformers for 4d scene reconstruction (2025), arXiv:2511.19971
21. Huang, T., Zheng, W., Wang, T., Liu, Y., Wang, Z., Wu, J., et al.: Voyager: Long-
range and world-consistent video diffusion for explorable 3d scene generation. ACM
Trans. Graph. (TOG) 44, 1 – 15 (2025)

22. Huang, X., Li, Z., He, G., Zhou, M., Shechtman, E.: Self forcing: Bridging the

train-test gap in autoregressive video diffusion (2025), arXiv:2506.08009

23. Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., et al.: VBench: Comprehensive bench-
mark suite for video generative models. In: IEEE Conf. Comput. Vis. Pattern
Recog. (CVPR). pp. 21807–21818 (2024)

24. Kling Team, Chen, J., Ci, Y., Du, X., Feng, Z., et al.: Kling-omni technical report.

Tech. rep., Kuaishou (2025), arXiv:2512.16776

25. Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., et al.: Hunyuanvideo: A systematic

framework for large video generative models (2025), arXiv:2412.03603

26. Kupyn, O., Manhardt, F., Tombari, F., Rupprecht, C.: Epipolar geometry improves

video generation models (2025), arXiv:2510.21615

27. Lee, K., Liu, H., Ryu, M., Watkins, O., Du, Y., Boutilier, C., Abbeel, P.,
Ghavamzadeh, M., Gu, S.S.: Aligning text-to-image models using human feedback
(2023), arXiv:2302.12192

28. Leroy, V., Cabon, Y., Revaud, J.: Grounding image matching in 3d with MASt3R.

In: Eur. Conf. Comput. Vis. (ECCV) (2024)

29. Lin, H., Chen, S., Liew, J., Chen, D.Y., Li, Z., Shi, G., Feng, J., Kang, B.: Depth
anything 3: Recovering the visual space from any views (2025), arXiv:2511.10647
30. Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for

generative modeling. In: Int. Conf. Learn. Represent. (ICLR) (2023)

31. Liu, J., Liu, G., Liang, J., Yuan, Z., Liu, X., et al.: Improving video generation

with human feedback (2025), arXiv:2501.13918

32. Liu, X., Gong, C., Liu, Q.: Flow straight and fast: Learning to generate and transfer

data with rectified flow. In: Int. Conf. Learn. Represent. (ICLR) (2022)

33. Lu, Y., Zeng, Y., Li, H., Ouyang, H., Wang, Q., et al.: Reward forcing: Effi-
cient streaming video generation with rewarded distribution matching distillation
(2025), arXiv:2512.04678

34. Ma, N., Tong, S., Jia, H., Hu, H., Su, Y.C., et al.: Inference-time scaling for diffusion

models beyond scaling denoising steps (2025), arXiv:2501.09732

35. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C.L., et al.: Training lan-
guage models to follow instructions with human feedback. In: Adv. Neural Inform.
Process. Syst. (NeurIPS) (2022)

36. Peebles, W., Xie, S.: Scalable diffusion models with transformers. In: IEEE Int.

Conf. Comput. Vis. (ICCV). pp. 4172–4182 (2023)

37. Polyak, A., Zohar, A., Brown, A., Tjandra, A., Sinha, A., et al.: Movie gen: A cast

of media foundation models (2025), arXiv:2410.13720

38. Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C.D., Finn, C.: Direct
preference optimization: Your language model is secretly a reward model. In: Adv.
Neural Inform. Process. Syst. (NeurIPS) (2024)

39. Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., et al.: DeepSeekMath: Pushing the
limits of mathematical reasoning in open language models (2024), arXiv:2402.03300

18

T. Yin et al.

40. Verdun, C.M., Oesterling, A.X., Lakkaraju, H., du Pin Calmon, F.: Soft best-of-n
sampling for model alignment. IEEE Int. Symp. Inform. Theory. (ISIT) pp. 1–6
(2025)

41. Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., et al.: Wan: Open and advanced

large-scale video generative models (2025), arXiv:2503.20314

42. Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotny, D.: Vggt: Vi-
sual geometry grounded transformer. In: IEEE Conf. Comput. Vis. Pattern Recog.
(CVPR). pp. 5294–5306 (2025)

43. Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., Revaud, J.: DUSt3R: Geometric
3d vision made easy. In: IEEE Conf. Comput. Vis. Pattern Recog. (CVPR). pp.
20697–20709 (2024)

44. Wang, X., Courant, R., Christie, M., Kalogeiton, V.: Akira: Augmentation kit on
rays for optical video generation. In: IEEE Conf. Comput. Vis. Pattern Recog.
(CVPR). pp. 2609–2619 (2025)

45. Wang, Z., Yuan, Z., Wang, X., Chen, T., Xia, M., et al.: MotionCtrl: A unified
and flexible motion controller for video generation. In: ACM SIGGRAPH (SIG-
GRAPH) (2024)

46. Weyand, T., Araujo, A., Cao, B., Sim, J.: Google Landmarks Dataset v2 – a
large-scale benchmark for instance-level recognition and retrieval. In: IEEE Conf.
Comput. Vis. Pattern Recog. (CVPR). pp. 2572–2581 (2020)

47. Wu, X., Sun, K., Zhu, F., Zhao, R., Li, H.: Human preference score: Better aligning
text-to-image models with human preference. In: IEEE Int. Conf. Comput. Vis.
(ICCV). pp. 2096–2105 (2023)

48. Xing, J., Xia, M., Liu, Y., Zhang, Y., Zhang, Y., He, Y.Y., Liu, H., Chen, H., Cun,
X., Wang, X., Shan, Y., Wong, T.T.: Make-your-video: Customized video gener-
ation using textual and structural guidance. IEEE Trans. Vis. Comput. Graph.
(TVCG) 31, 1526–1541 (2023)

49. Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., et al.: CogVideoX: Text-to-
video diffusion models with an expert transformer. In: Int. Conf. Learn. Represent.
(ICLR). vol. 2025, pp. 83048–83077 (2025)

50. Yin, T., Zhang, Q., Zhang, R., Freeman, W.T., Durand, F., et al.: From slow
bidirectional to fast autoregressive video diffusion models. In: IEEE Conf. Comput.
Vis. Pattern Recog. (CVPR). pp. 22963–22974 (2025)

51. Zhang, X., Lin, H., Ye, H., Zou, J., Ma, J., et al.: Inference-time scaling of diffusion

models through classical search (2025), arXiv:2505.23614

52. Zhou, T., Tucker, R., Flynn, J., Fyffe, G., Snavely, N.: Stereo magnification: Learn-
ing view synthesis using multiplane images. ACM Trans. Graph. (TOG) 37(4),
65:1–65:12 (2018)

53. Zhu, H., Zhao, M., He, G., Su, H., Li, C., Zhu, J.: Causal forcing: Autoregressive
diffusion distillation done right for high-quality real-time interactive video genera-
tion. In: Int. Conf. on Mach. Learn. (ICML) (2026)

