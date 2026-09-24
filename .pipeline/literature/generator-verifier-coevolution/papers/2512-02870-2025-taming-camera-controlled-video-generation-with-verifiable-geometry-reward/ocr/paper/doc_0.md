5
2
0
2

c
e
D
2

]

V
C
.
s
c
[

1
v
0
7
8
2
0
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

Taming Camera-Controlled Video Generation with Verifiable Geometry Reward

Zhaoqing Wang1 Xiaobo Xia2* Zhuolin Bie1

Jinlin Liu1

Dongdong Yu1

Jia-Wang Bian3 Changhu Wang1∗

1AIsphere

2National University of Singapore

3Nanyang Technological University

xiaoboxia.uni@gmail.com

wangchanghu@aishi.ai

Figure 1. Camera-controlled video generation with CAMVERSE. Given an initial frame, a text prompt, and a user-specified 3D camera
trajectory, our model synthesizes a video that follows the trajectory while preserving scene layout and appearance. Left: input trajectories
(start in blue, end in red). Right: frames sampled from T = 0 to T = 5s. Rows are grouped in pairs that share the same first frame and
prompt; only the camera path differs within each pair. These examples span diverse camera movements (e.g., planar curves and out-of-
plane arcs) and scenarios (e.g., indoor kitchens and outdoor driving), showcasing precise camera control.

Abstract

Recent advances in video diffusion models have remarkably
improved camera-controlled video generation, but most
methods rely solely on supervised fine-tuning (SFT), leav-
ing online reinforcement learning (RL) post-training largely
In this work, we introduce an online RL
underexplored.
post-training framework that optimizes a pretrained video
generator for precise camera control. To make RL effective

in this setting, we design a verifiable geometry reward that
delivers dense segment-level feedback to guide model opti-
mization. Specifically, we estimate the 3D camera trajec-
tories for both generated and reference videos, divide each
trajectory into short segments, and compute segment-wise
relative poses. The reward function then compares each
generated-reference segment pair and assigns an alignment
score as the reward signal, which helps alleviate reward
sparsity and improve optimization efficiency. Moreover, we

1

Camera trajectoryT=0sT = 5s 
 
 
 
 
 
construct a comprehensive dataset featuring diverse large-
amplitude camera motions and scenes with varied subject
dynamics. Extensive experiments show that our online
RL post-training clearly outperforms SFT baselines across
multiple aspects, including camera-control accuracy, geo-
metric consistency, and visual quality, demonstrating its su-
periority in advancing camera-controlled video generation.

1. Introduction

Video diffusion models have advanced rapidly, delivering
high-fidelity and temporally coherent frames directly from
textual prompts [2, 7, 9, 30, 51]. Beyond raw quality,
many methods increasingly accept user-specified controls
and conditioning signals, enabling flexible manipulation
In paral-
over content and motion [14, 16, 18, 46, 53].
lel, action-conditioned models in interactive media show
that policies can predict state transitions and future obser-
vations, pointing toward exploration of generated worlds
rather than mere depiction [12, 42, 50, 64]. Within general
video generation, camera control has emerged as a natural
interface for such exploration by injecting camera parame-
ters into pretrained diffusion models [3, 4, 31, 56, 58, 59].
These advances make it essential to reliably translate input
camera trajectories into geometrically consistent videos, es-
pecially for film creation, environment simulation, and im-
mersive experiences.

Despite their impressive progress,

the predominant
methods for camera-controlled video generation remain
supervised fine-tuning (SFT), while online reinforcement
learning (RL) is largely underexplored [36–38, 45, 61].
This gap mainly persists for two reasons. First, it is nontriv-
ial to design verifiable and geometry-aware reward func-
tions for high-dimensional video outputs, and even esti-
mating reliable camera trajectories from generated frames
can be challenging. Second, camera trajectory is intrinsi-
cally continuous, yet most rewards used in AI-generated
content (AIGC) are instance-level (e.g., image/clip scores),
resulting in sparse feedback and inefficient credit assign-
ment during optimization. Nevertheless, online RL retains
substantial untapped potential in this setting. Unlike SFT,
which passively fits to fixed training data, online RL ac-
tively generates samples from the learned conditional dis-
tribution, reinforcing target camera motions through pos-
itive feedback while discouraging failure modes through
negative feedback. Moreover, RL offers a principled mech-
anism to incorporate strong priors from large 3D mod-
els [29, 52, 55] directly into video generators,
thereby
enhancing geometric consistency and improving camera-
control accuracy of generated videos.

In this work, we introduce CAMVERSE, an online RL
post-training framework that optimizes a video generative

Figure 2. Comparison between the absolute and relative pose
error. Given a reference trajectory, the model randomly generates
two videos with corresponding trajectories. The absolute pose er-
ror prefers global-similar but locally incorrect results, while the
relative pose error emphasizes locally consistent and smooth re-
sults. The latter is more aligned with the task characteristics.

model for precise camera control. At its core lies a verifi-
able geometry reward that translates camera-control accu-
racy into dense segment-level feedback. Specifically, we
first estimate 3D camera trajectories for both the generated
and the reference videos using a large 3D model. Each tra-
jectory is then divided into short non-overlapping segments,
within which we compute relative camera motion instead
of absolute poses. Each generated reference segment pair
is subsequently scored with an alignment metric, forming a
dense reward signal. As illustrated in Figure 2, the abso-
lute pose error captures the global shape of trajectories and
favors globally similar but locally inaccurate trajectories,
whereas the relative pose error emphasizes local smooth-
ness and continuity, which aligns with the behavior desired
for camera-controlled video generation. Moreover, obtain-
ing absolutely accurate videos typically requires lots of roll-
out, which is sample-inefficient for online RL.

Apart from that, most available datasets mainly focus on
static video clips with camera parameters, which causes a
negative effect on the motion strength of generated videos,
and are challenging to encourage models to disentangle
camera movement from subject dynamics. To overcome
these limitations, we construct a large-scale dataset of video
clips collected from both real-world footage and video
games, emphasizing diverse and large-amplitude camera
movement. We estimate camera trajectories with state-of-
the-art 3D models [25, 55] and apply a multi-stage filtering
pipeline to remove failure reconstruction cases and broaden
the trajectory distribution. The resulting dataset contains
315k videos with annotated camera trajectories, of which
314k are used for training and 1k for testing. Before delving
into details, our contributions are summarized as follows:
• We propose CAMVERSE, the first online RL post-training
framework for camera-controlled video generation. The

2

(a) Reference Trajectory(c) Trajectory 2😎Absolute Pose ErrorRelative Pose Error☹(b) Trajectory 1Absolute Pose Error☹Relative Pose Error😎framework is equipped with a verifiable geometry reward
that delivers dense segment-level feedback, effectively al-
leviating reward sparsity.

• We curate a new large-scale dataset spanning real-world
footage and game environments, which is annotated with
camera trajectory and covers diverse camera movements
and varied subject dynamics.

• Extensive experiments demonstrate consistent improve-
ments over SFT baselines in camera-control accuracy, ge-
ometric consistency, and visual quality (cf., Figure 1).

2. Related Work

2.1. Video Diffusion Models

Recent years have seen rapid progress in video genera-
tion, driven by advances in model architectures [20], the
availability of large-scale datasets [6, 10, 60], the intro-
duction of comprehensive evaluation benchmarks [26, 27],
and improvements in training techniques [28, 39]. Two
major research directions in video generation are text-to-
video (T2V) and image+text-to-video (IT2V). The for-
mer directly synthesizes a video from a natural-language
prompt. Early approaches [7, 8, 15, 17, 21] adapt UNet-
based text-to-image (T2I) models by introducing tempo-
ral modules, which enable frame-to-frame dynamics while
reusing strong image priors. Subsequent works further
scale data and computation to enhance visual fidelity and
diversity [7, 8, 21, 47]. Beyond pure text conditioning,
IT2V methods [15, 24, 57] take a still image with a cap-
tion as input to guide video synthesis. More recent systems
employ diffusion transformer architectures [11, 32, 43],
large-scale training pipelines, advanced post-training tech-
niques [37, 38] to achieve long-range temporal coherence
and high visual quality, represented by models such as
[1, 2, 9, 30, 35, 40, 41, 49, 51, 63, 63]. Despite this progress,
most existing models still operate purely in 2D pixel space
without explicit 3D supervision, often leading to geometric
distortions under complex viewpoint changes. Motivated by
this limitation, we propose a post-training framework that
explicitly rewards adherence to desired camera trajectories
and multi-view geometric consistency, thereby improving
both controllability and consistency in generated videos.

2.2. Camera-Controled Video Generation

Controlling camera movement in video generation is com-
monly achieved by injecting camera information into pre-
trained models. For example, MotionCtrl [56], CameraC-
trl [18], and I2VControl-Camera [13] condition on camera
parameters (e.g., extrinsics, point trajectories, and Pl¨ucker
embeddings [48]). Afterwards, CamCo [59] incorporates
epipolar constraints into attention, while CamTrol [22]
leverages 3D point-cloud information.
AC3D [3, 4]
presents a deep analysis of camera-controlled video gener-

ation and designs an effective interface for injecting camera
representations into pretrained backbones. Beyond single-
camera settings, recent works [5, 31, 34, 58] address multi-
camera synchronization and cross-view consistency. Cam-
eraCtrl2 [19] further introduces a scalable data pipeline with
camera-trajectory annotations for dynamic scenes, substan-
tially improving the realism of dynamic content. Different
from previous works, we propose a reinforcement learning
post-training framework with verifiable geometry reward.
This directly optimizes the model to follow user-specified
camera trajectories while maintaining 3D-consistent con-
tent across frames.

2.3. Feed-Forward 3D Reconstruction

Feed-forward models that regress 3D scene structure from
image sets offer an efficient alternative to optimization-
heavy pipelines. In more detail, Dust3R [54] predicts pair-
wise point clouds in the first-view coordinate frame, but
scaling to larger scenes typically requires a brittle global
alignment. Fast3R [62] improves scalability by perform-
ing joint inference over thousands of images, mitigating the
need for costly post-hoc alignment. Subsequent work sim-
plifies the problem via decomposition. FLARE [65] first
estimates camera poses and then infers geometry, while
VGGT [52] jointly predicts camera parameters and dense
geometry using multi-task learning and large-scale data.
These feed-forward methods still anchor predictions to a
reference frame. To address that, π3 [55] addresses this
with a fully permutation-equivariant architecture that re-
moves reference-frame bias. Leveraging these advances, we
propose a novel post-training framework that injects power-
ful 3D prior learned by feed-forward models into camera-
controlled video generation. This sets new state-of-the-art
results across diverse scenes.

3. Methodology

3.1. Preliminary

Let x1:N =
Camera-controlled video generation.
{xn}N
n=1 denote a video and c denote a text prompt. Each
frame n is accompanied by camera intrinsics Kn ∈ R3×3
and camera-to-world extrinsics En = [Rn | tn] ∈ SE(3)
with Rn ∈ SO(3) and tn ∈ R3. Directly feeding
raw (Kn, Rn, tn) to the model is poorly coupled with
pixel-space features and leaves translations unconstrained.
In contrast, we construct the per-frame Pl¨ucker embed-
ding [48] as camera condition. Concretely, we compute
the world-space ray direction du,v ∝ Rn K−1
n [u, v, 1]⊤,
and the camera center on for each pixel (u, v). The
Pl¨ucker line coordinates for pixel (u, v) are pu,v = (cid:0) on ×
(cid:1) ∈ R6 and normalized to the unit direc-
du,v, du,v
tion. Stacking all pixels yields a per-frame embedding
Pn ∈ R6×h×w and the condition of a camera trajectory

3

Figure 3. Framework of CAMVERSE. Given the input first frame, a text prompt, and a reference camera trajectory, the video diffusion
model πθ samples G rollouts by a stochastic reverse process (SDE sampling). Each rollout is partitioned into non-overlapping segments.
For each rollout, a large 3D model is used to estimate the camera trajectory. After aligning the generated trajectory to the reference one, the
verifiable geometry reward compares each pair of generated-reference segments in a relative manner and assigns a dense alignment score.
With this type of reward signal, we conduct GRPO fine-tuning, improving camera-control accuracy and geometric consistency.

is P1:N = {Pn}N
n=1. Given a noisy latent zt, a diffusion
timestep t and these conditions, a pretrained video diffusion
model πθ is optimized with the standard flow-matching ob-
jective:

LFM(θ) = Ezt,t,c,P1:N

(cid:104)

∥ϵ − πθ(zt | t, c, P1:N )∥2
2

(cid:105)

, (1)

where ϵ denotes the target velocity.

Online RL. We treat the video diffusion model as a stochas-
tic policy that samples videos, x1:N ∼ πθ(· | c, P1:N ). In
each online iteration, given a pair of conditions (c, P1:N ),
we draw G rollouts from the current policy and evaluate
each with a reward function r(x1:N , c, P1:N ). Afterwards,
by omitting subscripts of x1:N and P1:N below, the policy
πθ is updated to increase the expected reward as:

max
θ

Ex∼πθ [r(x, c, P)] ,

(2)

where r(·) denotes a reward function. Interpreting the re-
ward as an “optimality” probability r(x, c, P1:N ) ∈ [0, 1],
a sample x drawn from the old policy is assigned to a pos-
itive subset with probability r(·) and to a negative subset
with probability 1 − r(·). This induces two reweighted dis-
tributions:

π+(x | c, P) =

π−(x | c, P) =

r(x, c, P) πθold (x | c, P)

Ex∼πθold

[r(x, c, P)]
(cid:0)1 − r(x, c, P)(cid:1) πθold(x | c, P)

,

Ex∼πθold

[1 − r(x, c, P)]

(3)

.

Policy improvement moves probability mass from π− to-
ward π+ while staying close to πθold , realizing via KL-
regularized maximum likelihood. Denoting J(θ, c, P) :=

4

Ex∼πθ [r(x, c, P)], the updated policy πθnew satisfies the im-
provement criterion:

∆J := J(θnew, c, P) − J(θold, c, P) > 0.

(4)

3.2. Verifiable Geometry Reward Design

For each generated video and its reference, a large 3D
model is applied to estimate per-frame camera-to-world ex-
trinsics and per-pixel confidence maps, allowing us to de-
rive the camera trajectory for both sides:

En = [Rn | tn],
˜ωn = Agg( ˜Ωn) ∈ [0, 1],

˜En = [ ˜Rn | ˜tn],

˜Ωn ∈ Rh×w.

(5)

Here E and ˜E denote extrinsics of the reference and gener-
ated video, ˜ω denotes per-frame confidence of the generated
video, and Agg(·) is the operation of mean pooling. To ad-
dress the scale ambiguity, we first estimate an Umeyama
(cid:1)
transform (s⋆, R⋆, t⋆) = Umeyama(cid:0){˜on}N
with two trajectories [33]. Then, we apply that to each cam-
era pose of the generated videos:

n=1, {ˆon}N

n=1

n = (cid:2) R⋆ ˜Rn
˜E′

(cid:12)
(cid:12) s⋆R⋆˜tn + t⋆ (cid:3).

(6)

This normalization helps alleviate the scale issue caused by
the randomness of generated content, making the following
reward calculation more convincing.

Afterward, we partition both trajectories into non-
overlapping segments of length L. Within the k-th segment,
we compute end-to-start relative transforms, emphasizing
local smoothness:

˜Tk = ( ˜E′
nk

)−1 ˜E′

nk+L,

ˆTk = ( ˆEnk )−1 ˆEnk+L.

(7)

                                     Prompt: CGI s                           tyle. Rural scene. T                                  The video shows                   a person walking along a dirt path in a desert-like environment. The individual, viewed from behind, wears a tattered gray robe with a red sash and a hooded garment…···k-thSegment2-ndSegment1-stSegment1-stSegment2-nd Segmentk-th Segment············1-stRolloutg-thRolloutVideo Diffusion ModelSDE SamplingReference TrajectoryReference TrajectoryVerifiableGeometryRewardWe form the error transform Terr
translation component Rerr
nent terr
k = t(Terr
are calculated as:

˜Tk and extract
k ) and rotation compo-
k ). Subsequently, the per-segment errors

k = R(Terr

k = ˆT−1

k

et(k) = clip (∥terr
k ∥2 , −1, 1) ,
(cid:16)
clip(cid:0) tr(Rerr
k )−1
2

eR(k) = arccos

, −1, 1(cid:1)(cid:17)

.

(8)

Errors are converted to alignment scores s by simple rever-
sal,

sk = −(cid:0)λt et(k) + λR eR(k)(cid:1),

(9)

where λt and λR are hyper-parameters to balance the im-
portance of translation and rotation.
In doing so, better-
aligned segments (smaller errors) yield larger sk (less nega-
tive). Given the per-frame confidence, we further mask out
unreliable segments via mk = 1[ωk ≥ τ ] and obtain the
final reward ˜sk. This construction yields dense feedback
that is insensitive to the ambiguous scale, locally descrip-
tive, and compatible with online RL.

3.3. Training Paradigm

By resorting to our verifiable geometry reward function, we
conduct online RL training on the video diffusion model
πθ with group-relative policy optimization (GRPO) [36,
45, 61]. As shown in Figure 3, for each pair of condi-
tions (c, P), we sample a group of G rollouts under a
stochastic reverse process so that each per-step policy is
Gaussian, πθ(xt−1 | xt, c, P) = N (cid:0)µθ(xt, t, c, P), σ2
t I(cid:1),
thereby injecting exploration noise while keeping the origi-
nal marginals. The reward function assigns a list of reward
values to the g-th rollout ˜sg
k=1. With a group of
reward values, we adopt a “z-score” normalization to obtain
group-relative advantages below:

1:K = {˜sg

k}K

Ag

k = (cid:0)sg

k − µ(cid:1)(cid:14)(cid:0)δ + γ(cid:1),

(10)

where µ and δ denote the mean and standard deviation of all
reward values in a group. Here γ is used to improve numer-
ical stability. Afterward, we optimize the video diffusion
model πθ by maximizing the following objective:

max
θ

1
G T K

G
(cid:88)

T
(cid:88)

K
(cid:88)

(cid:16)

g=1

t=1

k=1

f (cid:0)rg

t , Ag

k, ∆(cid:1)−β KL(πθ ∥ πref )

(cid:17)

,

(11)
t ,c,P)
where the ratio rg
t ,c,P) , and
t is computed by
f (·) denotes min(cid:0)rg
(cid:1). We
k, clip(rg
t Ag
initialize πref by the model after supervised fine-tuning,
which can mitigate reward hacking. In practice, we use a
few-step scheduler to collect rollouts efficiently.

t , 1 − ∆, 1 + ∆) Ag

pθ(xg
pθold (xg

t−1|xg
t−1|xg

k

4. Experiments

4.1. Implementation Details

We adopt the latent diffusion architecture [44], including a
spatial-temporal variation autoencoder (ST-VAE) and a dif-
fusion transformer. The input video is downsampled by 4
for temporal and 8 for spatial via ST-VAE. The diffusion
transformer is initialized by an internal pre-trained model,
with about 2B parameters. We interpolate input camera
poses to the same number of frames in the video data, be-
cause only keyframes are annotated with camera informa-
tion. These interpolated camera poses are further embed-
ded by a light-weight camera network, as the input camera
condition.

Supervised fine-tuning. We conduct supervised fine-
tuning on a curated dataset consisting of 315k text-video-
camera samples, with 314k used for training and 1k re-
served for evaluation. Each video is preprocessed by re-
sizing the longer side to 512 pixels while preserving the
aspect ratio. Video durations range from 2 to 10 seconds.
Our infrastructure supports variable-length sequences. We
use a packing dataloader to maintain load balance during
training. The training is performed with a batch size of ap-
proximately 128, and all model parameters are fine-tuned
for 10k iterations. We apply a text condition dropout ratio
of 0.3 during training, while the camera condition is always
retained to learn camera movement. Training is conducted
in a multi-task setting, incorporating both text-to-video and
image-to-video generation. Task sampling is imbalanced,
with 30% text-to-video and 70% image-to-video. We em-
ploy the AdamW optimizer with a learning rate of 5×10−5,
weight decay of 0.01, and ϵ set to 1 × 10−15. To effec-
tively learning camera movement, the timestep scheduler is
shifted by 5. Note that, for efficiency, we pre-extract both
the VAE latent representations and the caption embeddings
before training begins.

GRPO fine-tuning. We perform GRPO-based fine-tuning
on the image-to-video task, using identical hyperparameters
for both. A subset of approximately 3.2k samples is selected
from the training set for this stage. We apply LoRA [23]
to all linear layers within each transformer block, with a
rank of r = 64 and α = 128. Fine-tuning is conducted for
200 iterations. Each training group consists of 16 samples.
From each group, we select the top 4 and bottom 4 samples
based on reward values for GRPO optimization. We use 14
timesteps for the denoising process. The first three steps
adopt a stochastic differential equation (SDE) sampler with
a noise level of 0.7, while the remaining steps utilize a first-
order ordinary differential equation (ODE)-based sampler.
The timestep scheduler is shifted by 6. We set the classifier-
free guidance (CFG) scale to 3.5 and apply a KL divergence
loss with a weight of 1 × 10−4 to avoid reward hacking. All
experiments are conducted on 32×NVIDIA H200 GPUs.

5

Table 1. Quantitative comparison with state-of-the-arts. We evaluate each model using metrics of camera control accuracy (i.e.,
“Trans. Err.” and “Rot. Err.”), geometric consistency (i.e., “Geo. Con.”) and visual quality (i.e., “VQ”) on the RealEstate10K test set.
For CameraCtrl [18], we evaluate the ADv3-based model for the text-to-video (T2V) setting and the SVD-based model for the image-
∗” is only optimized by
to-video (I2V) setting. The released weight of AC3D [3] only supports text-to-video generation. “CAMVERSE
supervised fine-tuning. The best result in each case is marked in bold.

Method

CameraCtrl [18]
AC3D-2B [3]
AC3D-5B [3]
CAMVERSE∗
CAMVERSE

Text-to-Video

Image-to-Video

Trans. Err.↓

Rot. Err.↓ Geo. Con.↑

VQ↑

Trans. Err.↓

Rot. Err.↓ Geo. Con.↑

VQ↑

0.0887
0.0476
0.0428

0.0395
0.0293

1.4586
1.0451
0.9120

0.6506
0.5140

0.7567
0.8156
0.8820

0.9081
0.9173

2.93
3.82
4.69

5.30
5.91

0.1696
-
-

0.0337
0.0286

3.2809
-
-

0.5613
0.4685

0.6087
-
-

0.9174
0.9226

2.63
-
-

5.02
4.87

4.2. Evaluation

4.3. Main Results

Datasets. The evaluation is conducted on a test set consist-
ing of 1k video clips. This set includes 342 clips from the
RealEstate10K dataset [66] and 658 clips from our curated
dataset. These clips are sampled to cover a diverse range of
camera trajectories, ensuring a comprehensive assessment
of generation quality and robustness. All video durations
are constrained between 4 and 8 seconds. For the ablation
study, we randomly select a subset of 100 clips from the 342
RealEstate10K samples to accelerate the evaluation process
while preserving effectiveness.

Metrics. Our evaluation framework assesses model perfor-
mance across three key aspects: visual quality, camera con-
trol accuracy, and geometric consistency. Specifically,
• Camera control accuracy: We assess camera trajectory
alignment by measuring both translation and rotation er-
rors between the generated and reference videos. For each
input video, we use π3 [55] to estimate camera trajecto-
ries. To account for scale ambiguity, we align the esti-
mated trajectory of the generated video to that of the ref-
erence video using a similarity transformation. The trans-
lation error is computed as the average RMSE between
corresponding camera positions, while the rotation error
is defined as the mean angular difference between corre-
sponding orientations.

• Geometric consistency: In addition to trajectory estima-
tion, π3 [55] outputs a per-pixel confidence map. We
threshold this confidence map to obtain a binary mask,
and define the geometric consistency score as the ratio
of valid pixels to the total number of pixels. Here, the
threshold is set to 0.1.

• Visual quality: To evaluate visual fidelity, we uni-
formly sample 8 frames from each generated video clip
and compute the HPSv3 score using the prompt “A
high-quality, clear video frame.”
This
metric quantitatively reflects perceptual quality as judged
by a powerful vision-language model.

Quantitative comparison. As demonstrated in Table 1,
we compare our method with other state-of-the-art meth-
ods on the RealEstate10K test set, using camera-control ac-
curacy, geometric consistency, and visual quality (VQ) in
both text-to-video (T2V) and image-to-video (I2V) settings.
Benefiting from the advanced training recipe and our cu-
rated dataset, CAMVERSE∗ clearly surpasses existing meth-
ods [3, 18] by large margins. Despite our strong SFT base-
lines, our online RL post-training (see CAMVERSE) fur-
ther improves camera control and geometry. In T2V gen-
eration, the translation error drops 25.8% from 0.0395 to
0.0293, while the rotation error drops 21.0% from 0.6506 to
0.5140. Geometric consistency rises from 0.9081 to 0.9173
after post-training. Similar trends also show in I2V genera-
tion. Concretely, translation and rotation errors decrease by
15.1% and 16.5%, respectively.
In the I2V setting, VQ
slightly decreases after post-training (from 5.02 to 4.87),
consistent with our reward not directly optimizing percep-
tual quality.
Incorporating a multi-objective reward is an
effective way to alleviate this issue.

Qualitative analysis. We visualize results for both T2V and
I2V settings in Figure 4. Across a wide range of diverse,
large-amplitude trajectories, our model closely follows the
reference camera trajectories and maintains smooth local
viewpoint changes at turning points. Meanwhile, subjects
exhibit substantial dynamics (e.g., running and riding a
horse), and the rendered scenes remain temporally coher-
ent, indicating that the model can disentangle camera move-
ment from content motion while preserving visual fidelity.
For the shown failure, straightforward translation is still
the dominant pattern in the dataset, even diversifies the
trajectory distribution. The model therefore drifts toward
straightforward motion rather than exactly executing the
reference trajectory, resulting in early misalignment. Mit-
igating the dataset bias is an important direction for the re-
search community.

6

Figure 4. Illustrations of qualitative results. For each example, we condition on a reference camera trajectory (start in blue, end in red)
and show uniformly sampled frames from T = 0 to T = 5s. The first and second rows are text-to-video (T2V) cases. The third and fourth
rows are image-to-video (I2V) cases. The last row is a failure case. In the first half of the clip, the camera does not execute right-forward
translation and instead moves straight forward, resulting in early trajectory misalignment.

Table 2. Quantitative results of supervised fine-tuning. We evaluate each model using metrics of camera control accuracy (i.e., “Trans.
∗” adopts a training recipe consisting
Err.” and “Rot. Err.”) and visual quality (i.e., “VQ”) on the RealEstate10K test set. “CAMVERSE
of 3:7 T2V-I2V tasks by default. Results are reported for both text-to-video and image-to-video settings. The best result in each case is
marked in bold.

Method

CAMVERSE∗
w/ 7:3 T2V-I2V
w/o dynamic content
w/o shifted timesteps
w/o pose normalization

4.4. Ablation Study

Text-to-Video

Image-to-Video

Trans. Err.↓

Rot. Err.↓

0.0395
0.0411
0.0464
0.0420
0.0416

0.6506
0.7112
0.7306
0.7023
0.7458

VQ↑

5.30
5.42
5.45
5.12
5.18

Trans. Err.↓

Rot. Err.↓

0.0337
0.0363
0.0394
0.0342
0.0356

0.5613
0.6378
0.6555
0.6055
0.7135

VQ↑

5.02
5.05
5.35
4.96
4.99

We analyze 6 design choices that influence GRPO fine-
tuning. Unless otherwise noted, all of the results are re-
ported on the RealEstate10K test set [66], containing 100
clips, in the I2V setting. We mainly evaluate camera control
accuracy using translation error (Trans. Err.) and rotation
error (Rot. Err.). For training efficiency, we report the time
spent on the optimization step.

SFT training recipe. As illustrated in Table 2, we first
compare different sample ratios for T2V and I2V, demon-
strating that animation of the first frame encourages the
model to learn camera movement. Besides, training with-
out dynamic videos clearly degrades camera control accu-

7

racy. Afterwards, the shifted timestep scheduler and pose
normalization each contribute to advancing camera-control
video generation. We therefore adopt this recipe for all RL
experiments, excluding the sample ratio for T2V and I2V.

Error function. It is crucial to design an appropriate error
function for the reward model. As reported in Table 3, using
the relative error for both translation and rotation yields the
lowest overall errors, outperforming the other mixed con-
figurations. We further replace the dense, segment-level re-
ward with a clip-level reward and apply GRPO fine-tuning.
This variant results in noticeably worse camera-control ac-
curacy, demonstrating the importance of dense feedback.

Number of SDE timesteps. With total sampling steps

Camera trajectoryT=0sT = 5sTable 3. Effects on different combinations of error functions.
We evaluate each setting with metrics of camera control accuracy
(i.e., “Trans. Err.” and “Rot. Err.”) on the RealEstate10K test set
(100 clips). et(·) and eR(·) indicate the error function of transla-
tion and rotation, respectively. We report results on the image-to-
video setting. † denotes the clip-level reward. The best result in
each case is marked in bold.

et(·)

eR(·)

Trans. Err.↓

Rot. Err.↓

SFT baseline
abs.
rel.
rel.
abs.
rel.
rel.
rel.†
rel.†

0.0280
0.0292
0.0293
0.0238
0.0277

0.4444
0.4537
0.4281
0.3286
0.3739

Table 4. Effects on different numbers of timesteps. Given the
fixed number of sampling steps Ttotal, we evaluate effects on dif-
ferent numbers of SDE steps Tsde used in optimization. “Time.”
denotes the consumed time per optimization step. We report re-
sults on the image-to-video setting. The best result in each case is
marked in bold.

Tsde (Ttotal)

Trans. Err.↓

Rot. Err.↓

Time.↓

1 (14)
3 (14)
5 (14)

0.0272
0.0238
0.0278

0.4829
0.3286
0.4440

274s
343s
412s

Table 5. Effects on best-of-N sampling. We evaluate effects
on different numbers of positive and negative samples. “α-β” in-
dicates α positive and β negative samples. “Time.” denotes the
consumed time per optimization step. We report results on the
image-to-video setting. The best result in each case is marked in
bold.

Best-of-N

Trans. Err.↓

Rot. Err.↓

Time.↓

4-4
2-2
1-1

1-4
4-1

0.0238
0.0240
0.0249

0.0303
0.0262

0.3286
0.3760
0.4330

0.4948
0.3487

343s
294s
269s

305s
305s

fixed, we vary the number of stochastic steps Tsde used dur-
ing online rollouts, shown in Table 4. Tsde = 3 shows
the most accurate control with 0.0238 translation error and
0.3286 rotation error at a reasonable cost. Too few steps
are under-explored, while too many steps slow training and
slightly decrease accuracy.

Best-of-N sampling. We evaluate the groups with α posi-
tive and β negative samples. A balanced larger group (4−4)
achieves the best performance of 0.0238 translation error
and 0.3286 rotation error. Smaller or asymmetric groups

Table 6. Effects on different ranks of LoRA. We evaluate each
setting with metrics of camera control accuracy (i.e., “Trans. Err.”
and “Rot. Err.”) on the RealEstate10K test set (100 clips). We
report results on the image-to-video setting. The best result in
each case is marked in bold.

Rank r

Trans. Err.↓

Rot. Err.↓

32
64
128

0.0233
0.0238
0.0249

0.3550
0.3286
0.3778

Table 7. Effects on different timestep shift. We evaluate each
setting with metrics of camera control accuracy (i.e., “Trans. Err.”
and “Rot. Err.”) on the RealEstate10K test set (100 clips). We
report results on the image-to-video setting. The best result in
each case is marked in bold.

Timestep shift

Trans. Err.↓

Rot. Err.↓

4
6
8

0.0264
0.0238
0.0297

0.3878
0.3286
0.4165

(e.g., 1−1 and 1−4) increase both translation and rotation
errors, which indicate that both adequate positive and nega-
tive samples are important.

Rank in LoRA. In Table 6, we evaluate the effects of capac-
ity on different ranks in LoRA. As shown, r = 64 offers the
best overall trade-off between camera control accuracy and
trainable parameters. r = 32 slightly decreases translation
error but harms rotation accuracy, while r = 128 degrades
control accuracy and increases training costs.

Timestep shift.
In the GRPO fine-tuning stage, we off-
set the diffusion timesteps by different constants when sam-
pling and policy optimization, shown in Table 7. Camera
control accuracy is highly sensitive to the model’s behavior
near the start of denoising. If the shift is too small, these
noise-proximal steps are under-trained, and errors accumu-
late along the denoising process. Conversely, excessive shift
overpowers extremely noisy states, producing blurrier roll-
outs. This causes challenges to camera pose estimation, re-
sulting in inferior camera control accuracy.

5. Conclusion

In this paper, we presented CAMVERSE, an online RL
framework that post-trains video diffusion models for pre-
cise camera control. Through a verifiable geometry re-
ward that converts 3D alignment into dense segment-level
feedback, our method effectively mitigates reward sparsity
and leads to efficient model optimization. Coupled with
a large-scale dataset covering diverse and high-amplitude
camera trajectories, CAMVERSEachieves significant im-
provements in camera-control accuracy, geometric consis-

8

tency, and visual quality over SFT competitors. These
results highlight the promise of RL as a powerful post-
training paradigm for refining generative video models be-
yond static supervision. In future work, we will integrate
our method into the world model for precise action con-
trol and explore that as a scalable data engine for embodied
AI.

References

[1] Cosmos World Foundation Model Platform for Physical AI.

3

[2] AISphere. Pixverse, 2025. Accessed: 2025-10-30. 2, 3
[3] Sherwin Bahmani, Ivan Skorokhodov, Guocheng Qian, Ali-
aksandr Siarohin, Willi Menapace, Andrea Tagliasacchi,
David B Lindell, and Sergey Tulyakov. Ac3d: Analyzing
and improving 3d camera control in video diffusion trans-
formers. arXiv preprint arXiv:2411.18673, 2024. 2, 3, 6
[4] Sherwin Bahmani, Ivan Skorokhodov, Aliaksandr Siaro-
hin, Willi Menapace, Guocheng Qian, Michael Vasilkovsky,
Hsin-Ying Lee, Chaoyang Wang,
Jiaxu Zou, Andrea
Tagliasacchi, David B. Lindell, and Sergey Tulyakov. Vd3d:
Taming large video diffusion transformers for 3d camera
control. arXiv preprint arXiv:2407.12781, 2024. 2, 3

[5] Jianhong Bai, Menghan Xia, Xintao Wang, Ziyang Yuan,
Xiao Fu, Zuozhu Liu, Haoji Hu, Pengfei Wan, and
Di Zhang.
Syncammaster: Synchronizing multi-camera
video generation from diverse viewpoints. arXiv preprint
arXiv:2412.07760, 2024. 3

[6] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisser-
man. Frozen in time: A joint video and image encoder for
end-to-end retrieval. In ICCV, pages 1728–1738, 2021. 3
[7] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel
Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi,
Zion English, Vikram Voleti, Adam Letts, et al. Stable video
diffusion: Scaling latent video diffusion models to large
datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 3
[8] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dock-
horn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis.
Align your latents: High-resolution video synthesis with la-
tent diffusion models. In CVPR, pages 22563–22575, 2023.
3

[9] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue,
Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luh-
man, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya
Ramesh. Video generation models as world simulators.
2024. 2, 3

[10] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace,
Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon,
Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang,
et al. Panda-70m: Captioning 70m videos with multiple
In CVPR, pages 13320–13331,
cross-modality teachers.
2024. 3

[11] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim
Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik
Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling recti-
fied flow transformers for high-resolution image synthesis.
In ICML, 2024. 3

[12] Ruili Feng, Han Zhang, Zhantao Yang, Jie Xiao, Zhilei
Shu, Zhiheng Liu, Andy Zheng, Yukun Huang, Yu Liu,
and Hongyang Zhang. The matrix: Infinite-horizon world
generation with real-time moving control. arXiv preprint
arXiv:2412.03568, 2024. 2

[13] Wanquan Feng,

Jiawei Liu, Pengqi Tu, Tianhao Qi,
Mingzhen Sun, Tianxiang Ma, Songtao Zhao, Siyu Zhou,
and Qian He. I2vcontrol-camera: Precise video camera con-
trol with adjustable motion strength. In ICLR, 2025. 3
[14] Xiao Fu, Xian Liu, Xintao Wang, Sida Peng, Menghan
Xia, Xiaoyu Shi, Ziyang Yuan, Pengfei Wan, Di Zhang,
and Dahua Lin. 3dtrajmaster: Mastering 3d trajectory for
arXiv preprint
multi-entity motion in video generation.
arXiv:2412.07759, 2024. 2

[15] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang,
Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin,
and Bo Dai. Animatediff: Animate your personalized text-
to-image diffusion models without specific tuning. arXiv
preprint arXiv:2307.04725, 2023. 3

[16] Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala,
Dahua Lin, and Bo Dai. Sparsectrl: Adding sparse controls
to text-to-video diffusion models. In ECCV, pages 330–348,
2024. 2

[17] Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera
Hahn, Fei-Fei Li, Irfan Essa, Lu Jiang, and Jos´e Lezama.
Photorealistic video generation with diffusion models.
In
ECCV, pages 393–411, 2024. 3

[18] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo
Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling
camera control for text-to-video generation. arXiv preprint
arXiv:2404.02101, 2024. 2, 3, 6

[19] Hao He, Ceyuan Yang, Shanchuan Lin, Yinghao Xu, Meng
Wei, Liangke Gui, Qi Zhao, Gordon Wetzstein, Lu Jiang, and
Hongsheng Li. Cameractrl ii: Dynamic scene exploration
via camera-controlled video diffusion models. arXiv preprint
arXiv:2503.10592, 2025. 3

[20] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William
Chan, Mohammad Norouzi, and David J Fleet. Video dif-
fusion models. In NeurIPS, pages 8633–8646, 2022. 3
[21] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu,
and Jie Tang.
Cogvideo: Large-scale pretraining for
text-to-video generation via transformers. arXiv preprint
arXiv:2205.15868, 2022. 3

[22] Chen Hou, Guoqiang Wei, Yan Zeng, and Zhibo Chen.
Training-free camera control for video generation. arXiv
preprint arXiv:2406.10126, 2024. 3

[23] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-
Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al.
Lora: Low-rank adaptation of large language models.
In
ICLR, 2022. 5

[24] Yaosi Hu, Chong Luo, and Zhenzhong Chen. Make it
move: controllable image-to-video generation with text de-
scriptions. In CVPR, pages 18219–18228, 2022. 3

[25] Jiahui Huang, Qunjie Zhou, Hesam Rabeti, Aleksandr Ko-
rovko, Huan Ling, Xuanchi Ren, Tianchang Shen, Jun Gao,
Dmitry Slepichev, Chen-Hsuan Lin, et al. Vipe: Video
pose engine for 3d geometric perception. arXiv preprint
arXiv:2508.10934, 2025. 2

9

[26] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si,
Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin,
Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin
Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Com-
prehensive benchmark suite for video generative models. In
CVPR, 2024. 3

[27] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo
Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang
Si, Yuming Jiang, Yaohui Wang, Xinyuan Chen, Ying-
Cong Chen, Limin Wang, Dahua Lin, Yu Qiao, and Zi-
wei Liu. Vbench++: Comprehensive and versatile bench-
arXiv preprint
mark suite for video generative models.
arXiv:2411.13503, 2024. 3

[28] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine.
Elucidating the design space of diffusion-based generative
models. In NeurIPS, pages 26565–26577, 2022. 3

[29] Nikhil Keetha, Norman M¨uller, Johannes Sch¨onberger,
Lorenzo Porzi, Yuchen Zhang, Tobias Fischer, Arno
Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes,
et al. Mapanything: Universal feed-forward metric 3d re-
construction. arXiv preprint arXiv:2509.13414, 2025. 2

[30] KuaiShou. Kling, 2025. Accessed: 2025-02-23. 2, 3
[31] Zhengfei Kuang, Shengqu Cai, Hao He, Yinghao Xu, Hong-
sheng Li, Leonidas J Guibas, and Gordon Wetzstein. Collab-
orative video diffusion: Consistent multi-video generation
with camera control. In NeurIPS, pages 16240–16271, 2025.
2, 3

[32] Black Forest Labs, Stephen Batifol, Andreas Blattmann,
Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dock-
horn, Jack English, Zion English, Patrick Esser, et al. Flux.
1 kontext: Flow matching for in-context image generation
and editing in latent space. arXiv preprint arXiv:2506.15742,
2025. 3

[33] Jim Lawrence, Javier Bernal, and Christoph Witzgall. A
purely algebraic justification of the kabsch-umeyama algo-
rithm. Journal of research of the National Institute of Stan-
dards and Technology, 124:1, 2019. 4

[34] Bing Li, Cheng Zheng, Wenxuan Zhu, Jinjie Mai, Biao
Zhang, Peter Wonka, and Bernard Ghanem. Vivid-zoo:
Multi-view video generation with diffusion model.
In
NeurIPS, pages 62189–62222, 2025. 3

[35] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu,
Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Li-
uhan Chen, et al. Open-sora plan: Open-source large video
generation model. arXiv preprint arXiv:2412.00131, 2024.
3

[36] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng
Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli
Ouyang. Flow-grpo: Training flow matching models via on-
line rl. arXiv preprint arXiv:2505.05470, 2025. 2, 5

[37] Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun
Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Menghan Xia,
Xintao Wang, et al. Improving video generation with human
feedback. arXiv preprint arXiv:2501.13918, 2025. 3
[38] Runtao Liu, Haoyu Wu, Ziqiang Zheng, Chen Wei, Yingqing
He, Renjie Pi, and Qifeng Chen.
Videodpo: Omni-
preference alignment for video diffusion generation. In Pro-

ceedings of the Computer Vision and Pattern Recognition
Conference, pages 8009–8019, 2025. 2, 3

[39] Xingchao Liu, Chengyue Gong, and Qiang Liu.

Flow
straight and fast: Learning to generate and transfer data with
rectified flow. arXiv preprint arXiv:2209.03003, 2022. 3
[40] Guoqing Ma, Haoyang Huang, Kun Yan, Liangyu Chen, Nan
Duan, Shengming Yin, Changyi Wan, Ranchen Ming, Xi-
aoniu Song, Xing Chen, et al. Step-video-t2v technical re-
port: The practice, challenges, and future of video founda-
tion model. arXiv preprint arXiv:2502.10248, 2025. 3
[41] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Zi-
wei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte:
Latent diffusion transformer for video generation. arXiv
preprint arXiv:2401.03048, 2024. 3

[42] Jack Parker-Holder, Philip Ball, Jake Bruce, Vibhavari
Dasagi, Kristian Holsheimer, Christos Kaplanis, Alexandre
Moufarek, Guy Scully, Jeremy Shar, Jimmy Shi, Stephen
Spencer, Jessica Yung, Michael Dennis, Sultan Kenjeyev,
Shangbang Long, Vlad Mnih, Harris Chan, Maxime Gazeau,
Bonnie Li, Fabio Pardo, Luyu Wang, Lei Zhang, Fred-
eric Besse, Tim Harley, Anna Mitenkova, Jane Wang, Jeff
Clune, Demis Hassabis, Raia Hadsell, Adrian Bolton, Satin-
der Singh, and Tim Rockt¨aschel. Genie 2: A large-scale
foundation world model. 2024. 2

[43] William Peebles and Saining Xie. Scalable diffusion models
with transformers. In ICCV, pages 4195–4205, 2023. 3
[44] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Bj¨orn Ommer. High-resolution image syn-
thesis with latent diffusion models. In CVPR, pages 10684–
10695, 2022. 5

[45] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao
Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li,
Yang Wu, et al. Deepseekmath: Pushing the limits of math-
ematical reasoning in open language models. arXiv preprint
arXiv:2402.03300, 2024. 2, 5

[46] Xiaoyu Shi, Zhaoyang Huang, Fu-Yun Wang, Weikang Bian,
Dasong Li, Yi Zhang, Manyuan Zhang, Ka Chun Cheung,
Simon See, Hongwei Qin, et al. Motion-i2v: Consistent and
controllable image-to-video generation with explicit motion
modeling. In ACM SIGGRAPH, pages 1–11, 2024. 2
[47] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An,
Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual,
Oran Gafni, et al. Make-a-video: Text-to-video generation
without text-video data. arXiv preprint arXiv:2209.14792,
2022. 3

[48] Vincent Sitzmann, Semon Rezchikov, Bill Freeman, Josh
Tenenbaum, and Fredo Durand. Light field networks: Neu-
ral scene representations with single-evaluation rendering. In
NeurIPS, pages 19313–19325, 2021. 3

[49] Hunyuan Foundation Model Team. Hunyuanvideo: A sys-
tematic framework for large video generative models, 2024.
3

[50] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi
Fruchter. Diffusion models are real-time game engines.
arXiv preprint arXiv:2408.14837, 2024. 2

[51] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao,
Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao

10

[64] Jiwen Yu, Yiran Qin, Xintao Wang, Pengfei Wan, Di Zhang,
and Xihui Liu. Gamefactory: Creating new games with gen-
erative interactive videos. arXiv preprint arXiv:2501.08325,
2025. 2

[65] Shangzhan Zhang, Jianyuan Wang, Yinghao Xu, Nan Xue,
Christian Rupprecht, Xiaowei Zhou, Yujun Shen, and Gor-
don Wetzstein. Flare: Feed-forward geometry, appearance
and camera estimation from uncalibrated sparse views. arXiv
preprint arXiv:2502.12138, 2025. 3

[66] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe,
Learning
arXiv preprint

and Noah Snavely.
view synthesis using multiplane images.
arXiv:1805.09817, 2018. 6, 7

Stereo magnification:

Yang, et al. Wan: Open and advanced large-scale video gen-
erative models. arXiv preprint arXiv:2503.20314, 2025. 2,
3

[52] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea
Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Vi-
sual geometry grounded transformer. In CVPR, pages 5294–
5306, 2025. 2, 3

[53] Jiepeng Wang, Zhaoqing Wang, Hao Pan, Yuan Liu, Dong-
dong Yu, Changhu Wang, and Wenping Wang. Mmgen: Uni-
fied multi-modal image generation and understanding in one
go. arXiv preprint arXiv:2503.20644, 2025. 2

[54] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris
Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vi-
sion made easy. In CVPR, pages 20697–20709, 2024. 3
[55] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang,
Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chun-
hua Shen, and Tong He. π3: Permutation-equivariant visual
geometry learning. arXiv preprint arXiv:2507.13347, 2025.
2, 3, 6

[56] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li,
Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan.
Motionctrl: A unified and flexible motion controller for
video generation. In ACM SIGGRAPH, pages 1–11, 2024.
2, 3

[57] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen,
Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying
Shan, and Tien-Tsin Wong. Dynamicrafter: Animating
open-domain images with video diffusion priors. In ECCV,
pages 399–417. Springer, 2024. 3

[58] Dejia Xu, Yifan Jiang, Chen Huang, Liangchen Song,
Thorsten Gernoth, Liangliang Cao, Zhangyang Wang, and
Hao Tang. Cavia: Camera-controllable multi-view video
arXiv preprint
diffusion with view-integrated attention.
arXiv:2410.10774, 2024. 2, 3

[59] Dejia Xu, Weili Nie, Chao Liu, Sifei Liu, Jan Kautz,
Zhangyang Wang, and Arash Vahdat. Camco: Camera-
controllable 3d-consistent image-to-video generation. arXiv
preprint arXiv:2406.02509, 2024. 2, 3

[60] Hongwei Xue, Tiankai Hang, Yanhong Zeng, Yuchong Sun,
Bei Liu, Huan Yang, Jianlong Fu, and Baining Guo. Ad-
vancing high-resolution video-language representation with
large-scale video transcriptions. In CVPR, pages 5036–5045,
2022. 3

[61] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting
Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo,
Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual
generation. arXiv preprint arXiv:2505.07818, 2025. 2, 5
[62] Jianing Yang, Alexander Sax, Kevin J Liang, Mikael Henaff,
Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt
Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images
in one forward pass. In CVPR, pages 21924–21935, 2025. 3
[63] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu
Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiao-
han Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video
diffusion models with an expert transformer. arXiv preprint
arXiv:2408.06072, 2024. 3

11

