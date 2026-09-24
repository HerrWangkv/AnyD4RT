Gen3R: 3D Scene Generation Meets Feed-Forward Reconstruction

Jiaxin Huang1, Yuanbo Yang1, Bangbang Yang2, Lin Ma2, Yuewen Ma2, Yiyi Liao1(cid:66)
1 Zhejiang University 2 ByteDance
Project Page: https://xdimlab.github.io/Gen3R/

6
2
0
2

r
a

M
3
2

]

V
C
.
s
c
[

2
v
0
9
0
4
0
.
1
0
6
2
:
v
i
X
r
a

Figure 1. Gen3R bridges foundational reconstruction models with 2D video diffusion, enabling the joint generation of 2D videos and their
corresponding geometry in various settings.

Abstract

generative models.

We present Gen3R, a method that bridges the strong priors
of foundational reconstruction models and video diffusion
models for scene-level 3D generation. We repurpose the
VGGT reconstruction model to produce geometric latents
by training an adapter on its tokens, which are regularized
to align with the appearance latents of pre-trained video
diffusion models. By jointly generating these disentangled
yet aligned latents, Gen3R produces both RGB videos and
corresponding 3D geometry, including camera poses, depth
maps, and global point clouds. Experiments demonstrate
that our approach achieves state-of-the-art results in single-
and multi-image conditioned 3D scene generation. Addi-
tionally, our method can enhance the robustness of recon-
struction by leveraging generative priors, demonstrating
the mutual benefit of tightly coupling reconstruction and

(cid:66) Corresponding author.

1. Introduction

3D scene generation has become a fundamental problem
in computer vision and graphics, with wide applications in
simulation, gaming, robotics, and virtual reality. A method
capable of producing photorealistic and geometrically con-
sistent 3D scenes would enable the creation of immersive
environments at scale, serving as essential training data and
providing new tools for creative content design.

Prior methods attempt to extend 2D generative models
via score distillation [33, 47, 64, 78], incremental outpaint-
ing [9, 13, 91, 92], or multi-view synthesis followed by
reconstruction [14, 16, 37, 54, 60, 81]. Despite promis-
ing results, these methods often suffer from poor geometric
structure or high optimization cost. More recently, several
works [15, 28, 39, 63, 87, 97] have extended video diffu-
sion frameworks to feed-forward 3D scene generation for

 
 
 
 
 
 
improved efficiency. These approaches typically follow the
Latent Diffusion Model paradigm, training a VAE to learn
a compact latent space for 3D scenes and applying diffu-
sion within that space. However, the scarcity of large-scale
3D ground truth makes learning geometry-centric VAEs
highly challenging. One line of methods trains a VAE to re-
construct geometry from RGB inputs while simultaneously
learning a compressed latent representation [28, 56, 87]. Yet
this is inherently difficult, especially when supervision is
limited to 2D signals, which often results in suboptimal ge-
ometry and constrained generation quality.

In parallel, transformer-based feed-forward reconstruc-
tion models, such as Dust3R [74] and VGGT [72], have
shown strong reconstruction ability from 2D images. Re-
cent works attempt to build better VAEs by compressing
their 3D output [63, 97], but overlook a key fact: these re-
construction models already operate in a spatially compact
token space that encodes rich multi-view geometric infor-
mation, including depth, camera pose, and global structure.
This observation raises a central question: Can the intrinsic
latent manifold learned by reconstruction models be used to
fully exploit reconstruction priors for 3D scene generation?
Building on this insight, we introduce Gen3R, a 3D-
aware scene generation method that unifies advanced recon-
struction and generation models for jointly generating con-
trollable video and globally consistent 3D point clouds. Our
key idea is to recast a feed-forward reconstruction model,
VGGT [72], as a VAE-like provider of geometric latents and
combine these with appearance latents from a pre-trained
video diffusion model for joint generation. This allows us
to marry the rich geometric priors learned by reconstruc-
tion models over multiple 3D quantities with the strong
RGB priors of video diffusion models, effectively combin-
ing the strengths of both. To achieve so, we first project
the reconstruction model’s intermediate tokens to match the
spatial-temporal resolution of the appearance latents using
a learned adapter. Notably, simply compressing the tokens
is insufficient as their distribution significantly differs from
the corresponding appearance latents. We therefore propose
to align the two latent spaces, followed by fine-tuning a
video diffusion model for joint generation. By keeping ge-
ometric and appearance latents disentangled while aligning
their distributions, Gen3R demonstrates that the latent man-
ifold learned by reconstruction models can indeed serve as
a strong foundation for high-fidelity 3D scene generation.

Our framework supports flexible conditioning, enabling
generation from single or multiple input views, with or
without camera cues, as well as feed-forward scene recon-
struction within one unified model. It produces temporally
coherent RGB videos and globally aligned point clouds
across diverse configurations.

tion models, combining strong RGB priors with rich geo-
metric priors for 3D scene generation. 2) A disentangled
yet aligned appearance and geometry latent space, enabling
controllable and multi-view consistent scene synthesis. 3) A
flexible pipeline capable of handling various input settings,
producing high-fidelity videos and globally consistent 3D
point clouds.

2. Related Work

3D Scene Generation fom 2D Priors. A common strategy
for 3D scene generation is to leverage pretrained 2D gener-
ative models [53] to provide RGB priors. One line of work
employs score distillation sampling (SDS) [33, 47, 64, 78],
directly optimizing a 3D representation such as NeRF [1–
3, 42] and 3DGS [25, 46, 82, 89] to align with the distribu-
tion of a 2D diffusion model. Another line of methods first
synthesizes multi-view images using pretrained 2D diffu-
sion models, followed by 3D reconstruction through multi-
view synthesis [7, 14, 16, 36, 37, 54, 60, 62, 81, 99] or in-
cremental outpainting [9, 13, 58, 91, 92]. Both paradigms
leverage the strong RGB priors of 2D models but are lim-
ited by the lack of explicit 3D reasoning, often resulting in
inconsistent geometry, weak multi-view fidelity, and high
computational cost. Our method tackles this challenge by
bridging rich geometric priors of a reconstruction founda-
tion model with a 2D generative model.

Feed-Forward 3D Scene Generation. Object-level feed-
forward 3D generation methods [32, 40, 52, 84, 96] have
gained great success thanks to the large-scale 3D ground
truth datasets [10, 31]. However, extending this success
to full-scene 3D generation is challenging because high-
quality scene-level data is difficult to obtain. A practical
alternative is to synthesize a 3D representation in a feed-
forward manner and train it using only 2D supervision. Re-
cent works [15, 28, 29, 44, 57, 79, 87] follow this strategy
by generating Gaussians and using differentiable render-
ing to train directly on 2D images, thereby avoiding costly
3D data collection. However, these methods often strug-
gle with intricate geometric details and multi-view consis-
tency due to the lack of explicit 3D supervision. Other ap-
proaches [45, 63, 66, 97] address this limitation by lever-
aging off-the-shelf dense reconstruction models [26, 74] or
Unreal Engine to obtain 3D data for training. In contrast to
methods directly compressing the 3D output of reconstruc-
tion models [66, 97], our approach treats the reconstruction
model as an asymmetric VAE that encodes images into ge-
ometry latents, allowing us to inherit the strong geometric
priors across multiple 3D quantities and the high-level scene
understanding from the foundation geometry model.

Our contributions are threefold: 1) A novel framework
integrating video diffusion models with geometric founda-

Feed-forward 3D Scene Reconstruction. Traditional 3D
scene reconstruction pipelines [6, 43, 55, 67] have recently

Figure 2. Method. Left: We recast an advanced transformer-based feed-forward reconstruction model, VGGT, as a VAE to produce
geometry latents G by training an adapter on its latent tokens. The training is supervised with a reconstruction loss Lrec, along with a
regularization term LKL that aligns G with the appearance latent A, which is obtained from the VAE of a pre-trained video diffusion
model, WAN. Right: We fine-tune the video diffusion model to jointly generate geometry and appearance latents, Z = [A; G], under
various conditioning signals. At inference, varying the conditioning enables the generation of RGB videos and multiple 3D quantities,
including global point clouds, depth maps, and camera parameters, from a single or multiple frames, as well as performing reconstruction.

been complemented by learning-based methods [12, 19, 26,
65, 71, 73, 74, 94, 95], which leverage neural architectures
to capture structural regularities of the world. Among these,
seminal works such as [26, 74] demonstrated the ability
to infer geometrically consistent point clouds from uncal-
ibrated images. Recent advances, exemplified by [24, 72],
provide unified frameworks that jointly estimate camera pa-
rameters, dense geometry and point tracks. Subsequent
studies have further extended VGGT to new scene repre-
sentations [22, 27, 38, 70] or addressed its inherent limita-
tions [59, 76, 102].

Our method integrates the geometric prior of such feed-
forward reconstruction models [72] with a generative diffu-
sion model [18, 69, 88]. Different from prior reconstruc-
tion approaches [22, 59, 72, 74, 76], our method is in-
herently generative, capable of synthesizing coherent 3D
scenes from 1 or 2 views. Moreover, our method can also
be used for performing reconstruction and is able to miti-
gate errors of the original reconstruction model.

3. Method

Our goal is to generate high-fidelity 3D scenes with consis-
tent geometry and controllable cameras given one or more
images. To achieve this, we propose Gen3R, a 3D-aware la-
tent diffusion method bridging foundational reconstruction
models with pre-trained video diffusion models.

Specifically, we first design a unified latent space for
appearance and geometry by recasting the geometry fea-
tures of the feed-forward reconstruction model, VGGT, into
the latent space of a video diffusion model (Sec. 3.1). We
then fine-tune the video diffusion model to jointly gener-
ate the appearance and geometry latents under various con-
ditions (Sec. 3.2). Finally, these latents are decoded sep-
arately into RGB frames and scene geometry, including

global point clouds, depth maps and camera parameters
(Sec. 3.3). Fig. 2 illustrates our overall architecture.

3.1. Geometry Adapter for Unified Latent Space

Preliminary. VGGT [72], denoted as F, is a transformer-
based feed-forward reconstruction model that directly in-
fers multiple key 3D quantities of a scene from observed
It takes N input images I ∈ RN ×H×W ×3 and
views.
encodes them into high-dimensional geometry tokens V ∈
RN ×hv×wv×C by its encoder EV , which consists of 24 at-
tention blocks:

EV : I → V ∈ RN ×L×hv×wv×C,

(1)

where hv × wv is the token’s spatial resolution, C = 2048,
and L = 4 is the number of intermediate transformer
tokens—specifically those from the 4th, 11th, 17th and 23rd
blocks [86]—for subsequent decoding. For simplicity, we
omit the camera tokens of VGGT in the description; please
refer to Supplementary for details.

The geometry tokens V are then decoded by several indi-
vidual DPT heads [49] DV into multi-modal dense predic-
tions, such as point clouds P ∈ RN ×H×W ×3, depth maps
D ∈ RN ×H×W ×1 and camera parameters T ∈ RN ×9:

DV : V → (P, D, T ).

(2)

Token-to-Latent Adapter. We aim to recast VGGT [72]
as an asymmetric geometry VAE that takes as input N
RGB images I ∈ RN ×H×W ×3, produces geometric la-
tents G ∈ Rn×h×w×c for diffusion-based generation, and
decodes them into multi-modal geometric outputs, includ-
ing globally consistent point clouds ˆP ∈ RN ×H×W ×3, per-
view depth maps ˆD ∈ RN ×H×W ×1 and camera parame-
ters ˆT ∈ RN ×9. Since the latent space of a video diffu-
sion model exhibits a different spatial-temporal resolution

t
u
p
n
I

]
6
6
[

r
e
h
t
e
A

]
7
9
[

D
V
W

s
r
u
O

Figure 3. Qualitative Comparison of Geometry Generation in the 1-view based setting.

from VGGT tokens and operates in a substantially lower-
dimensional feature space (e.g., c = 16 [69]), we train an
adapter (Eadp, Dadp) to bridge the two feature spaces by
mapping the geometric tokens V into the latent space of the
video diffusion model and project them back:

Eadp : V → G ∈ Rn×h×w×c,
Dadp : G → V ∈ RN ×L×hv×wv×C,

(3)

(4)

where n × h × w is the downsampled resolution.

The resulting geometric latents G share the same spatial-
temporal resolution and feature dimension as those of the
video diffusion model, enabling joint generation of appear-
ance and geometry within a unified latent space.

Training of the Adapter. Our adapter is trained with a re-
construction loss and a distribution alignment loss wrt. the
appearance latents:

L = λ1Lrec + λ2LKL.

(5)

Specifically, the reconstruction loss enforces the recon-
structed geometry tokens ˆV = Dadp(G) to match the origi-
nal tokens V, and further regularize the consistency between
the decoded outputs ( ˆP, ˆD, ˆT ) and those derived from the
original tokens (P, D, T ) by the pretrained DPT heads:
Lrec =E(cid:2)∥ ˆV − V∥2(cid:3) + E(cid:2)∥ ˆT − T ∥1

(cid:3)

+E(cid:2)∥ ˆD − D∥2(cid:3) + E(cid:2)∥ ˆP − P∥2(cid:3).

(6)

Furthermore, we observed in practice that although this su-
pervision alone effectively compresses the geometry tokens,
it does not constrain the mapped latent space, which hin-
ders diffusion training from converging and degrades gen-
eration quality. While most LDMs use VAE or VQ-VAE to
constrain the latents, we propose to directly regularize our

latent space by aligning it with the pretrained appearance
latent distribution. Specifically, we impose a KL loss on the
geometry adapter, encouraging its latent distribution qG to
align with the pretrained RGB latent distribution qA:

LKL = DKL(qG ∥ qA).
(7)
This constraint ensures compatibility between the two la-
tent spaces and facilitates simultaneous modeling of both
appearance and geometry distributions.

3.2. Geometry-Aware Joint Latent Generation

Design of the Joint Latent Space. After training of the
adapter, we establish a compact latent space where geome-
try and appearance latents can be jointly processed. We then
fine-tune a video diffusion model [69], denoted as Gθ, to
generate both modalities of latents within this unified space.
Specifically, we aim to generate latent codes Z consist-
ing of two components: appearance latents A = EW (I) ∈
Rn×h×w×c and geometry latents G ∈ Rn×h×w×c. To avoid
introducing additional trainable parameters and to preserve
the pretrained video diffusion model’s generative capabil-
ity, we concatenate the two latents along the width dimen-
sion [8] to form a unified latent representation:

Z = [A; G] ∈ Rn×h×2w×c,
where [·; ·] denotes concatenation in the width dimension.

(8)

Training of the Diffusion Model.
To enhance control-
lability, we incorporate multiple condition signals into the
diffusion process, including a text prompt y, a condition
image sequence Icond with a flexible number of available
frames (where missing images are set to zero), correspond-
ing binary masks M and optional per-view camera condi-
tions Tcond. The overall diffusion process is defined as:

Gθ : (Zt; t, y, Icond, M, Tcond) → ˆZt−1,

(9)

.

d
n
o
C

w
e
i
v
-
1

w
e
i
v
-
2

Method

LVSM [23]
Gen3C [51]
GF [80]
Aether [66]
WVD [97]
Ours

DepthSplat [85]
LVSM [23]
Gen3C [51]
GF [80]
Aether [66]
WVD [97]
Ours

RealEstate10K

DL3DV-10K

PSNR ↑

SSIM ↑

LPIPS ↓

I2V Subj. ↑

I2V BG ↑

I.Q. ↑

PSNR ↑

SSIM ↑

LPIPS ↓

I2V Subj. ↑

I2V BG ↑

I.Q. ↑

18.97
20.26
16.32
16.57
17.62
20.51

26.67
29.58
23.83
23.28
21.77
23.78
27.05

0.7161
0.7186
0.5434
0.6374
0.6658
0.7388

0.8711
0.9197
0.8340
0.7426
0.7645
0.7948
0.8732

0.2992
0.2302
0.3803
0.3808
0.3300
0.2281

0.1742
0.1060
0.1947
0.2098
0.2241
0.1949
0.1352

0.9946
0.9931
0.9882
0.9927
0.9935
0.9951

0.9909
0.9954
0.9936
0.9893
0.9919
0.9935
0.9948

0.9933
0.9927
0.9789
0.9910
0.9932
0.9952

0.9867
0.9943
0.9930
0.9798
0.9901
0.9926
0.9946

0.4923
0.5200
0.5614
0.5419
0.5847
0.5993

0.4379
0.5173
0.5191
0.5614
0.5258
0.5795
0.6025

15.61
16.21
12.05
13.82
14.25
16.38

16.83
18.80
17.91
14.39
15.68
15.72
18.59

0.5384
0.5557
0.3458
0.5167
0.4848
0.5821

0.6094
0.6404
0.6120
0.4152
0.5565
0.5522
0.6149

0.4434
0.4575
0.5801
0.5272
0.5063
0.4234

0.3971
0.3575
0.4207
0.5160
0.4555
0.4510
0.3416

0.9635
0.9427
0.9335
0.9589
0.9531
0.9657

0.9505
0.9704
0.9470
0.9050
0.9619
0.9520
0.9685

0.9675
0.9547
0.9307
0.9653
0.9613
0.9715

0.9532
0.9736
0.9566
0.9110
0.9676
0.9597
0.9725

0.4262
0.4204
0.5410
0.4571
0.5466
0.5497

0.3855
0.4616
0.4239
0.4911
0.4708
0.5584
0.5623

Table 1. Quantitative Comparison of Appearance Generation. We compare both 1-view and 2-view based settings.

where Zt is the noised latent at timestep t, ˆZt−1 is the pre-
dicted latent at t − 1.

Note that we do not provide geometric latents as con-
dition signals, allowing the model to handle diverse tasks
from input images only. During training, we uniformly
sample conditions from the following options: (1) the first
frame (1-view based), (2) the first and last frames (2-view
based), (3) all frames, and adjust the binary masks corre-
spondingly. We also randomly drop camera conditions to
ensure they can be omitted during inference.

Inference.
Practically, we evaluate on three condition-
ing settings: (1) 1-view-based generation, (2) 2-view-based
generation, and (3) feed-forward reconstruction with a im-
age sequence. Each setting can be performed with or with-
out camera conditions. For fairness, we remove the camera
conditions in the feed-forward reconstruction experiments.

3.3. Decoding Latents into Scene Attributes

Based on the pipeline described above, we achieve feed-
forward 3D scene generation by sampling unified latents
from noise using Gθ, and decoding them into RGB frames
and geometry attributes using separate decoders.

The appearance latents A ∈ Rn×h×w×c are decoded
by the pretrained RGB VAE DW to synthesize photore-
alistic video frames I. Similarly, the geometry latents
G ∈ Rn×h×w×c are mapped by the geometry adapter Dadp
to recover geometry tokens V. These tokens are then de-
coded by pretrained VGGT heads DV to obtain scene at-
tributes, including globally consistent point clouds P, per-
view depth maps D and camera parameters T . Following
VGGT, we unproject the depth maps using the generated
camera parameters as the final geometry results.

4. Experiments

In this section, we compare our method with state-of-the-
art approaches across various conditions. We first describe
the training details in Sec. 4.1, followed by both quantita-
tive and qualitative evaluations on 3D generation (Sec. 4.2)

and reconstruction (Sec. 4.3). Finally, we present abla-
tion studies to further validate the effectiveness of our ap-
proach in Sec. 4.4. We highlight the best , second-best ,
and third-best scores achieved on any metrics.

4.1. Training Details

Datasets.
We train our model on a diverse collec-
tion of 3D datasets with camera calibrations, including:
RealEstate10K [101], DL3DV-10K [34], ACID [35], Tar-
tanAir [75], KITTI-360 [30], Waymo [61], Co3Dv2 [50],
MVImgNet [93], Virtual KITTI 2 [4] and WildRGB-D [83].
Together, these datasets provide over 300k multi-view con-
sistent 3D scenes, spanning a wide range of domains, in-
cluding object-centric, indoor, outdoor, driving, and syn-
thetic scenarios. For RealEstate10K, we follow the official
train-test split, while for the other datasets, we randomly
sample around 90% of the scenes for training and use the
rest for testing. Text prompts for each scene are generated
using a multi-modal large language model [5]. Notably, our
method does not require explicit 3D GT representations for
training, which are unavailable for many datasets.

Implementation Details.
For the geometry adapter, we
adopt a causal autoencoder architecture similar to [69], but
with different input, output, and hidden dimensions. The
adapter is trained on the mixed dataset described above. To
ensure stability, the model is initially trained with 25 frames
at a resolution of 560 × 560 for 15k iterations, using a batch
size of 2 and gradient accumulation steps of 4 on 24 H20
GPUs, resulting a total batch size of 192. It is then fine-
tuned with 49 frames for another 6k iterations, using a batch
size of 1 and gradient accumulation steps of 8 on the same
hardware. The adapter weights are randomly initialized.

For the video diffusion model, we fine-tune a pretrained
image-camera conditioned Wan2.1 [69]. Similar to the ge-
ometry adapter, during training we randomly sample 49
consecutive frames from each video clip, which are then re-
sized and center-cropped to 560×560. The model is trained
for 8k iterations with a batch size of 4 on 24 H20 GPUs.

.

d
n
o
C

w
e
i
v
-
1

w
e
i
v
-
2

Method

Aether [66]
WVD [97]
VGGT [72]
Ours

Aether [66]
WVD [97]
VGGT [72]
Ours

Co3Dv2

WildRGB-D

TartanAir

Accuracy ↓

Completeness ↓

CD ↓

Accuracy ↓

Completeness ↓

CD ↓

Accuracy ↓

Completeness ↓

CD ↓

1.2630
1.8038
0.3291
0.8284

0.9664
2.1153
0.3951
0.7237

2.6366
1.4237
4.3830
1.3811

2.1376
1.3009
2.1566
1.2298

1.9498
1.6137
2.3561
1.1047

1.5520
1.7081
1.2759
0.9767

0.3181
0.2708
0.0346
0.1581

0.3536
0.2483
0.0276
0.1109

0.2951
0.2562
0.6723
0.2402

0.2540
0.1813
0.2650
0.1744

0.3066
0.2635
0.3534
0.1992

0.3038
0.2148
0.1463
0.1426

3.1547
4.3944
0.7379
3.0250

2.7745
4.3794
0.9201
2.2825

4.5366
3.0660
5.3595
2.5367

3.4420
2.5268
3.2554
1.6643

3.8457
3.7302
3.0487
2.7809

3.1082
3.4531
2.0877
1.9734

Table 2. Quantitative Comparison of Geometry Generation. We compare both 1-view and 2-view based settings.

Inputs

LVSM [23]

DepthSplat [85]

Gen3C [51]

WVD [97]

Ours

Ground Truth

Figure 4. Qualitative Comparison of Novel View Synthesis with 2-view conditions. The input images are shown on the left, and error
maps are displayed overlaid on the results. Bluer colors indicate smaller errors, while redder colors indicate larger errors.

To enhance capability in handling diverse conditioning in-
puts, each training step has 1
3 probability of using (i) 1-view
condition, (ii) 2-view (first-last frame) conditions, or (iii)
the entire frame sequence as input. Additionally, the text
prompt is dropped with a 20% probability for CFG [17],
and the camera condition is omitted with a 50% probability.

4.2. 3D Generation

Datasets and Metrics.
We evaluate 3D generation
on RealEstate10K [101], DL3DV-10K [34], Co3Dv2 [50],
WildRGB-D [83] and TartanAir [75] datasets. For each
task, we assess both appearance (RGB) and geometry (point
clouds) metrics. Appearance metrics are computed across
all these datasets, while geometry metrics are evaluated
only on Co3Dv2, WildRGB-D, and TartanAir, as the other
datasets do not provide ground truth geometry. Note
that we report appearance metrics only on RealEstate10K
and DL3DV-10K, and geometry metrics only on Co3Dv2,

WildRGB-D, and TartanAir in the main text. Please refer to
Supp. for complete results on all datasets.

For appearance evaluation, we randomly sample 200
the
sequences with camera conditions from each of
RealEstate10K and DL3DV-10K, and compute PSNR,
SSIM [77], and LPIPS [98] between the generated and
ground-truth images. We additionally report the VBench
Score [20, 21] to assess the models’ generative capability,
focusing on I2V Subject (I2V Subj.), I2V Background (I2V
BG), and Imaging Quality (I.Q.) given the presence of
image-based conditioning.

For geometry evaluation, we randomly sample 300 se-
quences with camera conditions from each of the Co3Dv2
and WildRGB-D, along with an additional 80 sequences
from TartanAir. We first use the Umeyama algorithm [68]
to align the generated point clouds to the ground truth, then
sample 20k points from both point clouds using Farthest
Point Sampling (FPS) [48], and finally compute Accuracy,
Completeness, and Chamfer Distance (CD) [11].

Method

Co3Dv2

WildRGB-D

TartanAir

Accuracy ↓ Completeness ↓

CD ↓

Accuracy ↓ Completeness ↓

CD ↓

Accuracy ↓ Completeness ↓

CD ↓

VGGT [72]
WVD (VAE only)
Ours (VAE only)
Aether [66]
WVD [97]
Ours

0.9157
1.0533
0.9236
1.7755
1.7997
0.9270

1.0107
1.3627
1.0735
1.2280
1.4609
0.9980

0.9632
1.2080
0.9986
1.5018
1.6303
0.9625

0.0925
0.1273
0.0929
0.3033
0.2758
0.1058

0.1405
0.1780
0.1400
0.1665
0.1542
0.1463

0.1165
0.1526
0.1165
0.2349
0.2150
0.1260

2.2929
3.5337
2.2972
3.0287
3.8018
1.9243

0.8985
2.0396
1.0063
2.3684
2.0820
1.0959

1.5957
2.7867
1.6518
2.6985
2.9419
1.5101

Table 3. Quantitative Comparison of Geometry Reconstruction. WVD (VAE only) uses pretrained RGB VAE to encode and reconstruct
point clouds, while Ours (VAE only) projects and reconstruct VGGT tokens to decode scene geometry.

Input

VGGT

VAE (Ours)

Ours

Figure 5. Quali. Comparison of Geometry Reconstruction.

Comparison Baselines. We compare our method with
several state-of-the-art approaches that use image and cam-
era conditions, including 1) Reconstruction-based method:
DepthSplat [85]; 2) 2D generation methods: LVSM [23],
Gen3C [51], and Geometry Forcing (GF) [80]; and 3) Ex-
plicit 3D generation methods: Aether [66] and WVD [97].
We use the official implementations for all of these methods
except for WVD, as it is not open-sourced; we re-implement
it following the same training strategy as ours. Note that
Aether does not output point maps, so we back-project its
generated depths using predicted camera parameters to ob-
tain point clouds.

Comparison on Appearance Generation.
Tab. 1
presents quantitative results for 1-view-based and 2-view-
based (first-last frames) appearance generation. Fig. 4 pro-
vides the corresponding qualitative comparisons for the 2-
view setting (see Supp. for 1-view results). Gen3R outper-
forms or matches the baselines in most cases.
1) Reconstruction-based methods: We evaluate DepthSplat
only in the 2-view setting, as it requires multi-view inputs
to construct cost volumes. Although it achieves competitive
results, it leaves holes in occluded regions (see Fig. 4). In
contrast, our method can plausibly complete these regions
using diffusion-based generation.
2) 2D generation methods: LVSM performs best in the 2-
view case, as it is a non-generative model well suited for
interpolation. However, it often produces blurred results
in over-exposed scenes (Fig. 4, 1st row), and its perfor-
mance degrades notably in the 1-view case. Gen3C also
achieves competitive results in 1-view generation by com-
bining depth-based warping and inpainting, but its quality
heavily depends on depth accuracy, leading to misaligned
boundaries when the depth estimates are inaccurate. In ad-
dition, it sometimes exhibits color differences from the in-
put image, as shown in Fig. 4. More relevant to our method,

Cond.

RealEstate10K

DL3DV-10K

Co3Dv2 WildRGB-D TartanAir

Method

PSNR ↑

SSIM ↑ LPIPS ↓

PSNR ↑

SSIM ↑ LPIPS ↓

CD ↓

CD ↓

CD ↓

1-view

2-Stage
w/o LKL
Ours

2-view

2-Stage
w/o LKL
Ours

17.38
16.31
20.51

23.56
21.62
27.05

0.6617
0.6476
0.7388

0.3412
0.3941
0.2281

0.7883
0.7592
0.8732

0.1931
0.2185
0.1352

14.37
13.68
16.38

15.92
15.41
18.59

0.5085
0.4797
0.5821

0.5014
0.5094
0.4234

1.6223
1.9620
1.1047

0.5413
0.5222
0.6149

0.4427
0.4527
0.3416

1.3615
1.7144
0.9767

0.2330
0.3280
0.1992

0.1623
0.2898
0.1426

3.7029
4.0395
2.7809

2.6579
3.5508
1.9734

Table 4. Ablation Study on appearance and geometry generation.

GF similarly attempts to bridge reconstruction and genera-
tion. Unlike ours, which aligns latent spaces prior to dif-
fusion training, GF aligns intermediate diffusion features to
the reconstruction model during training, which is less ef-
fective in practice. Finally, all of the above baselines oper-
ate purely in 2D and do not produce any 3D outputs.
3) Explicit 3D generation methods: Our method clearly
surpasses the most relevant generative baselines, Aether
and WVD, both of which jointly generate RGB images
and scene geometry. As shown in Fig. 4, our approach
yields higher-quality results and better camera alignment
than WVD. This highlights the advantage of bridging recon-
struction and generation models in the latent space, rather
than compressing the reconstruction outputs for generation.

Comparison on Geometry Generation. We further com-
pare the generated point clouds with 3D-based methods
in Tab. 2, and Fig. 3 visualizes point clouds generated from
a single input view. Our method clearly outperforms Aether
and WVD in CD across both generation settings. The qual-
itative results in Fig. 3 are consistent with the quantitative
findings: Aether and WVD exhibit poor global consistency,
whereas our method produces more complete objects and
scenes from single-view observations, with plausible geom-
etry in unseen regions. We also include VGGT in Tab. 2 as a
reference. It performs pure reconstruction from one or two
input views without generation. Although VGGT achieves
better accuracy, it suffers from lower completeness since it
does not generate geometry for novel views, leading to a
worse CD than Gen3R.

4.3. Feed-forward 3D Reconstruction

Dataset and Metrics. Similar to 3D generation, we evalu-
ate feed-forward 3D reconstruction on the same sequences
sample from Co3Dv2, WildRGB-D and TartanAir datasets

Inputs

2-Stage

w/o LKL
Figure 6. Appearance Comparison with ablation baselines. We
highlight the artifacts of the baselines directly in the figures.

Ours

GT

Method

Aether [66]
WVD [97]
2-Stage
w/o LKL
Ours

. RealEstate10K WildRGB-D
d
n
o
C

AUC@30 ↑

AUC@30 ↑

. RealEstate10K WildRGB-D
d
n
o
C

AUC@30 ↑

AUC@30 ↑

w
e
i
v
-
1

0.6398
0.6727
0.6832
0.4100
0.7443

0.5375
0.6780
0.6798
0.4759
0.8004

w
e
i
v
-
2

0.6220
0.7249
0.7188
0.4683
0.7732

0.5068
0.7113
0.7211
0.4947
0.8098

Table 5. Quantitative Comparison of Camera Controllability
on RealEstate10K and WildRGB-D datasets.

as before, but without camera conditions. We assess both
geometric quality and camera pose estimation. For ge-
ometry, we first align the predicted point clouds with GT,
downsample both using FPS, and then compute Accuracy,
Completeness, and Chamfer Distance (CD). For camera
pose estimation, we use the RealEstate10K and WildRGB-
D datasets and follow VGGT [72] in reporting AUC@30,
which combines both Relative Rotation Accuracy (RRA)
and Relative Translation Accuracy (RTA). Please refer to
Supplementary for camera pose estimation results.

Comparison Baselines.
We compare our method
with 1) feed-forward 3D reconstruction approach, VGGT,
as well as different variants of VAE for compressing
VGGT. WVD (VAE only) encodes and decodes VGGT’s
global point clouds using a pre-trained RGB VAE, while
Ours (VAE only) encodes VGGT’s geometry tokens
through our adapter and decodes them back. We also com-
pare against 2) 3D generation methods Aether and WVD.

Comparison on Geometry Reconstruction. We present
quantitative results in Tab. 3. 1) Feed-forward 3D recon-
struction: Our VAE maintains the competitive performance
of VGGT, whereas WVD’s VAE produces subpar results
when encoding and reconstructing explicit point clouds,
as it is originally designed for RGB image reconstruction.
Moreover, our generative version even enhances the re-
construction performance. This improvement arises be-
cause our method jointly models the appearance and geom-
etry distribution, enabling mutual interactions between the
two modalities and thereby refining noisy geometric pre-
dictions. As shown in Fig. 5, VGGT occasionally exhibits
floaters in its predicted geometry, and our adapted VAE in-
herits these artifacts. However, our generative model cor-
rects the errors and produces cleaner depth. 2) 3D gener-

Input

RGB Latents

w/o LKL

Ours

Figure 7. Visualization of Latent Spaces from different VAEs.

ation methods: Our method significantly outperforms ex-
isting generative models, Aether and WVD, on the recon-
struction task, despite that our re-implemented WVD also
leverages the prior of VGGT.

4.4. Ablation Study

Effect of Joint Generation. We investigate the impact
of jointly generating RGB and geometry. To this end, we
design a 2-Stage baseline: a video diffusion model fine-
tuned on our training set generates only RGB under cam-
era control, while geometry is predicted separately using
VGGT from the generated images. Results in Tab. 4 show
that our joint generation approach outperforms the 2-Stage
pipeline in both appearance and geometry. This is because
the 2-Stage approach naively connects 2D generation with
3D reconstruction, leading to accumulated errors. Besides,
Tab. 5 and Fig. 6 further show that our method outperforms
this 2-Stage alternative in terms of camera control accuracy.

Effect of the Distribution Alignment Loss. We further
evaluate the impact of our distribution alignment loss LKL
by training a variant of the adapter without it and visualiz-
ing the resulting latents in Fig. 7. Without this constraint,
the geometry latents clearly deviate from the appearance la-
tents. Results in Tab. 4, Tab. 5, and Fig. 6 show that this mis-
alignment hinders convergence and significantly degrades
both camera controllability and generation quality.

5. Conclusion

We introduced Gen3R, a unified framework that couples
feed-forward reconstruction with video diffusion for high-
fidelity 3D scene synthesis. By reformulating VGGT as an
asymmetric geometry VAE with the geometry adapter and
aligning its latents with a video diffusion model, Gen3R
jointly generates RGB videos and globally consistent 3D
geometry. Extensive experiments show that Gen3R outper-
forms existing 2D and 3D based generative methods in both
appearance and geometry, while also delivering superior
camera controllability. Furthermore, Gen3R improves the
robustness of feed-forward reconstruction, highlighting the
mutual benefits of combining generative priors with strong
geometric foundations. We believe Gen3R offers a promis-
ing direction toward controllable and high-fidelity 3D scene
generation, and opens new possibilities for bridging recon-
struction and generative modeling at scale.

Acknowledgements

This work is supported by NSFC under grant 62441223.

References

[1] Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter
Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan.
Mip-nerf: A multiscale representation for anti-aliasing neu-
ral radiance fields. ICCV, 2021. 2

[2] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P.
Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded
anti-aliased neural radiance fields. CVPR, 2022. 4

[3] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P.
Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased grid-
based neural radiance fields. ICCV, 2023. 2

[4] Yohann Cabon, Naila Murray, and Martin Humenberger.

Virtual kitti 2, 2020. 5

[5] Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu
Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei
Chu, Xiaoyi Dong, Haodong Duan, Qi Fan, Zhaoye Fei,
Yang Gao, Jiaye Ge, Chenya Gu, Yuzhe Gu, Tao Gui, Aijia
Guo, Qipeng Guo, Conghui He, Yingfan Hu, Ting Huang,
Tao Jiang, Penglong Jiao, Zhenjiang Jin, Zhikai Lei, Jiax-
ing Li, Jingwen Li, Linyang Li, Shuaibin Li, Wei Li, Yining
Li, Hongwei Liu, Jiangning Liu, Jiawei Hong, Kaiwen Liu,
Kuikun Liu, Xiaoran Liu, Chengqi Lv, Haijun Lv, Kai Lv,
Li Ma, Runyuan Ma, Zerun Ma, Wenchang Ning, Linke
Ouyang, Jiantao Qiu, Yuan Qu, Fukai Shang, Yunfan Shao,
Demin Song, Zifan Song, Zhihao Sui, Peng Sun, Yu Sun,
Huanze Tang, Bin Wang, Guoteng Wang, Jiaqi Wang, Jiayu
Wang, Rui Wang, Yudong Wang, Ziyi Wang, Xingjian Wei,
Qizhen Weng, Fan Wu, Yingtong Xiong, Chao Xu, Ruil-
iang Xu, Hang Yan, Yirong Yan, Xiaogui Yang, Haochen
Ye, Huaiyuan Ying, Jia Yu, Jing Yu, Yuhang Zang, Chuyu
Zhang, Li Zhang, Pan Zhang, Peng Zhang, Ruijie Zhang,
Shuo Zhang, Songyang Zhang, Wenjian Zhang, Wenwei
Zhang, Xingcheng Zhang, Xinyue Zhang, Hui Zhao, Qian
Zhao, Xiaomeng Zhao, Fengzhe Zhou, Zaida Zhou, Jing-
ming Zhuo, Yicheng Zou, Xipeng Qiu, Yu Qiao, and Dahua
Lin. Internlm2 technical report, 2024. 5

[6] Carlos Campos, Richard Elvira, Juan J G´omez Rodr´ıguez,
Jos´e MM Montiel, and Juan D Tard´os. Orb-slam3: An ac-
curate open-source library for visual, visual–inertial, and
IEEE transactions on robotics, 37(6):
multimap slam.
1874–1890, 2021. 2

[7] Yuedong Chen, Chuanxia Zheng, Haofei Xu, Bohan
Zhuang, Andrea Vedaldi, Tat-Jen Cham, and Jianfei Cai.
Mvsplat360: Feed-forward 360 scene synthesis from sparse
views. Advances in Neural Information Processing Sys-
tems, 37:107064–107086, 2024. 2

[8] Zhaoxi Chen, Tianqi Liu, Long Zhuo, Jiawei Ren, Zeng
Tao, He Zhu, Fangzhou Hong, Liang Pan, and Ziwei Liu.
4dnex: Feed-forward 4d generative modeling made easy.
arXiv preprint arXiv:2508.13154, 2025. 4

[9] Jaeyoung Chung, Suyoung Lee, Hyeongjin Nam, Jaerin
Lee, and Kyoung Mu Lee. Luciddreamer: Domain-free

generation of 3d gaussian splatting scenes. arXiv preprint
arXiv:2311.13384, 2023. 1, 2

[10] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs,
Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana
Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse:
A universe of annotated 3d objects, 2022. 2

[11] Haoqiang Fan, Hao Su, and Leonidas Guibas. A point set
generation network for 3d object reconstruction from a sin-
gle image, 2016. 6

[12] Xianze Fang, Jingnan Gao, Zhe Wang, Zhuo Chen, Xingyu
Ren, Jiangjing Lyu, Qiaomu Ren, Zhonglei Yang, Xiaokang
Yang, Yichao Yan, and Chengfei Lyu. Dens3r: A foun-
dation model for 3d geometry prediction. arXiv preprint
arXiv:2507.16290, 2025. 3

[13] Rafail Fridman, Amit Abecasis, Yoni Kasten, and Tali
Dekel. Scenescape: Text-driven consistent scene genera-
tion. Advances in Neural Information Processing Systems,
36:39897–39914, 2023. 1, 2

[14] Ruiqi Gao, Aleksander Holynski, Philipp Henzler,
Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan,
Jonathan T Barron, and Ben Poole. Cat3d: Create anything
in 3d with multi-view diffusion models. arXiv preprint
arXiv:2405.10314, 2024. 1, 2

[15] Hyojun Go, Byeongjun Park, Jiho Jang, Jin-Young Kim,
Soonwoo Kwon, and Changick Kim. Splatflow: Multi-view
rectified flow model for 3d gaussian splatting synthesis. In
Proceedings of the Computer Vision and Pattern Recogni-
tion Conference, pages 21524–21536, 2025. 1, 2

[16] Junlin Hao, Peiheng Wang, Haoyang Wang, Xinggong
Zhang, and Zongming Guo. Gaussvideodreamer: 3d scene
generation with video diffusion and inconsistency-aware
gaussian splatting. arXiv preprint arXiv:2504.10001, 2025.
1, 2

[17] Jonathan Ho and Tim Salimans. Classifier-free diffusion

guidance, 2022. 6

[18] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu,
Cogvideo: Large-scale pretraining for
and Jie Tang.
text-to-video generation via transformers. arXiv preprint
arXiv:2205.15868, 2022. 3

[19] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou,
Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and
Hao Tan. Lrm: Large reconstruction model for single image
to 3d, 2024. 3

[20] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang
Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang
Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen,
Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench:
Comprehensive benchmark suite for video generative mod-
els. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, 2024. 6, 3

[21] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu,
Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang
Si, Yuming Jiang, Yaohui Wang, Xinyuan Chen, Ying-
Cong Chen, Limin Wang, Dahua Lin, Yu Qiao, and Zi-
wei Liu. Vbench++: Comprehensive and versatile bench-
mark suite for video generative models. arXiv preprint
arXiv:2411.13503, 2024. 6, 3

[22] Lihan Jiang, Yucheng Mao, Linning Xu, Tao Lu, Kerui
Ren, Yichen Jin, Xudong Xu, Mulin Yu, Jiangmiao Pang,
Feng Zhao, et al. Anysplat: Feed-forward 3d gaus-
sian splatting from unconstrained views. arXiv preprint
arXiv:2505.23716, 2025. 3

[23] Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi,
Tianyuan Zhang, Fujun Luan, Noah Snavely, and Zexiang
Xu. Lvsm: A large view synthesis model with minimal 3d
inductive bias, 2025. 5, 6, 7, 2, 3

[24] Nikhil Keetha, Norman M¨uller, Johannes Sch¨onberger,
Lorenzo Porzi, Yuchen Zhang, Tobias Fischer, Arno
Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes,
Jonathon Luiten, Manuel Lopez-Antequera, Samuel Rota
Bul`o, Christian Richardt, Deva Ramanan, Sebastian
Scherer, and Peter Kontschieder. Mapanything: Universal
feed-forward metric 3d reconstruction, 2025. 3

[25] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis. 3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics,
42(4), 2023. 2

[26] Vincent Leroy, Yohann Cabon, and J´erˆome Revaud.

Grounding image matching in 3d with mast3r, 2024. 2, 3

[27] Hao Li, Zhengyu Zou, Fangfu Liu, Xuanyang Zhang,
Fangzhou Hong, Yukang Cao, Yushi Lan, Manyuan Zhang,
Gang Yu, Dingwen Zhang, and Ziwei Liu. Iggt: Instance-
grounded geometry transformer for semantic 3d reconstruc-
tion, 2025. 3

[28] Xinyang Li, Zhangyu Lai, Linning Xu, Yansong Qu, Liu-
juan Cao, Shengchuan Zhang, Bo Dai, and Rongrong Ji. Di-
rector3d: Real-world camera trajectory and 3d scene gener-
ation from text. Advances in neural information processing
systems, 37:75125–75151, 2024. 1, 2

[29] Hanwen Liang, Junli Cao, Vidit Goel, Guocheng Qian,
Sergei Korolev, Demetri Terzopoulos, Konstantinos N. Pla-
taniotis, Sergey Tulyakov, and Jian Ren. Wonderland: Nav-
igating 3d scenes from a single image, 2025. 2

[30] Yiyi Liao, Jun Xie, and Andreas Geiger. Kitti-360: A novel
dataset and benchmarks for urban scene understanding in
2d and 3d, 2022. 5

[31] Chendi Lin, Heshan Liu, Qunshu Lin, Zachary Bright, Shi-
tao Tang, Yihui He, Minghao Liu, Ling Zhu, and Cindy Le.
Objaverse++: Curated 3d object dataset with quality anno-
tations, 2025. 2

[32] Chenguo Lin, Panwang Pan, Bangbang Yang, Zeming Li,
and Yadong Mu. Diffsplat: Repurposing image diffusion
models for scalable gaussian splat generation, 2025. 2

[33] Chen-Hsuan Lin,

Jun Gao, Luming Tang, Towaki
Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja
Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-
2023 IEEE/CVF
resolution text-to-3d content creation.
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 300–309, 2022. 1, 2

[34] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng
Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen
Lu, Xuanmao Li, Xingpeng Sun, Rohan Ashok, Aniruddha
Mukherjee, Hao Kang, Xiangrui Kong, Gang Hua, Tianyi
Zhang, Bedrich Benes, and Aniket Bera. Dl3dv-10k: A

large-scale scene dataset for deep learning-based 3d vision,
2023. 5, 6, 3

[35] Andrew Liu, Richard Tucker, Varun Jampani, Ameesh
Makadia, Noah Snavely, and Angjoo Kanazawa.
Infinite
nature: Perpetual view generation of natural scenes from
In Proceedings of the IEEE/CVF Inter-
a single image.
national Conference on Computer Vision, pages 14458–
14467, 2021. 5

[36] Fangfu Liu, Wenqiang Sun, Hanyang Wang, Yikai Wang,
Haowen Sun, Junliang Ye, Jun Zhang, and Yueqi Duan. Re-
conx: Reconstruct any scene from sparse views with video
diffusion model, 2024. 2

[37] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie
Liu, Taku Komura, and Wenping Wang. Syncdreamer:
Generating multiview-consistent images from a single-view
image. arXiv preprint arXiv:2309.03453, 2023. 1, 2
[38] Yang Liu, Chuanchen Luo, Zimo Tang, Junran Peng, and
Zhaoxiang Zhang. Vggt-x: When vggt meets dense novel
view synthesis, 2025. 3

[39] Yuanxun Lu, Jingyang Zhang, Tian Fang, Jean-Daniel Nah-
mias, Yanghai Tsin, Long Quan, Xun Cao, Yao Yao, and
Shiwei Li. Matrix3d: Large photogrammetry model all-
in-one. Computer Vision and Pattern Recognition (CVPR),
2025. 1

[40] Xuyi Meng, Chen Wang, Jiahui Lei, Kostas Daniilidis, Ji-
atao Gu, and Lingjie Liu. Zero-1-to-g: Taming pretrained
2d diffusion model for direct 3d generation, 2025. 2
[41] Ben Mildenhall, Pratul P. Srinivasan, Rodrigo Ortiz-Cayon,
Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and
Abhishek Kar. Local light field fusion: Practical view syn-
thesis with prescriptive sampling guidelines. ACM Trans-
actions on Graphics (TOG), 2019. 4

[42] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf.
Communications of the ACM, 65:99 – 106, 2020. 2
[43] Raul Mur-Artal, J. M. M. Montiel, and Juan D. Tardos.
Orb-slam: A versatile and accurate monocular slam system.
IEEE Transactions on Robotics, 31(5):1147–1163, 2015. 2
Lorenzo Porzi,
Samuel Rota Bul`o, Peter Kontschieder, and Matthias
Diffrf: Rendering-guided 3d radiance field
Nießner.
diffusion, 2023. 2

[44] Norman M¨uller, Yawar Siddiqui,

[45] Hieu T. Nguyen, Yiwen Chen, Vikram Voleti, Varun Jam-
pani, and Huaizu Jiang. Housecrafter: Lifting floorplans to
3d scenes with 2d diffusion model, 2025. 2

[46] Zhexi Peng, Tianjia Shao, Liu Yong, Jingke Zhou, Yin
Yang, Jingdong Wang, and Kun Zhou. Rtg-slam: Real-time
3d reconstruction at scale using gaussian splatting. 2024. 2
[47] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Milden-
hall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv
preprint arXiv:2209.14988, 2022. 1, 2

[48] Charles R. Qi, Li Yi, Hao Su, and Leonidas J. Guibas.
Pointnet++: Deep hierarchical feature learning on point sets
in a metric space, 2017. 6

[49] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vi-

sion transformers for dense prediction, 2021. 3

[50] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler,
Luca Sbordone, Patrick Labatut, and David Novotny. Com-
mon objects in 3d: Large-scale learning and evaluation of
real-life 3d category reconstruction. In International Con-
ference on Computer Vision, 2021. 5, 6, 2

[51] Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling,
Yifan Lu, Merlin Nimier-David, Thomas M¨uller, Alexan-
der Keller, Sanja Fidler, and Jun Gao. Gen3c: 3d-informed
world-consistent video generation with precise camera con-
trol, 2025. 5, 6, 7, 2, 3

[52] Barbara Roessle, Norman M¨uller, Lorenzo Porzi, Samuel
Rota Bul`o, Peter Kontschieder, Angela Dai, and Matthias
In SIG-
Nießner. L3dg: Latent 3d gaussian diffusion.
GRAPH Asia 2024 Conference Papers, page 1–11. ACM,
2024. 2

[53] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Bj¨orn Ommer. High-resolution image
synthesis with latent diffusion models, 2021. 2

[54] Kyle Sargent, Zizhang Li, Tanmay Shah, Charles Her-
rmann, Hong-Xing Yu, Yunzhi Zhang, Eric Ryan Chan,
Dmitry Lagun, Li Fei-Fei, Deqing Sun, et al. Zeronvs:
Zero-shot 360-degree view synthesis from a single real im-
age. 2023. 1, 2

[55] Johannes L Schonberger

Jan-Michael Frahm.
and
In Proceedings of
Structure-from-motion revisited.
the IEEE conference on computer vision and pattern
recognition, pages 4104–4113, 2016. 2

[56] Katja Schwarz, Seung Wook Kim, Jun Gao, Sanja Fidler,
Andreas Geiger, and Karsten Kreis. Wildfusion: Learning
3d-aware latent diffusion models in view space. In Inter-
national Conference on Learning Representations (ICLR),
2024. 2

[57] Katja Schwarz, Norman Mueller, and Peter Kontschieder.
Generative gaussian splatting: Generating 3d scenes with
video diffusion priors, 2025. 2

[58] Katja Schwarz, Denys Rozumnyi, Samuel Rota Bul`o,
Lorenzo Porzi, and Peter Kontschieder. A recipe for gener-
ating 3d worlds from a single image, 2025. 2

[59] You Shen, Zhipeng Zhang, Yansong Qu, and Liujuan Cao.
Fastvggt: Training-free acceleration of visual geometry
transformer. 2025. 3

[60] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie
Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d
generation, 2024. 1, 2

[61] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aure-
lien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo,
Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasude-
van, Wei Han, Jiquan Ngiam, Hang Zhao, Aleksei Timo-
feev, Scott Ettinger, Maxim Krivokon, Amy Gao, Aditya
Joshi, Sheng Zhao, Shuyang Cheng, Yu Zhang, Jonathon
Shlens, Zhifeng Chen, and Dragomir Anguelov. Scala-
bility in perception for autonomous driving: Waymo open
dataset, 2020. 5

[62] Wenqiang Sun, Shuo Chen, Fangfu Liu, Zilong Chen,
Yueqi Duan, Jun Zhang, and Yikai Wang. Dimensionx:
Create any 3d and 4d scenes from a single image with con-
In International Conference on
trollable video diffusion.
Computer Vision (ICCV), 2025. 2

[63] Stanislaw Szymanowicz, Jason Y. Zhang, Pratul Srinivasan,
Ruiqi Gao, Arthur Brussee, Aleksander Holynski, Ricardo
Martin-Brualla, Jonathan T. Barron, and Philipp Henzler.
Bolt3d: Generating 3d scenes in seconds, 2025. 1, 2
[64] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and
Gang Zeng. Dreamgaussian: Generative gaussian splat-
arXiv preprint
ting for efficient 3d content creation.
arXiv:2309.16653, 2023. 1, 2

[65] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei
Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-
view gaussian model for high-resolution 3d content cre-
ation. arXiv preprint arXiv:2402.05054, 2024. 3

[66] Aether Team, Haoyi Zhu, Yifan Wang, Jianjun Zhou,
Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen,
Chunhua Shen, Jiangmiao Pang, and Tong He. Aether:
Geometric-aware unified world modeling. arXiv preprint
arXiv:2503.18945, 2025. 2, 4, 5, 6, 7, 8, 1, 3

[67] Zachary Teed and Jia Deng. DROID-SLAM: Deep Visual
SLAM for Monocular, Stereo, and RGB-D Cameras. Ad-
vances in neural information processing systems, 2021. 2

[68] S. Umeyama. Least-squares estimation of transformation
parameters between two point patterns. IEEE Transactions
on Pattern Analysis and Machine Intelligence, 13(4):376–
380, 1991. 6

[69] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao,
Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianx-
iao Yang, et al. Wan: Open and advanced large-scale video
generative models. arXiv preprint arXiv:2503.20314, 2025.
3, 4, 5, 1, 2

[70] Chaoyang Wang, Ashkan Mirzaei, Vidit Goel, Willi
Menapace, Aliaksandr Siarohin, Avalon Vinella, Michael
Ivan Skorokhodov, Vladislav Shakhrai,
Vasilkovsky,
Sergey Korolev, Sergey Tulyakov, and Peter Wonka.
4real-video-v2: Fused view-time attention and feedfor-
ArXiv,
ward reconstruction for 4d scene generation.
abs/2506.18839, 2025. 3

[71] Jianyuan Wang, Nikita Karaev, Christian Rupprecht, and
David Novotny. Vggsfm: Visual geometry grounded deep
In Proceedings of the IEEE/CVF
structure from motion.
Conference on Computer Vision and Pattern Recognition,
pages 21686–21697, 2024. 3

[72] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea
Vedaldi, Christian Rupprecht, and David Novotny. Vggt:
Visual geometry grounded transformer. In Proceedings of
the Computer Vision and Pattern Recognition Conference,
pages 5294–5306, 2025. 2, 3, 6, 7, 8, 1, 5

[73] Qianqian Wang, Yifei Zhang, Aleksander Holynski,
Alexei A. Efros, and Angjoo Kanazawa. Continuous 3d
perception model with persistent state, 2025. 3

[74] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris
Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d
vision made easy, 2024. 2, 3

[75] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu,
Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and
Sebastian Scherer. Tartanair: A dataset to push the limits of
visual slam. 2020. 5, 6, 2

[76] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang,
Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chun-
π3: Scalable permutation-
hua Shen, and Tong He.
equivariant visual geometry learning, 2025. 3

[77] Zhou Wang, Alan Conrad Bovik, Hamid R. Sheikh, and
Eero P. Simoncelli. Image quality assessment: from error
visibility to structural similarity. IEEE Transactions on Im-
age Processing, 13:600–612, 2004. 6

[78] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongx-
uan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-
fidelity and diverse text-to-3d generation with variational
score distillation. Advances in neural information process-
ing systems, 36:8406–8441, 2023. 1, 2

[79] Christopher Wewer, Kevin Raj, Eddy Ilg, Bernt Schiele,
and Jan Eric Lenssen. latentsplat: Autoencoding variational
gaussians for fast generalizable 3d reconstruction, 2024. 2
[80] Haoyu Wu, Diankun Wu, Tianyu He, Junliang Guo, Yang
Ye, Yueqi Duan, and Jiang Bian. Geometry forcing: Mar-
rying video diffusion and 3d representation for consistent
world modeling, 2025. 5, 7, 2

[81] Rundi Wu, Ben Mildenhall, Philipp Henzler, Keunhong
Park, Ruiqi Gao, Daniel Watson, Pratul P Srinivasan, Dor
Verbin, Jonathan T Barron, Ben Poole, et al. Reconfusion:
3d reconstruction with diffusion priors. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 21551–21561, 2024. 1, 2

[82] Tong Wu, Yu-Jie Yuan, Ling-Xiao Zhang, Jie Yang, Yan-
Pei Cao, Ling-Qi Yan, and Lin Gao. Recent advances in
3d gaussian splatting. Computational Visual Media, 10(4):
613–642, 2024. 2

[83] Hongchi Xia, Yang Fu, Sifei Liu, and Xiaolong Wang.
Rgbd objects in the wild: Scaling real-world 3d object
learning from rgb-d videos, 2024. 5, 6, 2

[84] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng,
Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and
Jiaolong Yang. Structured 3d latents for scalable and versa-
tile 3d generation, 2025. 2

[85] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann
Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys.
Depthsplat: Connecting gaussian splatting and depth, 2025.
5, 6, 7, 2

[86] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiao-
gang Xu, Jiashi Feng, and Hengshuang Zhao. Depth any-
thing v2. arXiv:2406.09414, 2024. 3

[87] Yuanbo Yang, Jiahao Shao, Xinyang Li, Yujun Shen, An-
dreas Geiger, and Yiyi Liao. Prometheus: 3d-aware latent
diffusion models for feed-forward text-to-3d scene genera-
tion, 2025. 1, 2

[88] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding,
Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong,
Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-
video diffusion models with an expert transformer. arXiv
preprint arXiv:2408.06072, 2024. 3

[89] Keyang Ye, Tianjia Shao, and Kun Zhou. When gaussian
meets surfel: Ultra-fast high-fidelity radiance field render-
ing. ACM Trans. Graph., 44(4), 2025. 2

[90] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner,
and Angela Dai. Scannet++: A high-fidelity dataset of 3d

indoor scenes. In Proceedings of the International Confer-
ence on Computer Vision (ICCV), 2023. 4, 5

[91] Hong-Xing Yu, Haoyi Duan, Junhwa Hur, Kyle Sargent,
Michael Rubinstein, William T Freeman, Forrester Cole,
Deqing Sun, Noah Snavely, Jiajun Wu, et al. Wonderjour-
ney: Going from anywhere to everywhere. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 6658–6667, 2024. 1, 2

[92] Hong-Xing Yu, Haoyi Duan, Charles Herrmann, William T
Interactive 3d
Freeman, and Jiajun Wu. Wonderworld:
In Proceedings of
scene generation from a single image.
the Computer Vision and Pattern Recognition Conference,
pages 5916–5926, 2025. 1, 2

[93] Xianggang Yu, Mutian Xu, Yidan Zhang, Haolin Liu,
Chongjie Ye, Yushuang Wu, Zizheng Yan, Tianyou Liang,
Guanying Chen, Shuguang Cui, and Xiaoguang Han.
Mvimgnet: A large-scale dataset of multi-view images. In
CVPR, 2023. 5

[94] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jam-
pani, Trevor Darrell, Forrester Cole, Deqing Sun, and
Ming-Hsuan Yang. Monst3r: A simple approach for es-
timating geometry in the presence of motion, 2025. 3
[95] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan
Zhao, Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large
reconstruction model for 3d gaussian splatting, 2024. 3
[96] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu,
Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi
Yu. Clay: A controllable large-scale generative model for
creating high-quality 3d assets, 2024. 2

[97] Qihang Zhang, Shuangfei Zhai, Miguel Angel Bautista
Martin, Kevin Miao, Alexander Toshev, Joshua Susskind,
and Jiatao Gu. World-consistent video diffusion with ex-
plicit 3d modeling. In Proceedings of the Computer Vision
and Pattern Recognition Conference, pages 21685–21695,
2025. 1, 2, 4, 5, 6, 7, 8, 3

[98] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shecht-
man, and Oliver Wang. The unreasonable effectiveness of
deep features as a perceptual metric, 2018. 6

[99] Yuyang Zhao, Chung-Ching Lin, Kevin Lin, Zhiwen Yan,
Linjie Li, Zhengyuan Yang, Jianfeng Wang, Gim Hee Lee,
and Lijuan Wang. Genxd: Generating any 3d and 4d scenes.
In ICLR, 2025. 2

[100] Jensen (Jinghao) Zhou, Hang Gao, Vikram Voleti, Aarya-
man Vasishta, Chun-Han Yao, Mark Boss, Philip Torr,
Christian Rupprecht, and Varun Jampani. Stable virtual
camera: Generative view synthesis with diffusion models.
arXiv preprint arXiv:2503.14489, 2025. 2

[101] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe,
and Noah Snavely. Stereo magnification: Learning view
synthesis using multiplane images, 2018. 5, 6, 3

[102] Dong Zhuo, Wenzhao Zheng, Jiahe Guo, Yuqi Wu, Jie
Zhou, and Jiwen Lu. Streaming 4d visual geometry trans-
former. arXiv preprint arXiv:2507.11539, 2025. 3

Gen3R: 3D Scene Generation Meets Feed-Forward Reconstruction

Supplementary Material

6. Implementation Details

6.1. Processing Input Conditions

We employ multiple conditions into the diffusion process,
including a text prompt y, a condition image sequence
Icond with a flexible number of available frames (missing
images are set to zero), corresponding binary masks M and
optional per-view camera conditions Tcond. The condition
images Icond are encoded into appearance latents Acond by
pretrained RGB VAE EW :

EW : Icond → Acond ∈ Rn×h×w×c,
(10)
while the masks M are downsampled to Ma ∈ Rn×h×w×4
to match the latent resolution. To ensure dimensional con-
sistency with the noised latents, we initialize the geometry
branch’s condition latents Gcond ∈ Rn×h×w×c and corre-
sponding masks Mg ∈ Rn×h×w×4 as zeros.

Finally, the appearance and geometry latents are fused
with their respective latent masks along the channel dimen-
sion, and the two modalities are further concatenated in the
width dimension to construct the unified condition latent:
Zcond = [Acond ⊕ Ma; Gcond ⊕ Mg] ∈ Rn×h×2w×c′

,
(11)
where (·⊕·) means concatenation along channel dimension,
and c′ = c + 4. The input to the diffusion model is then
constructed by concatenating the noised latents Zt with the
condition latents Zcond along the channel dimension:

Zin = Zt ⊕ Zcond,
Gθ : Zin → ˆZt−1.

(12)
(13)

6.2. Model Architectures

Geometry Adapter. We obtain our adapter (Eadp, Dadp)
by modifying Wan’s causal VAE [69]. The adapter projects
VGGT [72] geometry tokens V ∈ RN ×L×hv×wv×C into the
video diffusion model’s latent space and maps them back:

Eadp : V → G ∈ Rn×h×w×c,
Dadp : G → V ∈ RN ×L×hv×wv×C,

(14)

(15)

where L = 5, since we broadcast VGGT’s camera tokens
of each frame to the spatial resolution hv × wv (hv = wv =
40), and concatenate it with the other 4 tokens along the L
dimension.

To match the VAE input format, we first reshape V into
V ′ ∈ RN ×hv×wv×(L×C). Accordingly, we set the adapter
input dimension to L × C = 10240 and use hidden di-
mensions [512, 256, 128, 128]. We then re-sample the in-
put tokens V ′ to a spatial resolution of h × w = 70 × 70

s
t
u
p
n
I

]
6
6
[

r
e
h
t
e
A

]
7
9
[

D
V
W

s
r
u
O

Figure 8. Qualitative Comparison of Geometry Generation in
the 2-view based setting.

Input

VGGT

Ours

Figure 9. Qualitative Comparison of Geometry Reconstruc-
tion.

using nearest-exact interpolation, and apply a 2D convolu-
tion to project the channels to 1024. The resulting features
are processed by causal convolution layers, where we keep
the spatial resolution unchanged, yielding geometry latents
G ∈ Rn×h×w×c. Similarly, the decoder Dadp mirrors the
encoder architecture in reverse, reconstructing the geometry
tokens V from the latents G.

Diffusion Transformer. We adapt the DiT architecture
from VideoX-Fun’s Wan2.1 [69] to accommodate our joint
appearance-geometry latents. Specifically, we set the input
channel dimension to c + c′ = 36. To support width-wise
concatenation of appearance and geometry latents, we mod-
ify the positional embeddings so that corresponding pix-
els in the left and right halves of the latents share identical
RoPE embeddings.

Cond.

Method

1-view

LVSM [23]
Gen3C [51]
GF [80]
Aether [66]
WVD [97]
Ours

2-view

DepthSplat [85]
LVSM [23]
Gen3C [51]
GF [80]
Aether [66]
WVD [97]
Ours

PSNR ↑

SSIM ↑ LPIPS ↓

I2V Subj. ↑

I2V BG ↑

I.Q. ↑

PSNR ↑

SSIM ↑ LPIPS ↓

I2V Subj. ↑

I2V BG ↑

I.Q. ↑

PSNR ↑

SSIM ↑ LPIPS ↓

I2V Subj. ↑

I2V BG ↑

I.Q. ↑

Co3Dv2

WildRGB-D

TartanAir

14.08
15.82
10.25
12.78
13.35
16.09

10.45
17.87
17.16
12.67
14.28
14.66
18.01

0.5623
0.5666
0.3150
0.5106
0.4733
0.5754

0.3262
0.5986
0.5927
0.3855
0.5405
0.5101
0.6085

0.5698
0.5095
0.6761
0.6052
0.5765
0.4997

0.6167
0.4534
0.4776
0.5998
0.5498
0.5334
0.4371

0.9482
0.9134
0.7933
0.9229
0.9339
0.9535

0.8314
0.9467
0.9149
0.7645
0.9322
0.9246
0.9547

0.9581
0.9355
0.8193
0.9395
0.9484
0.9588

0.8585
0.9519
0.9361
0.7925
0.9426
0.9409
0.9597

0.3579
0.4335
0.5320
0.4411
0.5355
0.5383

0.2992
0.4064
0.4263
0.4969
0.4647
0.5306
0.5405

13.9483
14.60
11.8944
11.87
12.95
14.73

16.22
19.13
17.81
13.51
13.79
16.27
18.88

0.5239
0.5463
0.4147
0.4289
0.4522
0.5501

0.5382
0.6789
0.6307
0.3991
0.4884
0.5421
0.6448

0.5195
0.4513
0.5940
0.5973
0.5362
0.4398

0.4518
0.3134
0.3882
0.4609
0.5161
0.4098
0.3256

0.9692
0.9622
0.9215
0.9595
0.9669
0.9715

0.9012
0.9747
0.9636
0.8785
0.9491
0.9631
0.9746

0.9713
0.9646
0.9214
0.9614
0.9671
0.9716

0.9067
0.9730
0.9651
0.8852
0.9512
0.9646
0.9755

0.4004
0.4629
0.5310
0.4786
0.5513
0.5609

0.3779
0.4555
0.4634
0.5374
0.4685
0.5627
0.5685

14.44
13.95
10.21
12.88
12.77
15.04

13.87
17.79
15.24
12.06
14.53
14.22
17.34

0.5044
0.4731
0.3249
0.4585
0.4513
0.5069

0.4585
0.5685
0.5055
0.3670
0.4989
0.4605
0.5581

0.5210
0.5385
0.6249
0.5645
0.5652
0.5073

0.5195
0.4265
0.5318
0.5666
0.5153
0.5266
0.4416

0.9325
0.9142
0.7447
0.9295
0.9271
0.9350

0.8073
0.9415
0.9119
0.7447
0.9294
0.9116
0.9385

0.9540
0.9403
0.7864
0.9480
0.9473
0.9546

0.8474
0.9569
0.9376
0.7946
0.9496
0.9371
0.9559

0.3542
0.3713
0.4379
0.4303
0.4571
0.4620

0.3301
0.3628
0.3668
0.4589
0.4267
0.4680
0.4748

Table 6. Quantitative Comparison of Appearance Generation. We compare both 1-view and 2-view settings with camera conditions.

.

d
n
o
C

Method

w Aether [66]
WVD [97]
Ours

e
i
v
-
1

w Aether [66]
WVD [97]
Ours

e
i
v
-
2

RealEstate10K

DL3DV-10K

I2V Subj. ↑

I2V BG ↑

Aes.Q. ↑

I.Q. ↑ M.S. ↑

I2V Subj. ↑

I2V BG ↑

Aes.Q. ↑

I.Q. ↑ M.S. ↑

0.9743
0.9815
0.9879

0.9852
0.9929
0.9949

0.9770
0.9843
0.9890

0.9843
0.9923
0.9947

0.5118
0.5125
0.5291

0.5278
0.5336
0.5369

0.5060
0.5653
0.5761

0.5187
0.5973
0.6009

0.9885
0.9895
0.9929

0.9923
0.9938
0.9947

0.9377
0.9274
0.9461

0.9485
0.9403
0.9549

0.9501
0.9412
0.9561

0.9521
0.9518
0.9576

0.4704
0.4555
0.4727

0.4846
0.4760
0.4881

0.4872
0.4916
0.5187

0.5026
0.5338
0.5357

0.9600
0.9542
0.9701

0.9685
0.9685
0.9719

Table 7. Quantitative Comparison of Appearance Generation without camera conditions.

Cond.

Method

1-view

LVSM [23]
SEVA [100]
Aether [66]
Ours

2-view

LVSM [23]
SEVA [100]
Aether [66]
Ours

LLFF

Mip-NeRF 360

ScanNet++

PSNR ↑ LPIPS ↓

PSNR ↑ LPIPS ↓

PSNR ↑ LPIPS ↓

12.39
11.43
11.01
13.19

18.39
16.86
13.66
17.66

0.5742
0.6562
0.6133
0.5078

0.3266
0.3769
0.4589
0.3496

12.71
12.77
11.06
13.17

15.56
14.86
11.43
15.32

0.6328
0.5898
0.6640
0.5703

0.5039
0.5080
0.6289
0.4941

15.25
13.97
12.47
15.42

21.31
18.07
17.14
20.11

0.4531
0.4688
0.5351
0.4420

0.2754
0.3438
0.4018
0.2964

Table 8. Quantitative Comparison of Appearance Generation
on Out-of-Distribution Datasets. We compare both 1-view and
2-view settings with camera conditions.

Method

Aether
VGGT
Ours

RealEstate10K WildRGB-D

AUC@30 ↑

AUC@30 ↑

0.7291
0.8387
0.8265

0.7303
0.8406
0.8391

Table 9. Quantitative Comparison of Camera Pose Estimation
in feed-forward 3D reconstruction.

7. Additional Comparison Results

7.1. 3D Generation

Comparison on 3D Generation with Camera Condi-
tions. We provide the full appearance evaluation results on
Co3Dv2 [50], WildRGB-D [83] and TartanAir [75] datasets
in Tab. 6. Gen3R consistently surpassing existing methods

Method

Aether
VGGT
Ours

ScanNet++

Accuracy ↓

Completeness ↓

CD ↓

0.3187
0.1396
0.1455

0.3022
0.1162
0.0963

0.3105
0.1279
0.1209

Table 10. Quantitative Comparison of Zero-shot Geometry Re-
construction in ScanNet++ dataset.

Method

RealEstate10K

DL3DV-10K

PSNR ↑

SSIM ↑

LPIPS ↓

PSNR ↑

SSIM ↑

LPIPS ↓

VGGT* [72]
RGB VAE [69]

23.3927
37.5770

0.8346
0.9819

0.2341
0.0288

22.6958
32.7673

0.7557
0.9057

0.2910
0.1031

Table 11. Quantitative Comparison for RGB Reconstruction.
We train an RGB head for VGGT to reconstruct images from ge-
ometry tokens. * indicates our implementation.

across all metrics and datasets in the 1-view setting, and
achieves leading performance in the 2-view setting. Addi-
tional qualitative comparisons of 3D generation are shown
in Fig. 10 and Fig. 8. As observed, LVSM [23], Aether [66]
and WVD [97] fail to synthesize images from novel view-
point in 1-view setting, primarily due to poor camera con-
trollability. While Gen3C [51] can generate plausible con-
tents, it exhibits notable shifts caused by inaccurate depth
estimation. In contrast, our methed produces high-fidelity
results that adhere closely to the camera conditions and
maintain better 3D structure, as shown in Fig. 8.

Comparison on 3D Generation without Camera Condi-
tions. We further demonstrate our capability to generate
3D scenes from images without camera conditions. To as-

Input

LVSM [23]

Gen3C [51]

Aether [66]

WVD [97]

Ours

Ground Truth

Figure 10. Qualitative Comparison of Novel View Synthesis in 1-view setting with camera conditions.

Input

Generated Frames

Point Cloud

Figure 11. More Qualitative Results in 1-view setting with camera conditions.

sess this, we report the VBench Score [20, 21], focusing on
I2V Subject (I2V Subj.), I2V Background (I2V BG), Aes-
thetic Quality (Aes.Q.), Imaging Quality (I.Q.) and Motion
Smoothness (M.S.) on RealEstate10K [101] and DL3DV-

10K [34] datasets. As shown in Tab. 7, our method clearly
outperforms Aether [66] and WVD [97], illustrating its su-
perior ability in generating high-quality 3D scenes.

Input

Generated Frames

Point Cloud

Figure 12. More Qualitative Results in 2-view setting with camera conditions.

Input

Generated Frames

Point Cloud

Figure 13. More Qualitative Results in 1-view and 2-view settings without camera conditions.

Comparison on 3D Generation on Out-of-Distribution
Datasets. We report appearance generation results in 1-
view and 2-view settings on LLFF [41], Mip-NeRF 360 [2]
and ScanNet++ [90] test sets, which are excluded from the
training datasets. As shown in Tab. 8, our method handles
these unseen scenes well, and yields consistent conclusions
with the main paper.

7.2. Feed-forward 3D Reconstruction

Comparison on Camera Pose Estimation. We evaluate
our method on RealEstate10K and WildRGB-D datasets for
camera pose estimation, as reported in Tab. 9. Our approach
achieves competitive results compared to VGGT, while no-
tably surpassing Aether, showing the versatility and robust-
ness of our model.

Generated Frames & Depths

Point Cloud

Figure 14. More Qualitative Results of feed-forward reconstruction.

tions (see Fig. 11 and Fig. 12); 2) Feed-Forward 3D Re-
construction (see Fig. 14); and 3) 3D Generation without
Camera Conditions (see Fig. 13). We visualize the gener-
ated frames, depth maps of the sequences, and the global
point clouds of the scenes.

Our method synthesizes globally consistent and photore-
alistic 3D scenes under diverse input conditions and effec-
tively handles a wide range of scenarios, including indoor
scenes, outdoor environments, and object-centric cases.
Thanks to our design, the model exhibits strong camera con-
trollability under conditioned settings, while also enabling
free scene navigation in the absence of camera inputs. Com-
bined with support for multiple output modalities, Gen3R
provides fine-grained and coherent 3D scene generation
across both constrained and unconstrained regimes.

Comparison on Geometry Reconstruction. We pro-
vide additional qualitative results of feed-forward 3D re-
construction compared with VGGT [72] in Fig. 9. It can
be observed that VGGT produces noticeable floaters in the
reconstructed point clouds, while our method generates sig-
nificantly cleaner geometry.

In addition,

to evaluate zero-shot generalization, we
compare Gen3R with VGGT on ScanNet++ [90] test split.
As shown in Tab. 10, VGGT slightly outperforms Gen3R in
accuracy; however, our method achieves better complete-
ness and better chamfer distance, indicating that our method
generalizes reasonably to unseen scenes.

7.3. Ablation Study

RGB Head for VGGT. To validate the effectiveness of our
joint latents design, we train an RGB head for VGGT to
enable direct RGB reconstruction from its geometry tokens
V. We then compare its RGB reconstruction quality with
that of Wan’s RGB VAE [69]. The results are presented
in Tab. 11. RGB VAE significantly outperforms VGGT*,
as VGGT is designed primarily for geometry modeling and
lacks sufficient capacity for RGB feature extraction and
high-fidelity appearance reconstruction. This observation
also motivates our choice to decode appearance and geom-
etry separately. By combining the strengths of both pre-
trained models, we achieve photorealistic video generation
together with high-quality 3D structure.

8. More Results of Gen3R

We present additional qualitative results for both 3D gen-
eration and feed-forward 3D reconstruction in this sec-
tion, including: 1) 3D Generation with Camera Condi-

