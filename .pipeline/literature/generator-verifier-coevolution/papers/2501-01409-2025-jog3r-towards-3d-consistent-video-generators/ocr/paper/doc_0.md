JOG3R: Towards 3D-Consistent Video Generators

Chun-Hao Huang1 Niloy Mitra1,2 Hyeonho Jeong1,3* Jae Shin Yoon1 Duygu Ceylan1

1Adobe Research

3KAIST
https://paulchhuang.github.io/jog3rwebsite

2University College London

5
2
0
2

r
a

M
6
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
9
0
4
1
0
.
1
0
5
2
:
v
i
X
r
a

Figure 1. We present JOG3R, a unified framework that fine-tunes a video generation model jointly with a 3D point map estimation task.
JOG3R improves the 3D-consistency of the generated videos compared to the pre-trained video diffusion transformer (DiT) as shown by
the warped feature maps (left) and scores (right) using MEt3R [2], lower scores indicating higher 3D-consistency across frames.

Abstract

1. Introduction

Emergent capabilities of image generators have led to
many impactful zero- or few-shot applications. Inspired by
this success, we investigate whether video generators sim-
ilarly exhibit 3D-awareness. Using structure-from-motion
as a 3D-aware task, we test if intermediate features of a
video generator – OpenSora in our case – can support cam-
era pose estimation. Surprisingly, at first, we only find a
weak correlation between the two tasks. Deeper investi-
gation reveals that although the video generator produces
plausible video frames, the frames themselves are not truly
3D-consistent. Instead, we propose to jointly train for the
two tasks, using photometric generation and 3D aware er-
rors. Specifically, we find that SoTA video generation and
camera pose estimation (i.e., DUSt3R [79]) networks share
common structures, and propose an architecture that unifies
the two. The proposed unified model, named JOG3R, pro-
duces camera pose estimates with competitive quality while
producing 3D-consistent videos. In summary, we propose
the first unified video generator that is 3D-consistent, gen-
erates realistic video frames, and can potentially be repur-
posed for other 3D-aware tasks.

foundational

image genera-
Following the success of
tors [63], video generators have quickly become a reality.
After the initial demonstration by Sora, many similar mod-
els have rapidly emerged, both in commercial and open-
source domains [7, 9, 24, 52, 97]. Trained on large-scale
datasets (e.g., WebVid-10M [4], Panda-70M [15]), these
models produce impressive diversity with both compelling
image quality and temporal consistency. Given the emerg-
ing behaviors observed in the case of image generators [35]
leading to several successful zero- or few-shot approaches
(e.g., feature detection[20, 70], segmentation [54], gener-
ative editing [58]), we investigate if video generators can
be similarly repurposed for 3D-aware tasks (e.g., structure
from motion, camera pose, correspondence tracking).

As a 3D-aware task, we pick the classical structure-from-
motion (SfM) problem as it requires reasoning about both
scene geometry and the relative viewpoint changes across
frames. We expect this 3D task to be compatible with video
generation as both tasks need to arrive at feature represen-
tations that capture the physical world [34]. We further
observe that the ViT backbone of the DUSt3R architec-
ture [79], which is the leading feedforward backbone for
establishing correspondence between video frames, actually
shares many architectural designs with the Diffusion Trans-

   1            3            5           7            9         11        13        15   0.160.120.080.040JOG3RframesMET3R01pre-trained OSframe 1frame 7frame 16pre-trained OSJOG3R 
 
 
 
 
 
formers (DiT) in state-of-the-art video generators. This al-
lows us to stitch the OpenSora backbone with a DUSt3R-
like point map – and hence camera pose – estimation head
into a unified architecture.

We train the 3D point map reconstruction head using
a frozen video generator (we used OpenSora [97] in our
tests). Somewhat surprisingly, we find although the cho-
sen tasks are seemingly compatible, the estimated 3D point
maps and relative camera poses were, at best, mediocre
(see Section 4). On deeper investigation, we found that al-
though the (pre-trained) video generators produced visually
compelling and temporally-smooth frames (as indicated by
FVD scores), they were not 3D-consistent. This is also ex-
posed by computing warped feature scores using MEt3R [2]
(see Figure 1); this explains why taking existing internal
video generator features leaves a gap between the tasks of
video generation and 3D estimation.

The above insight helps us design a video generator that
is both visually compelling (i.e., good FVD score) and 3D-
consistent (i.e., good MEt3R score). In particular, we fine-
tune the video generator using the proposed unified archi-
tecture supervised by both video generation and 3D geo-
metric (correspondence) losses. Such a formulation makes
the two tasks more ‘equivalent’, and hence improves both.
Thus, the secondary task of 3D reconstruction helps im-
prove the 3D coherence of the video generator (see Figure 1
and the supplementary webpage for more results).

To summarize, we present a novel architecture unify-
ing video generation with 3D point map estimation, which
we refer to as JOint Generation and 3D Reconstruction,
in short JOG3R. Being a unified model, JOG3R gener-
ates videos (T2V), estimates 3D point maps, hence camera
poses, given a video (V2C), or do both in one go (T2V+C).
In particular, we test whether with fine-tuning one can pro-
duce video generator features that can also be reused for
improved 3D reconstruction. Finally, we analyze the effect
of such fine-tuning on generation quality. Our experiments
show that while video generation features exhibit some 3D
awareness natively (i.e., obtained from pre-trained genera-
tor directly), adapting them with additional supervision on
the 3D reconstruction task boosts their 3D-consistency. As
a side benefit, the additional supervision produces compet-
itive camera pose estimations compared to state-of-the-art
solutions, and unifies the video generation and camera pose
estimation tasks. Code will be released on acceptance.

2. Related Work

2.1. Diffusion-Based Video Generation

Building on the success of diffusion models [30, 68] in
image synthesis [18, 63], the research community has ex-
tended diffusion-based methods to video generation. Early
works [31, 32] adapted image diffusion architectures by in-

corporating a temporal dimension, enabling the model to
be trained on both image and video data. Typically, UNet-
based architectures incorporate temporal attention blocks
after spatial attention blocks, and 2D convolution layers
are expanded to 3D convolution layers by altering ker-
nels [32, 85]. Latent video diffusion models [7, 8, 27, 77]
have been introduced to avoid excessive computing de-
mands,
implementing the diffusion process in a lower-
dimensional latent space. Seeking to generate spatially
and temporally high-resolution videos, another line of re-
search adopts cascaded pipelines [5, 31, 67, 81, 92], in-
corporating low-resolution keyframe generation, frame in-
terpolation, and super-resolution modules. To maximize
computational scalability, recent waves in video generation
[9, 14, 50, 52, 97] diverge from UNet-based architecture
and employ the Diffusion Transformer (DiT) [59] backbone
that processes space-time patches of video and image latent
codes. Following this direction, we build our method on
OpenSora [97], a publicly available DiT-based latent video
diffusion model. These models are trained with only pho-
tometric error and, as demonstrated in our evaluation, not
3D-consistent (see Section 4 and [2]). This, in turn, means
that their internal features are not suitable for mixing [40],
with no or little fine-tuning, for 3D-aware tasks [3, 36].

2.2. 3D Reconstruction

The fundamental principles of multiview geometry [83] in-
cluding feature extraction [10, 48], matching [1, 25, 47, 84],
and triangulation with epipolar constraints are well known
to produce accurate (yet sparse) 3D point clouds with pre-
cise camera pose estimation from multiview images of
real scenes [65]. The efficiency of 3D reconstruction has
been improved with linear-time incremental structure-from-
motion [84] and coarse-to-fine hybrid approaches [16, 17].
To improve robustness to outliers, researchers proposed
global camera rotation averaging [17], camera optimization
techniques based on features of points vanishing with ori-
ented planes [33], or from a learned neural network [45]
to prevent rotation and scale drift issues. Global camera
pose registration and approximation with geometric linear-
ity [12, 38] or joint 3D point position estimation [57] are
designed to further push the scalability and efficiency of the
3D reconstruction as well as the robustness particularly to
the image sequence with small baselines.

Given estimated camera poses and sparse 3D point
clouds, multiview stereo can then produce a dense 3D sur-
face using hand-created visual features [66] or neural fea-
tures with a cost volume [51, 51, 72, 87, 95] to predict
globally coherent depth estimates. Existing neural render-
ing methods reconstruct such a dense surface by modeling
the implicit or explicit cost volume and differentiable ren-
dering of the scene for photometric supervision from multi-
view images [23, 44, 53, 55, 60, 69, 78, 80, 89] or monoc-

Figure 2. We propose a unified framework to investigate if the intermediate features from a video generation model can be repurposed for
3D point map estimation by routing them to the SoTA decoder of DUSt3R. We investigate the effect of freezing vs training certain modules
of the generator using different combination of generation and reconstruction losses.

ular depth estimation [64]. Some pose-free methods fur-
ther erase the requirement of camera calibration: test time
optimization produces globally consistent depth map un-
der unknown scale and poses using frozen depth predic-
tion model [86]; the unsupervised signals from dense cor-
respondences such as optical flow are integrated to learn
from unlabeled data [71, 88, 98]. Recent works proposed
a direct regression framework for dense surface reconstruc-
tion from pairwise images by learning to predict globally
coherent depths and camera parameters [73] or to directly
predict per-pixel 3D point clouds from two views [42, 79]
using a vision transformer with dense tokenization [62].
However, these methods are designed for real videos, and
fail to handle generated videos, which so far have not
been 3D-consistent. This has also been observed in recent
warped feature analysis among video frames as reported by
MEt3R [2].

2.3. Diffusion Model as Features for Reconstruction

A generative diffusion model is often trained on millions
of paired image and text prompts and in the process de-
velops a semantically meaningful visual prior. Naturally,
researchers are interested in using this strong prior for
many downstream 3D vision tasks.
Injecting 3D aware-
ness into the diffusion prior greatly improves the accu-
racy and generalizability of the monocular depth estima-
tion and correspondence search tasks [21, 90]. The latent
features from the frozen pretrained diffusion model are of-
ten used as a backbone, and a task-specific decoder with
cross attention is newly trained for semantic correspon-
dences [20, 28, 28, 37, 70, 93, 94], semantic segmentation
and monocular depth estimation [96], material and shadow
prediction [91], general object 3D pose estimation [11, 56].
However, such image diffusion features do not inherently
consider the temporal relation between the frames, leading
to temporally unstable 3D prediction results from videos. In

contrast, we investigate video diffusion features as a back-
bone for the multitasking prediction of video generation
and 3D camera poses estimation (see [3, 35]). Recently,
video generators have been probed for multiview consis-
tency — MEt3R evaluates multiview consistency by per-
forming dense 3D reconstructions from image pairs using
DUSt3R [79], and uses the estimated tranformation to warp
image contents from one view into the other. The warped
features are then compared, using a view-independent sim-
ilarity score, with lower scores indicating higher multiview
consistency – for static scenes, this measures quality of 3D-
consistency (see also, Figure 1).

3. Method

3.1. Preliminaries

Video diffusion model. We consider OpenSora [97] as our
base video generation model, which is a DiT-based video
diffusion model inspired by the notable success of Sora [9].
It performs the diffusion process in a lower-dimensional la-
tent space defined by a pre-trained VAE encoder E. Each
frame x of the input video is first projected into this la-
tent space, z0 = E(x). Given a diffusion time step t, the
forward process incrementally adds Gaussian noise to the
latent code z0 via a Markov chain to obtain noisy latent
zt. The denoising model ϵθ takes the noisy latents of all
frames, the time step t, and the text prompt y as input to pre-
dict the added noise: ϵθ({zf
t }F
f =1, t, y), where F is the total
number of frames and θ denotes the parameters of the DiT
network [59]. The network consists of 28 spatial-temporal
diffusion transformer (STDiT) blocks {b1, . . . , b28}, similar
to [50]. The iterative process of noise prediction and noise
removal is referred to as the reverse process.
3D reconstruction model. We consider the state-of-
the-art multi-view stereo reconstruction (MVS) framework
DUSt3R [79] as our 3D module. Given an image pair,

......єpredicted paired point map (cid:31)єΘ(·)(cid:30)ground truth paired point map (cid:31)L(cid:29)(cid:28)(cid:27)L(cid:26)(cid:28)(cid:25)(cid:31)(cid:30)(cid:29)(cid:28)(cid:27)(cid:26)(cid:30)(cid:25)(cid:24)(cid:29)(cid:23)(cid:28)(cid:22)(cid:27)(cid:21)(cid:28)(cid:20)(cid:26)(cid:19)(cid:30)(cid:18)(cid:17)(cid:19)(cid:30)(cid:26)(cid:25)(cid:16)(cid:25)(cid:15)(cid:30)(cid:29)(cid:19)_(cid:21)(cid:23)(cid:29)(cid:28)(cid:14)(cid:29)(cid:13)(cid:25)(cid:29)(cid:28)(cid:28)(cid:30)(cid:12)(cid:28)(cid:14)(cid:17)(cid:12)(cid:28)(cid:30)(cid:24)(cid:23)(cid:17)(cid:26)(cid:29)(cid:13)(cid:25)(cid:29)(cid:28)(cid:28)(cid:30)(cid:12)(cid:28)(cid:14)(cid:17)(cid:12)(cid:18)(cid:26)(cid:17)(cid:21)(cid:21)(cid:25)(cid:29)(cid:28)(cid:28)(cid:30)(cid:12)(cid:28)(cid:14)(cid:17)(cid:12)(cid:11)(cid:10)(cid:11)(cid:9)(cid:11)(cid:12)(cid:11)(cid:9)(cid:8)DUSt3R encodes each image independently with a ViT en-
coder [19, 82]. Two decoders process both features to en-
able cross-view information sharing, followed by separate
heads that estimate point maps X ∈ RH×W ×3, represented
in the coordinates of the first view as X 1,1 and X 2,1, re-
spectively. The relative camera pose is then estimated by
aligning X 1,1 and X 1,2 using Procrustes alignment [49]
with PnP-RANSAC [22, 41]. A global optimization scheme
is employed to register more than a pair of views from the
same scene as post-processing.

3.2. Unified Video Generation & 3D Reconstruction

Feature routing via model stitching. We argue that both
a foundational video generation model as well as a 3D re-
construction model are trying to arrive at feature representa-
tions that capture the physical reality and hence are likely to
converge to feature spaces [35] that exhibit equivalence. To
verify this hypothesis and facilitate our analysis, inspired
by the model stitching technique [39], we propose a uni-
fied framework that routes intermediate features from the
video generator to the 3D reconstruction task. We observe
that ViT and DiT share many architectural designs in com-
mon since they both belong to the broad transformer fam-
ily. Hence, our key insight is to replace the image-based
ViT encoder in DUSt3R with the video DiT backbone in
OpenSora. In other words, we stitch the transformer blocks
of the DiT network ϵθ with the DUSt3R decoders and heads
(see Figure 2 for illustration). In this setup, the intermedi-
ate features from the video generation model are effectively
routed to DUSt3R decoders to perform the reconstruction
task. Specifically, we extract the output of the intermedi-
ate STDiT block bn at a particular time step t during the
reverse process. Following Tang et al. [70], we consider
small t where the feature focuses more on low-level details,
making it useful as a geometric feature descriptor to build
correspondence across frames.
3D-aware training for video generation. Given a stitched
and unified model, as described above, a naive baseline is to
use pre-trained features from the video model and only train
the stitched DUSt3R decoders to perform a 3D reconstruc-
tion task. As discussed in Section 4 and shown in Table 2,
while this baseline (row 1) provides reasonable 3D recon-
struction quality, i.e., more effective than DUSt3R trained
on the same dataset (row 4) but not as effective as the pre-
trained DUSt3R (row 6). This shows that the feature repre-
sentations of a video generator and a 3D reconstructor are
compatible but the features of a pre-trained video genera-
tor are not fully 3D-consistent. This hinders performance.
Hence, we further fine-tune both the video generator and
the reconstruction heads jointly to perform generation and
reconstruction tasks.

For training, we consider two losses: generation loss
Lgen and reconstruction loss Lrec. The generation loss Lgen

is the common objective in training diffusion models that
aims to match the added noise ϵ and helps to retain the gen-
erative power of the video model. The reconstruction loss
Lrec is aimed to improve the 3D-awareness of the interme-
diate features and follows the formulation in DUSt3R. It is
defined as the sum of confidence-weighted Euclidean dis-
tance L2(f, i) between the regressed point maps X and the
ground truth point maps ¯X over all valid pixels i and all
frames f . Formally,

Lgen

Lrec

:=

:=

(cid:16)

(cid:13)
(cid:13)
(cid:13)ϵ − ϵθ
(cid:88)

{zf
(cid:88)

t }F

f =1, t, y

(cid:17)(cid:13)
2
(cid:13)
(cid:13)
2

(1)

C f→1
i

L2(f, i) − α log C f,1

(2)

i

f ∈{2,...F }

i

with L2(f, i) =

(cid:13)
(cid:13)
(cid:13)
(cid:13)

1
s

X f→1
i

−

1
¯s

¯X f→1
i

(cid:13)
(cid:13)
(cid:13)
(cid:13)2

i

where the scaling factors s and ¯s handles the scale ambigu-
ity between prediction and ground-truth by bringing them to
a normalized scale, C f→1
is the confidence score for pixel
i, which encourages network to extrapolate in harder areas,
and α is a hyper-parameter controlling the regularization
term [75] (see [79] for more details). In our experiments,
we study the effect of different combinations of these two
losses. Specifically, activating Lrec alone analyzes the na-
tive 3D awareness of the video generation features while
using them in conjunction, Ltotal = Lgen + λLrec (we em-
pirically set λ = 1) investigates if the features can further
be adapted for both video generation and camera pose es-
timation tasks. JOG3R fine-tunes the video model starting
with its pre-trained weights while the DUSt3R decoder and
heads stitched to this generation model are always trained
from scratch.
Our modification of DUSt3R. The features extracted from
the video generator encode a sequence of F frames and are
provided to DUSt3R in a pair-wise manner. During training,
the first frame can ideally be paired with all other frames f .
In practice, due to memory constraints, we sample 4 pairs
from the set {(1 → f )}F

f =2 to predict the 3D point maps.

Once the unified model is trained, if it is desired to obtain
camera parameters for a given sequence, at inference time,
we first predict the point maps between all pairs (1 → f )
and perform the global camera registration in DUSt3R to re-
fine the camera pose, depth and focal length for each frame,
akin to bundle adjustment. Since our input is not a set of
sparse views but a temporal sequence, we append two reg-
ularization terms, Ltranl and Lfl, to the original global reg-
istration objective, encouraging smooth camera translation
and consistent focal length between neighboring frames, re-
spectively. The two terms are used only at inference, and we
refer to the supplemental for the detailed formulation.
Implementation details. We adopt OpenSora 1.0 as
our video generator, which uses 2D VAE (from Stability-

Figure 3. We base our analysis on three main tasks: text-to-video (T2V), video to camera estimation (V2C), and joint video generation and
camera estimation (T2V+C) at inference time.

AI) [63], T5 text encoder [61], and an STDiT (ST stands
for spatial-temporal) architecture similar to variant 3 in [50]
as the denoising network. Among the 28 STDiT blocks, we
empirically set the first 4 frozen and update only the weights
of the temporal attention layers for the remaining 24 blocks.
We extract the output of the 26th block b26 as feature maps
for DUSt3R decoders. The final two blocks behave as a
“generation” branch whose weights are only updated by the
gradient of generation loss Lgen. We refer to the supplemen-
tary material where we describe how the features obtained
from different STDiT blocks perform.

We adopt the linear prediction head of DUSt3R for final
point map estimation. DUSt3R originally uses a decoder
with 12 transformer blocks that is duplicated for each of the
pair of frames. However, information sharing is enabled be-
tween the two decoders. In our experiments, we find that a
decoder structure with six transformer blocks provides sim-
ilar performance and report our results accordingly. Note
that the decoder blocks used in our training experiments
are trained from scratch. Furthermore, since the features
we get from the generator encode all the frames in a video
sequence, we also experiment with replacing the duplicate
decoder architecture with a single decoder consisting of 6
transformer blocks that perform full 3D attention across all
the frames. We empirically find that this performs on par
with duplicate decoders (see Table 2), and hence we use the
latter to provide a more fair comparison to DUSt3R.

When training, we sample the time step t ∈ [0, 10] (cor-
responding to 10% of noise level) and consider the empty
prompt for computing the reconstruction loss Lrec, while for
the generation loss Lgen we sample the full range of time
steps and use the captions of the videos as text prompts.

4. Experiments

We base our analysis on evaluations conducted on three
(i) Text-to-video (T2V): We
specific tasks (see Figure 3).
evaluate the effect of using additional supervision from

the 3D reconstruction task on generation quality for the
task of text to video generation where we sample Gaus-
sian noise and iteratively denoise it with the text guidance.
(ii) Video-to-camera (V2C): We add noise to a given in-
put video based on a sampled time step t sampled in the
range [0, 5], denoise it for one time step, route the feature
maps to DUSt3R decoders and heads, followed by registra-
tion of point maps X to obtain camera poses. (iii) Text-to-
Video+Camera (T2V+C): Once trained with a combination
of generation and reconstruction losses, JOG3R performs
text-to-video generation while simultaneously routing the
intermediate features to the reconstruction module at the
desired time step (sampled in the range [0, 5]), without the
overhead of adding noise and passing it through the net-
work again. As a result, cameras are generated alongside
the video in one go, unifying the two tasks.

We follow standard metrics to assess the generated video
quality (FID and FVD for image/video quality; MEt3R for
3D-consistency) for T2V setup; while validating the accu-
racy of 3D reconstruction, hence camera pose estimation
(for static videos), on real videos (V2C). We further pro-
vide results for jointly generating videos along with camera
pose estimation (T2V+C). Since there is no ground truth in
this case, we report self-consistency.

4.1. Setup

Data. We choose RealEstate10K [99] as our main dataset,
which has around 65K video clips of static scenes paired
with camera parameter annotations. We use the captions
of RealEstate10K provided in [26] and also follow their
train/test split. As pre-processing, we pre-compute the VAE
latents of the video frames and the T5 text embeddings
of the captions. We sample F =16 frames from the orig-
inal sequences with a frame stride randomly chosen from
{1, 2, 4, 8} and also randomly reverse the frame order with
a probability of 0.5.

To obtain point map annotations ¯X, we estimate metric
depth with ZoeDepth [6], unproject them to 3D and trans-

caption 𝑦+time step 𝑡DiTnetworkcaption 𝑦x 𝑇time step 𝑡noisegenerated videoDUSt3RDecoder & Headpoint maps 𝑋ϵDiTnetworkinput videocamera posesregistration𝑦x 𝑇𝑡noiseDUSt3R Decoder & Headcamera posesregistrationpoint maps 𝑋T2VT2V+CV2CDiTMethod

Lgen

Lrec

FID ↓

FVD ↓ MEt3R ↓

pre-trained OS

n.a.

n.a.

115.36

1872.41

frozen OS + rec
fine-tuned OS
JOG3R (ours)

✗
✓
✓

✓
✗
✓

94.52
88.02
79.94

1797.07
1440.92
1742.73

0.0819

0.0772
0.0913
0.0736

Table 1. Generation quality comparison. We compute FID and
FVD for photometric quality and MEt3R [2] for 3D-consistency
measure, both on the RealEstate10K-test data. While being com-
parable in terms of video quality against OpenSora (OS) baseline
and variants, JOG3R produces most 3D-consistent results.

form to the coordinate of the first frame using the camera
parameters in RealEstate10K. All camera extrinsic param-
eters are expressed with respect to the first frame.

In addition,

to check generalization, we consider
DL3DV10K [46], which also provides camera annotations,
as an additional test set. We choose a random set of
70 videos for testing and caption the first frame of each
video using [43]. We prepare point map annotations using
ZoeDepth [6].

Baselines. Since there is no existing method that can per-
form both video generation and 3D reconstruction jointly,
we can only compare to task-specific methods. For video
generation, we consider the pre-trained OpenSora (pre-
trained OS) as well as a version fine-tuned on our dataset
using generation loss only (fine-tuned OS). We also de-
fine a baseline where we route the features from the pre-
trained and frozen OpenSora model and only train the
reconstruction heads using the reconstruction loss only
(frozen OS + rec). We use the original pair-wise method
DUSt3R [79] with linear head as a camera pose estima-
tion method to provide a reference for our analysis. For
DUSt3R we consider three variants: (i) off-the-shelf pre-
trained weights (DUSt3R†), (ii) initialized with pretrained
weights and trained with the same data as ours (DUSt3R*),
and (iii) trained from scratch with the same data as ours
(DUSt3R0).
In all three variants, we perform the final
global optimization step with our newly introduced tempo-
ral loss Ltranl and Lfl.
Metrics. We use the standard FID [29] and FVD [74] met-
rics to measure image and video quality, respectively. How-
ever, since such metrics do not reflect 3D consistency in
the generated videos, we additionally adopt the recently
proposed MEt3R [2] metric. We estimate the per-frame
DINO [13] features and warp them to the first frame using
the estimated 3D point maps. We report the average cosine
similarity between the features of the first frame and any
other warped feature map weighted by the visibility masks.
We validate the quality of camera pose estimation on
real videos (V2C) by comparing the estimated camera poses
(R, t) against the ground truth poses ( ¯R,¯t). For rotation,

we compute the relative error between two rotation ma-
trices [76]. Since the estimated and ground-truth transla-
tions can differ in scale, we follow [76] to compute the
angle between the two normalized translation vectors, i.e.,
arccos(t⊤¯t/(∥t∥∥¯t∥)). Besides reporting the average of the
two errors, we also follow [79] to report Relative Rotation
Accuracy (RRA) and Relative Translation Accuracy (RTA),
i.e., the percentage of camera pairs with rotation/translation
error below a threshold. Due to the small number of frames
handled by the video generator, each video sequence ex-
hibits small rotation variation. Hence, we select a threshold
5◦ to report RTA@5 and RRA@5. Additionally, we calcu-
late the mean Average Accuracy (mAA@30), defined as the
area under the curve accuracy of the angular differences at
min(RRA@30, RTA@30).

4.2. Generation Evaluation

For each method, we generate 180 videos using the captions
in RealEstate10K-test. We report the FID/FVD against
the real images/videos in RealEstate10K-test as well as the
MEt3R metric where we use JOG3R to estimate point maps.
Table 1 suggests that our full model generates more realis-
tic images/videos than pre-trained OpenSora (rows 1 and
4). Third row corresponds to a baseline where Lrec is dis-
abled by removing DUSt3R decoders/heads, i.e., it is equiv-
alent to standard diffusion model fine-tuning except only the
weights of the temporal attention layers are updated. We
see that fine-tuning with more data leads to lower FVD but
not lower MEt3R error. In contrast, JOG3R achieves the
lowest MEt3R error. This suggests that 3D consistency of
video generators cannot simply be improved with more data
whereas additional 3D-aware training tasks (point map es-
timation in our case) and losses are more effective. Finally,
if we fine-tune the video model only with the reconstruc-
tion loss (row 2), while 3D consistency is improved, FID or
FVD degrades. This is intuitive because without the gen-
eration loss, there is nothing to enforce the model to re-
tain its full generation capability. The supplemental web-
page shows examples of videos generated along with their
MEt3R error maps.

4.3. Reconstruction Evaluation

In Table 2, we compare the camera pose estimation (V2C)
errors on RealEstate10K-test and the withheld DL3DV10K.
When we freeze the video model and only train the recon-
struction decoder (row 1), we get noticeably worse results
than the trainable counterparts (rows 2 and 3). This shows
that the raw features from the pre-trained video generators
are not fully 3D consistent and hence ill suited for point map
estimation. Furthermore, we see that the models trained
with reconstruction loss only (row 2) lead to overall simi-
lar results to our full model, which uses both reconstruction
and generation losses. This confirms that the two tasks are

Figure 4. Qualitative camera pose estimation (V2C) results. Red to purple indicates the progression from the first to the last frame. Note
that on these test videos, JOG3R yields improved point maps leading to improved camera tracks compared to pretrained DUSt3R.

Method

Lgen

Rot.↓

Trans.↓

RRA@5◦↑

RTA@5◦↑ mAA@30◦↑

Rot.↓

Trans.↓

RRA@5◦↑

RTA@5◦ ↑ mAA@30◦↑

RealEstate10K-test

DL3DV10K

pre-trained OS
frozen OS + rec
JOG3R

DUSt3R† [79]
DUSt3R*
DUSt3R0

n.a.
✗
✓

n.a.
n.a.
n.a.

0.40
0.28
0.29

0.53
0.23
0.32

26.72
21.13
22.15

27.86
9.17
20.66

99.80%
99.90%
99.80%

99.12%
99.90%
99.70%

11.63%
19.23%
19.18%

13.70%
54.17%
15.33%

38.40%
49.49%
47.25%

40.20%
75.50%
47.47%

6.49
4.54
4.20

1.33
1.83
6.25

39.53
30.05
29.17

14.75
14.17
39.37

65.06%
76.27%
78.63%

97.53%
93.63%
65.84%

2.81%
7.88%
7.86%

47.03%
38.57%
3.41%

19.70%
33.60%
34.22%

65.73%
64.40%
19.73%

Table 2. V2C error comparison on RealEstate10K-test and DL3DV10K. DUSt3R† indicates pre-trained DUSt3R weights, whereas
DUSt3R* is further trained with the same training set as our method – RealEstate10K-train. DUSt3R0 denotes training with RealEstate10K-
train from scratch without initializing with pre-trained weights. In the sub-tables, bold is best in each sub-table; underlined is the second
place. JOG3R produces good camera estimates on RealEstate10K-test and acceptable quality on out of distribution data DL3DV10K.

compatible and do not degrade each other’s performance.

Our full method, JOG3R, performs overall better than
pre-trained DUSt3R† (rows 3 vs 6) but is inferior than
DUSt3R* (row 5) which fine-tunes DUSt3R† with our data.

This suggests that the pre-trained DUSt3R weights, which
were trained towards a 3D-aware task, contain richer in-
formation for camera pose estimation than the pre-trained
OpenSora weights. In the future, we would like to explore

correspondences between thefirst& the last frame(ours)camera paths(ours)camera paths(pretrained DUSt3R)camera paths(ground truth)Figure 5. Qualitative generation T2V+C results. It is coherent with the camera paths from T2V→V2C. Please see suppmat. for videos.

training the video model from scratch using both genera-
tion and reconstruction losses, which we believe will re-
sult in a more 3D-consistent feature space. Meanwhile, the
overall results in Table 2 suggests JOG3R is not as compet-
itive in generalization, which we attribute to the faster mo-
tion in DL3DV10K. DUSt3R† and DUSt3R* contain pre-
trained knowledge learned from matching wide-baseline
view pairs.

Figure 4 shows the qualitative comparison of our method
and baselines in terms of point maps and resultant camera
estimates. Since camera poses are estimated through reg-
istration, which builds 3D correspondences along the way,
we visualize the final camera trajectories as well as the cor-
respondence between the first and the last frame. One can
see that our method produces good camera trajectories sim-
ilar to pre-trained DUSt3R, which is a method tailored for
reconstruction only, but without generation loss. In some
cases ours are even closer to ground truth camera paths. The
tests indicate that video generation and 3D-consistency are
compatible tasks, and help improve each other.

Self consistency of T2V→V2C and T2V+C. Since JOG3R
can generate camera trajectories in two ways – cascading
T2V and V2C or the tightly coupled T2V+C pipeline – it is
worth comparing how much the two results differ. To test

this, we run the two pipelines with 100 prompts and report
0.45◦ average difference in rotation and 19.20◦ in transla-
tion; both are low errors compared with the corresponding
numbers in Table 2, indicating that the camera poses from
joint T2V+C pipeline are consistent with T2V→V2C. The
qualitative results in Figure 5 also confirm this conclusion.

5. Conclusions and Future Work

We have demonstrated that the tasks of video generation
and pointMap estimation (equivalently, camera estimation
for static scenes) can be made to be compatible and hence
can be simultaneously trained with a unified architecture.
To our knowledge, JOG3R is the first method for joint video
generation and 3D point map prediction. The new unified
architecture enables end-to-end training of the two tasks,
generates realistic and 3D-consistent videos, and achieves
competitive camera pose estimation performance. This
finding reveals that the tasks are synergetic – video gen-
erators can be made 3D-consistent without loss of visual
quality, and its internal features reused for 3D-aware tasks.
Since it is not trivial to obtain accurate camera annota-
tions for dynamic scenes, our analysis is limited to videos
of static scenes only. An important future direction is to

generated videos(T2V)correspondences between frame 1 & F (T2VàV2C)camera poses (T2VàV2C)camera poses (T2V+C)analyze how video generation features adapt to camera es-
timation for dynamic scenes. Also, the length of the video
sequences our method can handle is currently limited by
the number of frames the generator can synthesize. As the
video generators continue to improve to enable generation
of longer sequences, our proposed solutions will also natu-
rally extend to handling longer videos with larger baseline.

References

[1] Sameer Agarwal, Yasutaka Furukawa, Noah Snavely, Ian Si-
mon, Brian Curless, Steven M. Seitz, and Richard Szeliski.
Building rome in a day. 2009 IEEE 12th International Con-
ference on Computer Vision, pages 72–79, 2009. 2

[2] Mohammad Asim, Christopher Wewer, Thomas Wimmer,
Bernt Schiele, and Jan Eric Lenssen. Met3r: Measuring
multi-view consistency in generated images. In CVPR, 2025.
1, 2, 3, 6

[3] Sherwin Bahmani, Ivan Skorokhodov, Guocheng Qian, Ali-
aksandr Siarohin, Willi Menapace, Andrea Tagliasacchi,
David B. Lindell, and Sergey Tulyakov. Ac3d: Analyzing
and improving 3d camera control in video diffusion trans-
formers. Proc. CVPR, 2025. 2, 3

[4] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisser-
man. Frozen in time: A joint video and image encoder for
end-to-end retrieval. In ICCV, 2021. 1

[5] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Her-
rmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur,
Yuanzhen Li, Tomer Michaeli, et al. Lumiere: A space-
time diffusion model for video generation. arXiv preprint
arXiv:2401.12945, 2024. 2

[6] Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka,
and Matthias M¨uller. ZoeDepth: Zero-shot transfer by com-
bining relative and metric depth. arXiv, 2023. 5, 6

[7] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel
Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi,
Zion English, Vikram Voleti, Adam Letts, et al. Stable video
diffusion: Scaling latent video diffusion models to large
datasets. arXiv preprint arXiv:2311.15127, 2023. 1, 2
[8] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dock-
horn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis.
Align your latents: High-resolution video synthesis with la-
tent diffusion models. In CVPR, pages 22563–22575, 2023.
2

[9] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue,
Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luh-
man, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya
Ramesh. Video generation models as world simulators.
2024. 1, 2, 3

[10] Matthew A. Brown, Gang Hua, and Simon A. J. Winder. Dis-
criminative learning of local image descriptors. IEEE Trans-
actions on Pattern Analysis and Machine Intelligence, 33:
43–57, 2011. 2

[11] Junhao Cai, Yisheng He, Weihao Yuan, Siyu Zhu, Zilong
Dong, Liefeng Bo, and Qifeng Chen. Open-vocabulary
IEEE
category-level object pose and size estimation.
Robotics and Automation Letters, 2024. 3

[12] Qi Cai, Lilian Zhang, Yuanxin Wu, Wenxian Yu, and Dewen
Hu. A pose-only solution to visual reconstruction and navi-
gation. IEEE Transactions on Pattern Analysis and Machine
Intelligence, 45(1):73–86, 2021. 2

[13] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e Jegou,
Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerg-
ing properties in self-supervised vision transformers.
In
2021 IEEE/CVF International Conference on Computer Vi-
sion (ICCV), pages 9630–9640, 2021. 6

[14] Shoufa Chen, Mengmeng Xu, Jiawei Ren, Yuren Cong, Sen
He, Yanping Xie, Animesh Sinha, Ping Luo, Tao Xiang, and
Juan-Manuel Perez-Rua. Gentron: Delving deep into dif-
fusion transformers for image and video generation. arXiv
preprint arXiv:2312.04557, 2023. 2

[15] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace,
Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon,
Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang,
and Sergey Tulyakov. Panda-70M: Captioning 70m videos
In CVPR, pages
with multiple cross-modality teachers.
13320–13331, 2024. 1

[16] David J Crandall, Andrew Owens, Noah Snavely, and
Daniel P Huttenlocher. Sfm with mrfs: Discrete-continuous
IEEE
optimization for large-scale structure from motion.
transactions on pattern analysis and machine intelligence,
35(12):2841–2853, 2012. 2

[17] Hainan Cui, Xiang Gao, Shuhan Shen, and Zhanyi Hu.
In CVPR, pages

Hsfm: Hybrid structure-from-motion.
1212–1221, 2017. 2

[18] Prafulla Dhariwal and Alexander Nichol. Diffusion models
beat gans on image synthesis. Advances in neural informa-
tion processing systems, 34:8780–8794, 2021. 2

[19] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,
Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
Mostafa Dehghani, Matthias Minderer, Georg Heigold, Syl-
vain Gelly, et al. An image is worth 16x16 words: Trans-
formers for image recognition at scale. In ICLR, 2021. 4
[20] Niladri Shekhar Dutt, Sanjeev Muralikrishnan, and Niloy J
Mitra. Diffusion 3d features (diff3f): Decorating untextured
In CVPR, pages
shapes with distilled semantic features.
4494–4504, 2024. 1, 3

[21] Mohamed El Banani, Amit Raj, Kevis-Kokitsi Maninis, Ab-
hishek Kar, Yuanzhen Li, Michael Rubinstein, Deqing Sun,
Leonidas Guibas, Justin Johnson, and Varun Jampani. Prob-
ing the 3d awareness of visual foundation models. In CVPR,
pages 21795–21806, 2024. 3

[22] Martin A Fischler and Robert C Bolles. Random sample
consensus: a paradigm for model fitting with applications to
image analysis and automated cartography. Communications
of the ACM, 24(6):381–395, 1981. 4

[23] Haoyu Guo, Sida Peng, Haotong Lin, Qianqian Wang,
Guofeng Zhang, Hujun Bao, and Xiaowei Zhou. Neural 3d
scene reconstruction with the manhattan-world assumption.
In CVPR, pages 5501–5510, 2022. 2

[24] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang,
Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and
Bo Dai. AnimateDiff: Animate your personalized text-to-
In ICLR,
image diffusion models without specific tuning.
2024. 1

[25] Michal Havlena and Konrad Schindler. Vocmatch: Effi-
cient multiview correspondence for structure from motion.
In ECCV, 2014. 2

[26] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo
Dai, Hongsheng Li, and Ceyuan Yang. CameraCtrl: En-
abling camera control for text-to-video generation. arXiv
preprint arXiv:2404.02101, 2024. 5

[27] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and
Qifeng Chen. Latent video diffusion models for high-fidelity
long video generation. arXiv preprint arXiv:2211.13221,
2022. 2

[28] Eric Hedlin, Gopal Sharma, Shweta Mahajan, Hossam Isack,
Abhishek Kar, Andrea Tagliasacchi, and Kwang Moo Yi.
Unsupervised semantic correspondence using stable diffu-
sion. In NeurIPS, 2024. 3

[29] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner,
Bernhard Nessler, and Sepp Hochreiter. GANs trained by
a two time-scale update rule converge to a local nash equi-
librium. NeurIPS, 2017. 6

[30] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising dif-
fusion probabilistic models. In NeurIPS, pages 6840–6851,
2020. 2

[31] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang,
Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben
Poole, Mohammad Norouzi, David J Fleet, et al.
Imagen
video: High definition video generation with diffusion mod-
els. arXiv preprint arXiv:2210.02303, 2022. 2

[32] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William
Chan, Mohammad Norouzi, and David J Fleet. Video dif-
fusion models. In NeurIPS, pages 8633–8646, 2022. 2
[33] Aleksander Holynski, David Geraghty, Jan-Michael Frahm,
Chris Sweeney, and Richard Szeliski. Reducing drift in
structure from motion using extended features. In 2020 In-
ternational Conference on 3D Vision (3DV), pages 51–60.
IEEE, 2020. 2

[34] Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip
In
Isola. Position: the platonic representation hypothesis.
ICML. JMLR.org, 2024. 1

[35] Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip
Isola. Position: the platonic representation hypothesis.
In
ICML. JMLR.org, 2024. 1, 3, 4

[36] Hyeonho Jeong, Chun-Hao Paul Huang, Jong Chul Ye, Niloy
Mitra, and Duygu Ceylan. Track4gen: Teaching video diffu-
sion models to track points improves video generation, 2024.
2

[37] Hanwen Jiang, Arjun Karpur, Bingyi Cao, Qixing Huang,
and Andr´e Araujo. OmniGlue: Generalizable feature match-
In CVPR, pages
ing with foundation model guidance.
19865–19875, 2024. 3

[38] Nianjuan Jiang, Zhaopeng Cui, and Ping Tan. A global linear
method for camera pose registration. In ICCV, pages 481–
488, 2013. 2

[39] Karel Lenc and Andrea Vedaldi. Understanding image repre-
sentations by measuring their equivariance and equivalence.
In CVPR, pages 991–999, 2015. 4

[40] Karel Lenc and Andrea Vedaldi. Understanding image repre-
sentations by measuring their equivariance and equivalence.
In CVPR, pages 991–999, 2015. 2

[41] Vincent Lepetit, Francesc Moreno-Noguer, and Pascal Fua.
Ep n p: An accurate o (n) solution to the p n p problem. IJCV,
81:155–166, 2009. 4

[42] Vincent Leroy, Yohann Cabon, and Jerome Revaud. Ground-

ing image matching in 3d with mast3r, 2024. 3

[43] Dongxu Li, Junnan Li, Hung Le, Guangsen Wang, Silvio
Savarese, and Steven C.H. Hoi. LAVIS: A one-stop library
for language-vision intelligence. In Proceedings of the 61st
Annual Meeting of the Association for Computational Lin-
guistics (Volume 3: System Demonstrations), pages 31–41,
Toronto, Canada, 2023. Association for Computational Lin-
guistics. 6

[44] Zhaoshuo Li, Thomas M¨uller, Alex Evans, Russell H Tay-
lor, Mathias Unberath, Ming-Yu Liu, and Chen-Hsuan Lin.
Neuralangelo: High-fidelity neural surface reconstruction. In
CVPR, pages 8456–8465, 2023. 2

[45] Philipp Lindenberger, Paul-Edouard Sarlin, Viktor Larsson,
and Marc Pollefeys. Pixel-perfect structure-from-motion
with featuremetric refinement. In ICCV, pages 5987–5997,
2021. 2

[46] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin,
Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu,
et al. DL3DV-10K: A large-scale scene dataset for deep
In CVPR, pages 22160–22169,
learning-based 3d vision.
2024. 6

[47] Yin Lou, Noah Snavely, and Johannes Gehrke. MatchMiner:
Efficient spanning structure mining in large image collec-
tions. In ECCV, 2012. 2

[48] David G. Lowe. Distinctive image features from scale-

invariant keypoints. IJCV, 60:91–110, 2004. 2

[49] Bin Luo and Edwin R. Hancock. Procrustes alignment with
the em algorithm. In Computer Analysis of Images and Pat-
terns, pages 623–631, Berlin, Heidelberg, 1999. Springer
Berlin Heidelberg. 4

[50] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Zi-
wei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte:
Latent diffusion transformer for video generation. arXiv
preprint arXiv:2401.03048, 2024. 2, 3, 5

[51] Zeyu Ma, Zachary Teed, and Jia Deng. Multiview stereo with
cascaded epipolar raft. In ECCV, pages 734–750. Springer,
2022. 2

[52] Willi Menapace, Aliaksandr Siarohin, Ivan Skorokhodov,
Ekaterina Deyneka, Tsai-Shien Chen, Anil Kag, Yuwei
Fang, Aleksei Stoliar, Elisa Ricci, Jian Ren, et al. Snap
video: Scaled spatiotemporal transformers for text-to-video
synthesis. In CVPR, pages 7038–7048, 2024. 1, 2

[53] Zak Murez, Tarrence van As, James Bartolozzi, Ayan Sinha,
Vijay Badrinarayanan, and Andrew Rabinovich. Atlas: End-
to-end 3d scene reconstruction from posed images. In ECCV,
2020. 2

[54] Minheng Ni, Yabo Zhang, Kailai Feng, Xiaoming Li, Yiwen
Guo, and Wangmeng Zuo. Ref-diff: Zero-shot referring im-
age segmentation with generative models, 2023. 1

[55] Michael Oechsle, Songyou Peng, and Andreas Geiger.
UNISURF: Unifying neural implicit surfaces and radiance
fields for multi-view reconstruction. In ICCV, pages 5569–
5579, 2021. 2

[56] Evin Pınar ¨Ornek, Yann Labb´e, Bugra Tekin, Lingni Ma,
Cem Keskin, Christian Forster, and Tomas Hodan. Found-
pose: Unseen object pose estimation with foundation fea-
tures. arXiv preprint arXiv:2311.18809, 2023. 3

[57] Linfei Pan, D´aniel Bar´ath, Marc Pollefeys, and Jo-
hannes Lutz Sch¨onberger. Global structure-from-motion re-
visited. In ECCV, 2024. 2

[58] Karran Pandey, Paul Guerrero, Metheus Gadelha, Yannick
Hold-Geoffroy, Karan Singh, and Niloy J. Mitra. Diffusion
handles: Enabling 3d edits for diffusion models by lifting
activations to 3d. CVPR, 2024. 1

[59] William Peebles and Saining Xie. Scalable diffusion models

with transformers. In ICCV, pages 4195–4205, 2023. 2, 3

[60] Rui Peng, Xiaodong Gu, Luyang Tang, Shihe Shen, Fanqi
Yu, and Ronggang Wang. GenS: Generalizable neural sur-
In NeurIPS,
face reconstruction from multi-view images.
2023. 2

[61] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee,
Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and
Peter J Liu. Exploring the limits of transfer learning with a
unified text-to-text transformer. Journal of machine learning
research, 21(140):1–67, 2020. 5

[62] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vi-
In ICCV, pages

sion transformers for dense prediction.
12159–12168, 2021. 3

[63] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Bj¨orn Ommer. High-resolution image syn-
thesis with latent diffusion models. In CVPR, 2021. 1, 2, 5

[64] Mohamed Sayed, John Gibson, Jamie Watson, Victor Adrian
Prisacariu, Michael Firman, and Cl´ement Godard. Simplere-
con: 3d reconstruction without 3d convolutions. In ECCV,
2022. 3

[65] Johannes L Schonberger and Jan-Michael Frahm. Structure-
In CVPR, pages 4104–4113, 2016.

from-motion revisited.
2

[66] Johannes L Sch¨onberger, Enliang Zheng,

Jan-Michael
Frahm, and Marc Pollefeys. Pixelwise view selection for
In ECCV, pages 501–518.
unstructured multi-view stereo.
Springer, 2016. 2

[67] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An,
Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual,
Oran Gafni, et al. Make-a-video: Text-to-video generation
without text-video data. arXiv preprint arXiv:2209.14792,
2022. 2

[68] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Ab-
hishek Kumar, Stefano Ermon, and Ben Poole. Score-based
generative modeling through stochastic differential equa-
tions. arXiv preprint arXiv:2011.13456, 2020. 2

[69] Jiaming Sun, Xi Chen, Qianqian Wang, Zhengqi Li, Hadar
Averbuch-Elor, Xiaowei Zhou, and Noah Snavely. Neural
In ACM SIGGRAPH 2022
3d reconstruction in the wild.
conference proceedings, pages 1–9, 2022. 2

[70] Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng
Phoo, and Bharath Hariharan. Emergent correspondence
from image diffusion. In NeurIPS, 2023. 1, 3, 4

[71] Zachary Teed and Jia Deng. Deepv2d: Video to depth with
differentiable structure from motion. ArXiv, abs/1812.04605,
2018. 3

[72] Benjamin Ummenhofer and Vladlen Koltun. Adaptive sur-
face reconstruction with multiscale convolutional kernels. In
ICCV, pages 5651–5660, 2021. 2

[73] Benjamin Ummenhofer, Huizhong Zhou, Jonas Uhrig, Niko-
laus Mayer, Eddy Ilg, Alexey Dosovitskiy, and Thomas
Brox. DeMoN: Depth and motion network for learning
monocular stereo. In CVPR, pages 5622–5631, 2016. 3
[74] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach,
Rapha¨el Marinier, Marcin Michalski, and Sylvain Gelly.
In ICLR work-
FVD: A new metric for video generation.
shop, 2019. 6

[75] Sheng Wan, Tung-Yu Wu, Wing H. Wong, and Chen-Yi Lee.
In 2018 IEEE Interna-
Confnet: Predict with confidence.
tional Conference on Acoustics, Speech and Signal Process-
ing (ICASSP), pages 2921–2925, 2018. 4

[76] Jianyuan Wang, Christian Rupprecht, and David Novotny.
Posediffusion: Solving pose estimation via diffusion-aided
bundle adjustment. In ICCV, pages 9773–9783, 2023. 6
[77] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang,
Xiang Wang, and Shiwei Zhang. Modelscope text-to-video
technical report. arXiv preprint arXiv:2308.06571, 2023. 2
[78] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku
Komura, and Wenping Wang. Neus: Learning neural implicit
surfaces by volume rendering for multi-view reconstruction.
ArXiv, abs/2106.10689, 2021. 2

[79] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris
Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vi-
sion made easy. In CVPR, 2024. 1, 3, 4, 6, 7

[80] Yiqun Wang, Ivan Skorokhodov, and Peter Wonka. Improved
surface reconstruction using high-frequency details. ArXiv,
abs/2206.07850, 2022. 2

[81] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou,
Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo
Yu, Peiqing Yang, et al. Lavie: High-quality video gener-
ation with cascaded latent diffusion models. arXiv preprint
arXiv:2309.15103, 2023. 2

[82] Philippe Weinzaepfel, Vincent Leroy, Thomas Lucas, Ro-
main Br´egier, Yohann Cabon, Vaibhav ARORA, Leonid
Antsfeld, Boris Chidlovskii, Gabriela Csurka, and Jerome
Revaud. Croco: Self-supervised pre-training for 3d vision
tasks by cross-view completion. In NeurIPS, 2022. 4
[83] Bernhard P. Wrobel. Multiple view geometry in computer

vision. K¨unstliche Intell., 15:41, 2001. 2

[84] Changchang Wu. Towards linear-time incremental structure
from motion. 2013 International Conference on 3D Vision,
pages 127–134, 2013. 2

[85] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian
Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu
Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning
In
of image diffusion models for text-to-video generation.
ICCV, pages 7623–7633, 2023. 2

[86] Guangkai Xu, Wei Yin, Hao Chen, Chunhua Shen, Kai
Cheng, and Feng Zhao. Frozenrecon: Pose-free 3d scene
In ICCV, pages
reconstruction with frozen depth models.
9276–9286. IEEE, 2023. 3

[87] Xinyi Ye, Weiyue Zhao, Tianqi Liu, Zihao Huang, Zhiguo
Cao, and Xin Li. Constraining depth map geometry for

multi-view stereo: A dual-depth approach with saddle-
shaped depth cells. In ICCV, pages 17661–17670, 2023. 2

[88] Zhichao Yin and Jianping Shi. Geonet: Unsupervised learn-
ing of dense depth, optical flow and camera pose. In CVPR,
pages 1983–1992, 2018. 3

[89] Zehao Yu, Songyou Peng, Michael Niemeyer, Torsten Sat-
tler, and Andreas Geiger. Monosdf: Exploring monocu-
lar geometric cues for neural implicit surface reconstruction.
ArXiv, abs/2206.00665, 2022. 2

[90] Yuanwen Yue, Anurag Das, Francis Engelmann, Siyu Tang,
and Jan Eric Lenssen. Improving 2d feature representations
by 3d-aware fine-tuning. arXiv preprint arXiv:2407.20229,
2024. 3

[91] Guanqi Zhan, Chuanxia Zheng, Weidi Xie, and Andrew Zis-
serman. What does stable diffusion know about the 3d scene?
arXiv preprint arXiv:2310.06836, 2023. 3

[92] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu,
Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and
Mike Zheng Shou. Show-1: Marrying pixel and latent dif-
fusion models for text-to-video generation. arXiv preprint
arXiv:2309.15818, 2023. 2

[93] Junyi Zhang, Charles Herrmann, Junhwa Hur, Luisa Pola-
nia Cabrera, Varun Jampani, Deqing Sun, and Ming-Hsuan
Yang. A tale of two features: Stable diffusion complements
DINO for zero-shot semantic correspondence. In NeurIPS,
2023. 3

[94] Junyi Zhang, Charles Herrmann, Junhwa Hur, Eric Chen,
Varun Jampani, Deqing Sun, and Ming-Hsuan Yang. Telling
left from right: Identifying geometry-aware semantic corre-
spondence. In CVPR, pages 3076–3085, 2024. 3

[95] Zhe Zhang, Rui Peng, Yuxi Hu, and Ronggang Wang. Ge-
omvsnet: Learning multi-view stereo with geometry percep-
tion. In CVPR, pages 21508–21518, 2023. 2

[96] Wenliang Zhao, Yongming Rao, Zuyan Liu, Benlin Liu, Jie
Zhou, and Jiwen Lu. Unleashing text-to-image diffusion
models for visual perception. In ICCV, 2023. 3

[97] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen,
Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang
You. Open-sora: Democratizing efficient video production
for all, 2024. 1, 2, 3

[98] Huizhong Zhou, Benjamin Ummenhofer, and Thomas Brox.
DeepTAM: Deep tracking and mapping with convolutional
neural networks. IJCV, 128:756 – 769, 2019. 3

[99] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe,
Learning
arXiv preprint

and Noah Snavely.
view synthesis using multiplane images.
arXiv:1805.09817, 2018. 5

Stereo magnification:

Appendices

The supplementary material consists of this document and
the webpage. We provide qualitative results in our webpage.

A. Temporal Regularizer Terms

Given the estimated camera pose (Rf , tf ) and focal length
lf for each frame f , we define Lfl and Ltranl as below:

Lfl =

Ltranl =

(cid:88)

f
(cid:88)

f

∥lf − lf +1∥2
2

∥tf − tf +1∥2

2 + ∥vf − vf +1∥2

2

(3)

(4)

where vf = tf − tf +1. The first term of Ltranl encour-
ages static camera position while the second term encour-
ages constant velocity.

B. Architectural Design Choices

In Table 1 below, we compare the quality of estimated cam-
era poses using the feature maps in different DiT block bi.
Row 1c and 2c correspond respectively to our full JOG3R
model (row 1c in Table 1 and 2 in the main paper), which
is i=26. We first confirm using the features one block later,
i=27, does not result in significant difference (row 1d vs. 1c;
2d vs. 2c). Next, we consider i=10 and 20, representing
approximately one-third and two-third of total DiT blocks.
We see that compared to our results in the main paper (row
1c and 2c), i=10 yields lower errors in RealEstate10k-test
but higher errors in DL3DV10k (row 1a and 2a), suggesting
that using features of earlier blocks has a higher risk of poor
generalization. Meanwhile, i=20 attains the lowest errors in
both datasets, while ours remains on-par. We therefore con-
clude that features of later blocks, e.g., i ∈ [20, 27] is pre-
ferred than earlier blocks, and all later blocks should lead
to similar results. Instead of solely relying on one block
bi, one can potentially devise a module fusing features of
all DiT blocks and projecting to the input space of DUSt3R
decoders, which we consider future work.

Backbone / Method

bi

Rot. err. (◦) ↓

Transl. err. (◦) ↓

RRA@5◦ ↑

RTA@5◦ ↑ mAA@30◦ ↑

1a. trainable DiT
1b. trainable DiT
1c. trainable DiT
1d. trainable DiT

2a. trainable DiT
2b. trainable DiT
2c. trainable DiT
2d. trainable DiT

b10
b20
b26 (JOG3R)
b27

b10
b20
b26 (JOG3R)
b27

RealEstate10k

18.94
19.10
22.15
23.09

DL3DV10k

30.90
27.71
29.17
30.36

0.28
0.28
0.29
0.28

4.68
4.01
4.20
4.65

99.94%
99.85%
99.79%
99.80%

73.87%
77.91%
78.63%
77.15%

18.75%
21.59%
19.18%
17.74%

6.08%
10.56%
7.86%
7.98%

52.63%
53.35%
47.25%
46.88%

30.76%
36.62%
34.22%
32.86%

Table 3. Ablation study on which feature maps bi get passed to DUSt3R. Row 1c and 2c correspond respectively to our full JOG3R
model (row 1c in Table 1 and 2 in the main paper).

