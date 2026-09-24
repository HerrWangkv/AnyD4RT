5
2
0
2

p
e
S
6
2

]

G
L
.
s
c
[

2
v
6
5
4
1
0
.
2
0
5
2
:
v
i
X
r
a

Preprint

PROCESS REINFORCEMENT THROUGH IMPLICIT RE-
WARDS

Ganqu Cui1,2†∗, Lifan Yuan3†∗, Zefan Wang2∗, Hanbin Wang4∗, Yuchen Zhang1∗,
Jiacheng Chen1∗, Wendi Li2∗, Bingxiang He2∗, Yuchen Fan1,5∗, Tianyu Yu2∗, Qixin Xu2∗,
Weize Chen2, Jiarui Yuan2, Huayu Chen2, Kaiyan Zhang2, Xingtai Lv2, Shuo Wang2,
Yuan Yao2, Xu Han2, Hao Peng3, Yu Cheng1,6, Zhiyuan Liu2, Maosong Sun2,
Bowen Zhou1,2, Ning Ding2†
1Shanghai AI Lab
4Peking University
cuiganqu@pjlab.org.cn

2Tsinghua University
5Shanghai Jiaotong University

6CUHK
lifan4@illinois.edu

3University of Illinois Urbana-Champaign

ABSTRACT

Dense process rewards have proven a more effective alternative to the sparse
outcome-level rewards in the inference-time scaling of large language models
(LLMs), particularly in tasks requiring complex multi-step reasoning. While
dense rewards also offer an appealing choice for the reinforcement learning (RL)
of LLMs since their fine-grained rewards have the potential to address some inher-
ent issues of outcome rewards, such as training efficiency and credit assignment,
this potential remains largely unrealized. This can be primarily attributed to the
challenges of training process reward models (PRMs) online, where collecting
high-quality process labels is prohibitively expensive, making them particularly
vulnerable to reward hacking. To address these challenges, we propose PRIME
(Process Reinforcement through IMplicit rEwards), which enables online PRM
updates using only policy rollouts and outcome labels through implicit process
rewards. PRIME combines well with various advantage functions and forgoes
the dedicated reward model training phase that existing approaches require, sub-
stantially reducing the development overhead. We demonstrate PRIME’s effec-
tiveness on competitional math and coding. Starting from Qwen2.5-Math-7B-
Base, PRIME achieves a 15.1% average improvement across several key reason-
ing benchmarks over the SFT model. Notably, our resulting model, Eurus-2-7B-
PRIME, surpasses Qwen2.5-Math-7B-Instruct on seven reasoning benchmarks
with 10% of its training data.

1

INTRODUCTION

Dense process rewards, which provide feedback at each intermediate step rather than only the whole
trajectory, have proven effective in inference-time scaling of large language models (LLMs) on
challenging reasoning tasks (Uesato et al., 2022; Lightman et al., 2023; Wang et al., 2023; Yuan
et al., 2024b). On the training side, they also present superiority in the reinforcement learning (RL)
of LLMs, particularly in improving training efficiency (Sutton & Barto, 2018) and credit assign-
ment (Leike et al., 2018) compared with sparse outcome rewards. However, successful applications
of dense rewards in RL for LLMs are limited (Setlur et al., 2024), as current industry-leading models
primarily depend on verifiable outcome rewards and have not yet demonstrated meaningful progress
with dense rewards (DeepSeek-AI et al., 2025; Team et al., 2025).

We identify the central challenge as how to acquire and utilize high-quality dense rewards at scale,
which enables online process reward model (PRM) update efficiently. The reason is that, optimizing
towards a static reward model eventually leads to overoptimization or reward hacking (Gao et al.,
2022) due to distribution shift.
Ideally, this can be solved by improving the reward model on-
line (Leike et al., 2018). However, acquiring dense process labels for training is prohibitively more

∗Core Contributors.
†Project Lead.

1

 
 
 
 
 
 
Preprint

expensive. Existing methods either need to build complicated human annotation pipelines (Light-
man et al., 2023) or rely on estimation-based methods, which require about 10× more rollouts for
each step than sampling only the response-level trajectories (Wang et al., 2023; Kazemnejad et al.,
2024). Neither of them is scalable in online RL. Moreover, to the best of our knowledge, it remains
underexplored how to incorporate dense rewards into RL for LLMs.

In this work, we propose Process Reinforcement through Implicit Rewards (PRIME), a scalable
framework for enhancing reasoning capabilities via efficient reinforcement learning with dense
token-level rewards. At its core,
the framework employs recently proposed implicit process
reward modeling (Yuan et al., 2024b) to train dense reward models with only outcome-level labels.
This enables PRIME to perform online learning of reward signals using only outcome labels on
policy rollouts, thereby fundamentally mitigating reward hacking while maintaining the same
computational cost as traditional outcome reward models (ORMs). Besides scalability, PRIME also
(1) serves as a general method to fuse token-level dense rewards and sparse outcome rewards by
calculating their returns separately before summing together, which is compatible with diverse RL
algorithms (Williams, 1992; Kool et al., 2019; Shao et al., 2024; Ahmadian et al., 2024; Schulman
et al., 2017); (2) eliminates the dedicated reward modeling stage, which is required by existing
works, by simply initializing from the SFT model or even the base model (§ B.4). In summary,
starting from one single language model, the PRIME framework can efficiently accomplish the
generation of dense rewards, the initialization and updating of reward models, as well as the
reinforcement learning (RL) training of the policy model.

In experiments, we train Qwen2.5-Math-7B-Base (Yang et al., 2024b) with PRIME after a
lightweight SFT warmup stage. Compared to RL using outcome rewards only, PRIME achieves
a 2.5× sample efficiency gain and a 6.9% performance improvements on challenging math prob-
lems. As shown in Figure 12, through PRIME, we successfully achieve substantial improvement on
key mathematical reasoning benchmarks over the SFT model, leading to 16.7% improvement on
average, and over 20% on AMC&AIME competitions. Our final model Eurus-2-7B-PRIME sur-
passed Qwen2.5-Math-7B-Instruct on five key mathematical benchmarks. Notably, this is achieved
with only 10% of the data used by Qwen-Math, as in Table 3.

Our analysis shows that updating the PRM online is key to the success of PRIME (§5.1). We also
show that PRIME could generally boost various RL algorithms, including RLOO (Ahmadian et al.,
2024), REINFORCE (Williams, 1992), PPO (Schulman et al., 2017), and GRPO (Shao et al., 2024)
(§5.3). In terms of the design choices of advantage estimate, we observe that Implicit PRMs are
better to be used as reward models than value models (§5.4).

2 REINFORCEMENT LEARNING FOR LLMS AND THE CHALLENGES OF

INCOPORATING DENSE REWARDS

Reinforcement Learning (RL) aims to learn an optimal policy πθ that maximizes the expected cu-
mulative discounted reward, namely return, when interacting with an environment. In the context
of autoregressive language modeling, state at step t is the concatenation of prompt x and current
response y<t, and the action is the t-th token or step yt.

2.1 RL PRELIMINARIES FOR LLMS

Policy Gradient. Policy gradient is a fundamental algorithm that directly optimizes this objective.
Central to this approach is the advantage function At, which quantifies how much better an action is
compared to alternatives in a given state:

∇θJ(θ) = Ex∼D,y∼πθ

∇θ log πθ(yt|y<t)At

(1)

(cid:34) T

(cid:88)

(cid:35)

where (x, y) represents a pair of input and output. x is omitted for brevity. In practice, the advantage
function is implemented as cumulative discounted rewards subtracting a baseline:

t=0

At =

T
(cid:88)

s=t

γs−tr(ys) − b

2

(2)

Preprint

γ ∈ [0, 1] is a discount factor that optionally decays future rewards, and r(ys) is the reward provided
by the environment at time step s with x and y<s being omitted in conditions. Eq. 2 is the general
formula of the Monte-Carlo (MC) advantage estimate, which indicates that, the high-quality and
dense reward at each step is crucial for RL. Different choices of b include, e.g. directly using values
(Williams, 1992), group average of rewards (Shao et al., 2024), and leave-one-out average of rewards
(Ahmadian et al., 2024; Kool et al., 2019).

Value Models. Though the MC estimate is unbiased, it suffers from high variance because of the
reliance on all future actions and rewards, which can be random and noisy. Value models, which
predict expected accumulated rewards starting from a state, are adopted to help reduce the variance
in advantage estimation, such as Generalized Advantage Estimation (GAE; Schulman et al., 2016):
AGAE(γ,λ)
s=0(γλ)sδt+s, where δt = r(yt) + γV (y<t+1) − V (y<t) is the temporal difference
(TD) error (Sutton, 1988), V is a value model, and λ controls the bias-variance tradeoff in advantage
estimation. PPO (Schulman et al., 2017) is a representative of such actor-critic algorithms that
explicitly train a value model along with the policy.

= (cid:80)∞

t

Reward Sparsity. Although dense rewards can be naturally integrated into the advantage function
through Eq. 2, unfortunately, only outcome reward models (ORMs) are available in most practices
of LLMs, i.e., only the final token bears a meaningful reward while intermediate tokens receive no
rewards (Rafailov et al., 2023; Shao et al., 2024; DeepSeek-AI et al., 2025). In this bandit setting,
r(yt) = 0 for t < T while r(yT ) can be non-zero, and Eq. 2 becomes A = r(yT ) − b. This
formulation, while simpler, can suffer from reward sparsity issues as the policy receives feedback
only at the end of the entire generation. This may (1) encourage spurious solutions with incorrect
processes but correct answers, (2) largely reduce sample efficiency in training, and (3) encounter
the credit assignment problem (Sutton & Barto, 2018). These drawbacks could be further amplified
on complicated tasks, which require more thinking and execution steps, urging the need of dense
rewards (Uesato et al., 2022; Lightman et al., 2023). Some may consider employing a value model
to mitigate the problem, as it predicts values at every step t. However, previous work showed
that value models may not be able to solve the reward sparsity issue effectively due to training
challenges, despite the additional computation overhead (Shao et al., 2024; Ahmadian et al., 2024).
We will also empirically validate this claim in §5.4.

2.2 KEY CHALLENGES IN SCALABLE DENSE REWARDS

The way to mitigate the reward sparsity prob-
lem is to adopt dense reward models, namely
PRMs, which score model responses over each
token or step. However, it is usually infeasi-
ble in practice to incorporate dense rewards into
online RL because of three critical challenges
in implementation.

C1. Process rewards are hard to define.
It is difficult to collect step-level labels since
reasoning steps do not naturally occur in se-
quences. Although tokens are easily distin-
guishable, annotating labels for each token is
too costly. Moreover, defining the absolute
correctness of intermediate processes as dense
rewards can be ambiguous, as some incorrect
steps can also positively contribute to the final
answer by pruning searching branches (Ope-
nAI, 2024; DeepSeek-AI et al., 2025).

C2. PRM online updates are not scalable.
It is crucial to prevent reward overoptimiza-
tion or reward hacking, which requires the re-
ward model or value model to be updated online
along with the policy model (Schulman et al.,
2017; Gao et al., 2022). However, training PRMs often requires extensive nuanced step-level anno-

Figure 1: Illustration of PRIME. PRIME follows
that (1) initialize policy model and the Implicit
PRM both with the reference model; (2) sample
multiple responses for each prompt and filter with
output accuracy; (3) obtain implicit process re-
wards by the Implicit PRM and update it using
cross-entropy (CE) loss; (4) compute advantage
and policy loss then update the policy model.

3

SFT ModelImplicit PRMPolicy ModelImplicit PRMPolicy ModelPromptResponseOutcome Verifier𝒓𝒐𝒓𝒑UpdateUpdate𝝅𝒓𝒆𝒇𝝅𝒓𝒆𝒇SFT ModelPreprint

tation, which is infeasible in online RL training. Therefore, this brings about considerable scalability
and generalization concerns in dense rewards for RL.

C3. Explicit reward modeling brings extra cost. Training reward models requires extensive
annotation and broad data coverage to ensure a good balance between adaptability to the policy
distribution and generalization to distribution shifts. Hence, the explicit training stage introduces a
very costly data collection and an additional training overhead, especially for PRMs which typically
require stepwise labels.

Notably, DeepSeek-AI et al. (2025) shares similar conclusions and thus is impeded from incorpo-
rating PRMs into large-scale RL training.

3 PRIME

To address the above challenges, we propose PRIME, a scalable online RL method with dense re-
wards. The key insight of PRIME is to apply implicit process rewards, which are derivable from the
Implicit PRM that is trained with only outcome labels (Yuan et al., 2024b). This property enables us
to update the PRMs online to avoid reward hacking. We then design a flexible framework to incorpo-
rate implicit process rewards with outcome rewards into any kind of MC advantage estimate. PRIME
is illustrated in Figure 1 and Algorithm 1. Next, we will detail the implicit process rewards (§3.1) and
how we leverage them to calculate advantages (§3.2), and introduce other techniques we used (§3.3).

3.1 ENABLING SCALABLE REWARD UPDATE WITH IMPLICIT REWARD MODELING

We consider dense rewards from the Implicit PRM because of the scalability. In short, Implicit PRM
enables training an ORM with outcome labels only while repurposing it as a PRM at inference. The
training stage is the same as standard ORM pipelines, with the only difference being representing
the reward as rϕ(y) := β log πϕ(y)
πref(y) , where πϕ is the RM and πref is the reference model, both of
which are causal LMs. At inference, the process rewards are obtained by:

rϕ(yt) := β log

πϕ(yt|y<t)
πref(yt|y<t)

(3)

In PRIME, upon rollouts being generated and graded by the (ground truth) outcome verifier,
we update the Implicit PRM online with on-policy rollouts and outcome supervision and
then calculate token-level dense rewards to estimate advantages, which solves C1 and C2
mentioned in §2.2 respectively: (1) To prevent overoptimization and reward hacking, it is crucial to
update reward models online. However, updating previous PRMs (Lightman et al., 2023) requires
annotating step labels on the latest policy rollouts, which is neither efficient nor scalable during
online RL. In contrast, the Implicit PRM only demands outcome labels to train due to its special
reward representation, and thus it can be easily updated with policy rollouts and outcome labels or
rewards, both of which have already been collected to update the policy model.
(2) Unlike common PRMs that produce only step-level rewards, the Implicit PRM provides more
fine-grained token-level rewards at no additional cost. This addresses the ambiguity in identifying
steps in LLM responses while not introducing extra overhead, making it easy to combine with any
RL algorithms for advantage estimation. More discussions on Implicit PRMs are in § C.

3.2 ADVANTAGE ESTIMATION AND POLICY UPDATE

Estimating advantages using Monte Carlo estimator with a leave-one-out baseline. After ob-
taining token-level dense rewards, we calculate advantages based on either MC estimators or GAE.
To determine the advantage function in PRIME, we compare GAE with several MC estimators, in-
cluding REINFORCE (Williams, 1992), RLOO (Ahmadian et al., 2024), and GRPO (Shao et al.,
2024). Experimental details and results can be found in §5.3.

We find that MC estimators, despite being simpler, are strong enough to produce stable results.
Therefore, we choose MC estimate as our advantage function and despite PRIME being compatible
with any baseline estimation approaches, we instantiate it with a leave-one-out baseline from K

4

Preprint

Algorithm 1 Process Reinforcement through Implicit Rewards (PRIME)
Input Language model πθinit; outcome verifier ro; dataset D; sample number K; total iteration N .
1: Initialize policy model πθ, πθold ← πθinit , implicit PRM and reference model πϕ, πref ← πθinit
2: for iteration = 1, . . . , N do
3:
4:
5:
6:
7:

Sample batch of prompts B ∼ D
Generate K responses: {y1, ..., yK} ∼ πθ(·|x) for x ∈ B
(cid:0)y1:K(cid:1)
Compute outcome rewards: ro
Apply accuracy filter (§3.3) on all prompts: T ← Filter(x, y1:K, ro
Forward pass πϕ, πref on each (x, y) ∈ T to obatin implicit process reward rϕ(yt) with

(cid:0)y1:K(cid:1)) for x ∈ B

Eq. 3

8:

Update Implicit PRM πϕ by CE loss on (x, y, ro (y)) ∈ T :
LCE(ϕ) = −E(x,y,ro(y))∼T [ro (y) · log σ (rϕ (y)) + (1 − ro (y)) · log (1 − σ (rϕ (y)))]
Compute advantages A with Eq. 5
Update policy πθ by PPO loss in Eq. 6
Update old parameters: θold ← θ

9:
10:
11:
12: end for
Output Optimized policy model πθ

samples (Ahmadian et al., 2024) in this paper, as it performs better in the experiments:

Ai = ro(yi) −

1
K − 1

(cid:88)

j̸=i

ro(yj)

(4)

where ro(yi) denotes the reward of i-th response, K is the number of samples for one prompt. The
leave-one-out (LOO) baseline helps reduce variances.

More specifically, we use an Implicit PRM πϕ and an outcome verifier or reward model ro. We
calculate the return of implicit process rewards and outcome rewards separately if both are available,
since directly mixing their values may lead to numerical instability (Shao et al., 2024). For implicit
process rewards, we perform a three-step process to calculate return: (1) Use the averaged implicit
process rewards to calculate the leave-one-out baseline; (2) Normalize the process reward at step
t by subtracting the baseline; (3) Calculate the discounted return for each response. For outcome
rewards, we directly adopt LOO without any modification. Finally, the advantage is set to the
combination of both returns:
|yi|
(cid:88)

(cid:88)

(cid:88)



(cid:0)yj(cid:1)

rϕ

+ ro



(cid:0)yi(cid:1) −

1
K − 1

j̸=i

(cid:0)yj(cid:1)

ro

(5)


rϕ(yi

s) −

1
K − 1

j̸=i

(cid:123)(cid:122)
RLOO with implicit process rewards

(cid:125)

(cid:124)

(cid:123)(cid:122)
RLOO with outcome rewards

(cid:125)

Ai

t =

γs−t ·

s=t
(cid:124)

Updating policy with PPO loss. We adopt PPO clip surrogate loss for more stable policy updates:

LCLIP(θ) =Et

(cid:34)

min

(cid:18) πθ(yt|y<t)
πθold(yt|y<t)

At, clip

(cid:16) πθ(yt|y<t)
πθold(yt|y<t)

, 1 − ϵ, 1 + ϵ

(cid:19)(cid:35)

(cid:17)

At

(6)

where ϵ is a clipping parameter. The loss prevents the updated policy from deviating too far from
the original distribution, which is the prerequisite of importance sampling.

3.3 OTHER TECHNIQUES

Initializing PRM with SFT/base model. In practice, we find that the starting policy model itself
serves as a decent initialization of PRM, bypassing the PRM training stage. This solves C3 in §2.2
and outperforms a dedicatedly trained PRM, as shown in § 5.1.

Online Prompt Filtering. As we sample multiple trajectories for each prompt, we introduce online
prompt filtering which filters prompts within a certain accuracy range. This (1) preserves only the
prompts within a certain median-level difficulty range (Yang et al., 2024b) and (2) balances data
distribution for the Implicit PRM online training.

5

Preprint

Table 1: Detailed results of PRIME and RLOO w/ outcome verifier (OV). At the same 240 steps, the
model trained by PRIME is generally better than the model trained by outcome rewards. We also
reported avg@16 results in Table 12.

Method

Step AIME 2024 AMC MATH-500 MinervaMath OlympiadBench LeetCode LiveCodeBench Avg.

GPT-4o
Llama-3.1-70B-Inst.
Qwen2.5-Math-7B-Inst.
Eurus-2-7B-SFT

RLOO w/ OV Only

Eurus-2-7B-PRIME

-
-
-
0

240

80
160
240
320
592

9.3
20.0
13.3
3.3

20.0

20.0
13.3
20.0
16.7
26.7

45.8
37.3
50.6
30.1

47.0

41.0
42.2
50.6
51.8
57.8

76.4
65.0
79.8
66.2

73.2

68.2
72.0
78.2
77.8
79.2

36.8
37.1
34.6
32.7

36.4

38.2
37.1
39.3
39.7
38.6

43.3
30.5
40.7
29.8

35.4

37.0
38.7
40.3
41.5
42.1

58.9
35.0
11.7
21.7

28.3

26.7
26.7
31.1
36.1
33.3

48.8
34.4
11.3
17.8

26.7

26.6
25.6
27.5
28.5
28.6

45.6
37.0
34.6
28.8

36.9

36.8
36.5
41.0
41.7
43.9

We present the ablation study results in Figure 2 us-
ing RLOO with outcome rewards only, from which we
can see that the online prompt filter largely lowers the
variance of RL training.

How PRIME addresses challenges in §2.2. In sum-
mary, as illustrated in Figure 1 and Algorithm 1,
PRIME adopts implicit process rewards for efficient
PRM online update (C2), then integrates token-level
dense rewards with outcome rewards in MC advantage
estimate (C1). The PRMs are directly initialized from
SFT or base models, which foregoes explicit reward
modeling (C3).

4 EXPERIMENTS

Figure 2: Effect of online prompt filtering.

We first perform supervised finetuning on the base model to get a starter model for RL. Please refer
to § D for more details of this stage. All experiments are conducted on 8×A800 GPUs.

4.1 RL SETTINGS

Rule-based Outcome Verifier. Consistent with recent research that adopts an exact match with
ground truth as unhackable rewards (Gao et al., 2024; Lambert et al., 2024; DeepSeek-AI et al.,
2025), we define the rule-based ground truth outcome verifiers (OV) for math and coding as follows:

rmath
o

(y) =

(cid:26)1, matched
otherwise

0,

rcode
o

(y) =

(cid:80) #passes
(cid:80) #test cases

Hyperparameters. We use veRL (Sheng et al., 2024) to conduct experiments. By default, we
initialize the Implicit PRM with SFT model and retain the SFT model for reference logprobs. For
hyperparameters, we use a constant 5 × 10−7 learning rate together with AdamW optimizer for
policy model, and use a 10−6 learning rate for PRMs. Both policy and PRMs use a batch size of
256 and micro batchsize of 8. The rollout stage collects 256 prompts and samples 4 responses for
each prompt. We set β = 0.05 for PRM training. We set KL coefficient to 0 in all experiments.

Evaluation Benchmarks. We evaluate on 7 reasoning benchmarks, focusing on competition-level
mathematics and programming tasks, including AIME 2024 (Li et al., 2024), AMC (Li et al.,
2024), MATH-500 (Hendrycks et al., 2021b), Minerva Math (Lewkowycz et al., 2022), Olympiad-
Bench (He et al., 2024), LeetCode (Guo et al., 2024), and LiveCodeBench (v2) (Jain et al., 2024).

4.2 MAIN RESULTS

As shown in Figure 12 and Table 1, Eurus-2-7B-PRIME achieves substantial improvements on key
reasoning benchmarks over the SFT version of the model, leading to 15.1% improvement on av-
erage, and over 20% on AMC and AIME competitions. Besides, Eurus-2-7B-PRIME achieves

6

050100150200Steps0.360.380.400.420.440.460.48Outcome Training Rewardsw/ filterw/o filterPreprint

(a) Outcome training rewards (10-step moving).

(b) Test accuracy across different gradient steps.

Figure 3: The effect of dense reward. We compare PRIME and RLOO with outcome verifier (OV).
PRIME leads to 2.5× sample efficiency (wall clock as X axis can be found in Figure 17) and 6.9%
performance improvement. PRIME also substantially outperforms RLOO on downstream tasks.

(a) Outcome training rewards (10-step moving).

(b) Test accuracy across different gradient steps.

Figure 4: Comparison of different PRMs. Online PRM initialized from SFT model achieved the
best results. However, using PRMs trained on extra rollouts hurts the performance.

26.7% pass@1 on AIME 2024, surpassing GPT-4o, Llama-3.1-70B-Instruct, and Qwen2.5-Math-
7B-Instruct, demonstrating its excellent reasoning ability. Additional results are in § B.

4.3 DENSE REWARDS V.S. SPARSE REWARDS

Performance. We first validate the effect of PRIME with dense rewards compared to RLOO with
outcome rewards only. We train this model for 240 steps. For PRIME, we use the same setting and
train the model for 592 steps. We plot the training rewards measured by the outcome verifier and
test accuracy in Figure 3. Compared with sparse reward, PRIME improves the final rewards
by 6.9%, with lower variances. On downstream tasks, PRIME also consistently outperforms OV
only setup. Detailed results are listed in Table 1.

Table 2: Step-wise time cost of PRIME and RLOO.

Training Efficiency. We provide de-
tailed training time of each training
step for PRIME and RLOO in Ta-
ble 2. PRIME requires 24% more
time cost compared with RLOO.
However, as shown in Figure 3,
PRIME only takes 40% of the training steps to achieve the same training rewards as RLOO. This
means PRIME would still be 2× more efficient than RLOO when estimated by training time.
Additionally, the single controller design of veRL requires no extra GPU memory since all other
components (policy model, rollout engine) would be offloaded to CPU during PRM update.

Time(s) Rollout Policy update PRM update Others

PRIME

RLOO

157.9

156.6

281.7

150.9

282.4

680.3

530.7

Sum

91.1

90.4

0

7

050100150200Steps0.360.380.400.420.440.460.480.50Outcome Training Rewards2.5x Efficient6.9% HigherPRIMERLOO w/ OV Only0326496128160192224256288320Steps30323436384042Avg. Test AccPRIMERLOO w/ OV Only050100150200Steps0.360.380.400.420.440.460.480.50Outcome Training RewardsPRIME w/ online SFT PRMPRIME w/ offline EurusPRMPRIME w/ online EurusPRM0326496128160192224Steps303234363840Avg. Test AccPRIME w/ online SFT PRMPRIME w/ offline EurusPRMPRIME w/ online EurusPRMPreprint

5 ANALYSIS

5.1 DESIGN CHOICES FOR THE IMPLICIT PRM

The Implicit PRM is the key component of PRIME, and its design choices greatly affect RL. In this
section, we explore two major factors: (1) the initialization model and (2) the update mechanism.

SFT model initializes a good PRM. Conventionally, we
need to collect data to train RMs and PRMs, and then we
can use them in RL. However, the Implicit PRM is a lan-
guage model, so we can initialize it from any language
model with the same tokenizer as the policy model. To
investigate whether it is still necessary to train a PRM in
advance, we conduct experiments with different PRM ini-
tialization strategies: with the SFT model itself and with
a specially trained PRM. For the later one, we train Eu-
rusPRM from Eurus-2-7B-SFT with additional 500K data
generated by Llama3.1 and Qwen2.5 series (data details
in § E.5).

Sur-
We report the experiment results in Figure 4.
prisingly, directly using Eurus-2-7B-SFT to initialize
the PRM greatly outperforms EurusPRM which was
trained on more samples. We conjecture that initializing policy model and PRM from the same
model largely alleviates the distribution shift issue, as the PRM is only trained on the online rollouts
from the policy model.

Impact of PRM online
Figure 5:
update.
Offline PRM is gradually
been overoptimized while online PRMs
achieve higher accuracy during training.

Online PRM update is essential. To verify the effect of online PRM update, we pair the correct
and wrong samples and calculate the PRM prediction accuracy using rϕ(y). We report the PRM
classification accuracy in Figure 5. The figure clearly shows that, online update mitigates overop-
timization and reward hacking. The offline PRM, though starting with high accuracy, gradually
drops during RL training procedure due to distribution shift.
In contrast, online PRMs that are
trained on policy rollouts show the reverse curve.

This is further validated with training rewards and downstream performance. To breakdown, Eurus-
2-7B-SFT is both used as PRM initialization and the reference model in the main experiment, so
the PRM is totally trained from scratch, which means the initial PRM outputs zero reward for all
tokens. Therefore, Figure 3 also demonstrates the effect of online PRM update. For EurusPRM
initialization, the online run outperforms the offline run as well in Figure 4.

5.2 SCALING PRIME WITH MORE COMPUTE

(a) PRIME training up to 800 steps.

(b) PRIME training with 16 rollouts per prompt.

Figure 6: RL training with more training steps (Left) and larger rollout numbers (Right).

To validate the scalability of PRIME with increased computational resources, we first conduct an
extended training process. Specifically, we conduct RL training for 800 rollout steps (3200 gradient
steps) with PRIME and RLOO with outcome-reward only. The training results, shown in Figure 6

8

050100150200Steps0.500.550.600.650.700.75PRM AccuracyPRIME w/ online SFT PRMPRIME w/ offline EurusPRMPRIME w/ online EurusPRM0100200300400500600700800Steps2628303234363840Avg. AccuracyTest PeformancePRIMERLOO w/ OV Only0100200300400500600Steps2628303234363840Avg. AccuracyTest PeformancePRIMERLOO w/ OV OnlyPreprint

(Left), reveal that throughout the training, PRIME consistently exhibits stable growth and outper-
forms the baseline with an improvement of 3.7%. Moreover, we increase the number of responses
sampled for each prompt from 4 to 16. The results in Figure 6 (Right) show that PRIME achieves
non-trivial improvement of approximately 4.4% compared to RLOO.

5.3 PRIME WITH OTHER RL ALGORITHMS

As we stated before, PRIME is equally applicable to other RL algorithms beyond RLOO. In this
section, we implement PRIME with REINFORCE (Williams, 1992), GRPO (Shao et al., 2024),
and PPO (Schulman et al., 2017). Similarly to RLOO, we only modify the advantage estimation
functions and leave the clip surrogate loss unchanged. Detailed functions can be found in § B.3.
From Figure 7 and Table 4, We show that PRIME boosts these algorithms on both efficiency and
performance as it does with RLOO. PRIME contributes consistently regardless of the policy update
method, making it a generic algorithm. It indicates that PRIME is a general plug-in for almost
any RL algorithm for LLM., which largely extends the use cases of PRIME.

Figure 7: PRIME also generally benefits REIN-
FORCE, GRPO, and PPO.

Figure 8: Comparison of value models and pro-
cess reward models.

5.4 VALUE OR REWARD, HOW TO USE THE IMPLICIT PRM?

Besides using process rewards to estimate returns, we can also employ the Implicit PRM to pre-
dict values for advantage estimation in Eq. 2. Therefore, we compare four variants of MC esti-
mate to determine the best way to incorporate dense supervision. Recall that the Implicit PRM has
vϕ(y<t+1) = (cid:80)t
πref(yi|y<i) with the process reward being rϕ(yt) = vϕ(y<t+1)−vϕ(y<t),
and we assume a ground-truth outcome verifier ro, γ = 1, then we represent the variants as follows:

i=1 β log πϕ(yi|y<i)

(1) REINFORCE: At = ro(y). (2) On top of (1), using a linear-head value model V to calculate
the baseline: At = ro(y) − V (y<t). This is the original PPO in Figure 7 as we set γ = 1 and
λ = 1. (3) On top of (1), using values from the Implicit PRM to serve as the baseline: At =
ro(y) − vϕ(y<t). This is equivalent to PPO with its value model being replaced by values from the
Implicit PRM when γ = 1 and λ = 1. (4) On top of (1), using process rewards from the Implicit
PRM to calculate the return: At = ro(y) + (cid:80)T
s=t rϕ(ys). This is exactly the REINFORCE w/
PRIME in Figure 7.

Figure 8 reports the results. Comparing PPO and REINFORCE, we find that an additional value
model does not benefit policy performance. Notably, using rewards from the Implicit PRM to cal-
culate returns, which is the default setting in PRIME, greatly outperforms all three baselines. This
indicates that PRMs work better than value models in RL for LLMs. both two kinds of value models
(PPO value model and Implicit PRM as value model) fall behind reward models.

6 RELATED WORK

RL for LLM Reasoning. In the area of LLMs, reinforcement learning has been widely used for
aligning human preferences (Christiano et al., 2017; Ouyang et al., 2022; Cui et al., 2024), but the

9

050100150200250Steps0.360.380.400.420.440.460.48Outcome Training RewardsREINFORCEGRPOPPOREINFORCE w/ PRIMEGRPO w/ PRIMEPPO w/ PRIME050100150200250Steps0.360.380.400.420.440.460.48Outcome Training RewardsREINFORCE+ linear-head value model+ Implicit PRM as value+ Implicit PRM as rewardPreprint

open-source community mostly adopt imitation learning methods (Yuan et al., 2024a; Yue et al.,
2024; Wei et al., 2024; Liu et al., 2024) to enhance the reasoning capabilities of LLMs. Over the
past few months, the paradigm gradually shifted. OpenAI o1 (Jaech et al., 2024) first showed the
tremendous potential of large-scale RL for reasoning LLMs, and recent works have verified the scal-
ing effect of RL with outcome rewards (DeepSeek-AI et al., 2025; Team et al., 2025). Meanwhile,
the role of dense rewards in RL remains underexplored, serving the main focus of PRIME.

Implicit Rewards. Implicit rewards are broadly adopted in LLM alignment (Rafailov et al., 2023;
Chen et al., 2024b; Azar et al., 2024; Ethayarajh et al., 2024; Rosset et al., 2024; Chen et al., 2024a).
Rafailov et al. (2024) first showed that optimizing DPO objective learns a Q function implicitly.
Zhou et al. (2024) utilized implicit rewards in PPO, and showed the effectiveness of dense implicit
rewards. Yuan et al. (2024b) further extended the conclusion to any loss function optimizing Eq. 3.

7 CONCLUSION

As the fuel of LLMs, data, will be depleted in the near future, we are entering a new era of experi-
ence, which is exemplified by RL (Sutton, 2019). This work develops PRIME, which produces and
leverages dense rewards in online RL for LLM reasoning. Throughout the experiments, we validate
that PRIME (1) greatly benefits sample efficiency and policy performance, (2) is easy to use with
minimum cost, and (3) is a general method that works with broad RL algorithms together.

10

Preprint

ETHICS STATEMENT

This paper presents PRIME whose goal is to advance the field of reinforcement learning for LLMs.
There are many potential societal consequences of our work, none of which we feel must be specif-
ically highlighted here.

REPRODUCIBILITY STATEMENT

We have provided sufficient details to for reproduction, including algorithm pseudocode in Algo-
rithm 1, experiment configurations and hyperparameters in Section 4 and Appendix. We have upload
our code in Supplementary Material.

REFERENCES

Arash Ahmadian, Chris Cremer, Matthias Gall´e, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin,
Ahmet ¨Ust¨un, and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learn-
ing from human feedback in llms. arXiv preprint arXiv:2402.14740, 2024.

Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal
Valko, and R´emi Munos. A general theoretical paradigm to understand learning from human
preferences. International Conference on Artificial Intelligence and Statistics, abs/2310.12036,
2024.

Changyu Chen, Zichen Liu, Chao Du, Tianyu Pang, Qian Liu, Arunesh Sinha, Pradeep Varakan-
tham, and Min Lin. Bootstrapping language models with dpo implicit rewards. arXiv preprint
arXiv:2406.09760, 2024a.

Huayu Chen, Guande He, Lifan Yuan, Ganqu Cui, Hang Su, and Jun Zhu. Noise contrastive align-

ment of language models with explicit rewards. arXiv preprint arXiv:2402.05369, 2024b.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep
reinforcement learning from human preferences. Advances in neural information processing sys-
tems, 30, 2017.

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, Yuan Ni, Guotong Xie,
Ruobing Xie, Yankai Lin, Zhiyuan Liu, and Maosong Sun. Ultrafeedback: Boosting language
models with scaled ai feedback. In ICML, 2024.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu,
Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu,
Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao
Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan,
Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao,
Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding,
Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang
Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai
Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang,
Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang,
Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang,
Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang,
R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng
Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing
Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen
Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong
Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu,
Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xi-
aosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia
Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng
Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong
Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong,

11

Preprint

Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou,
Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying
Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda
Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu,
Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu
Zhang, and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforce-
ment learning, 2025. URL https://arxiv.org/abs/2501.12948.

Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Kto: Model

alignment as prospect theoretic optimization. ICML, 2024.

Jiaxuan Gao, Shusheng Xu, Wenjie Ye, Weiling Liu, Chuyi He, Wei Fu, Zhiyu Mei, Guangju
Wang, and Yi Wu. On designing effective rl reward at training time for llm reasoning. ArXiv,
abs/2410.15115, 2024.

Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In

International Conference on Machine Learning, 2022.

Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao
Bi, Yu Wu, YK Li, et al. Deepseek-coder: When the large language model meets programming–
the rise of code intelligence. arXiv preprint arXiv:2401.14196, 2024.

Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine. Reinforcement learning with
deep energy-based policies. In Doina Precup and Yee Whye Teh (eds.), Proceedings of the 34th
International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August
2017, volume 70 of Proceedings of Machine Learning Research, pp. 1352–1361. PMLR, 2017.
URL http://proceedings.mlr.press/v70/haarnoja17a.html.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han,
Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. OlympiadBench:
A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scien-
tific problems. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the
62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Pa-
pers), pp. 3828–3850, Bangkok, Thailand, August 2024. Association for Computational Linguis-
tics. doi: 10.18653/v1/2024.acl-long.211. URL https://aclanthology.org/2024.
acl-long.211/.

Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin
Burns, Samir Puranik, Horace He, Dawn Song, et al. Measuring coding challenge competence
with apps. arXiv preprint arXiv:2105.09938, 2021a.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song,
and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv
preprint arXiv:2103.03874, 2021b.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec
Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv
preprint arXiv:2412.16720, 2024.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando
Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free
evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.

Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy,
Aaron Courville, and Nicolas Le Roux. Vineppo: Unlocking rl potential for llm reasoning through
refined credit assignment. arXiv preprint arXiv:2410.01679, 2024.

Wouter Kool, Herke van Hoof, and Max Welling. Buy 4 reinforce samples, get a baseline for
free! In DeepRLStructPred@ICLR, 2019. URL https://api.semanticscholar.org/
CorpusID:198489118.

12

Preprint

Nathan Lambert, Jacob Daniel Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze
Brahman, Lester James Validad Miranda, Alisa Liu, Nouha Dziri, Xinxi Lyu, Yuling Gu, Saumya
Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris
Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and Hanna Hajishirzi.
T¨ulu 3: Pushing frontiers in open language model post-training. ArXiv, abs/2411.15124, 2024.

Jan Leike, David Krueger, Tom Everitt, Miljan Martic, Vishal Maini, and Shane Legg. Scalable
agent alignment via reward modeling: a research direction. arXiv preprint arXiv:1811.07871,
2018.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ra-
masesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative
reasoning problems with language models. Advances in Neural Information Processing Systems,
35:3843–3857, 2022.

Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif
Rasul, Longhui Yu, Albert Q Jiang, Ziju Shen, et al. Numinamath: The largest public dataset in
ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository,
13:9, 2024.

Rongao Li, Jie Fu, Bo-Wen Zhang, Tao Huang, Zhihong Sun, Chen Lyu, Guang Liu, Zhi Jin, and
Ge Li. Taco: Topics in algorithmic code generation dataset. arXiv preprint arXiv:2312.14852,
2023.

Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, R´emi Leblond, Tom
Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, Thomas Hubert, Peter Choy, Cyprien
de Masson d’Autume, Igor Babuschkin, Xinyun Chen, Po-Sen Huang, Johannes Welbl, Sven
Gowal, Alexey Cherepanov, James Molloy, Daniel Mankowitz, Esme Sutherland Robson, Push-
meet Kohli, Nando de Freitas, Koray Kavukcuoglu, and Oriol Vinyals. Competition-level code
generation with alphacode. arXiv preprint arXiv:2203.07814, 2022.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harrison Edwards, Bowen Baker, Teddy Lee,
Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. ArXiv,
abs/2305.20050, 2023.

Zhenghao Lin, Zhibin Gou, Yeyun Gong, Xiao Liu, Yelong Shen, Ruochen Xu, Chen Lin, Yujiu
Yang, Jian Jiao, Nan Duan, and Weizhu Chen. Rho-1: Not all tokens are what you need, 2024.

Zihan Liu, Yang Chen, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Acemath: Ad-
arXiv preprint

vancing frontier math reasoning with post-training and reward modeling.
arXiv:2412.15084, 2024.

Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin
Cai, Jeffrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-
preview with a 1.5b model by scaling rl. https://pretty-radio-b75.notion.site/
DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2,
2025. Notion Blog.

Andrew Y. Ng, Daishi Harada, and Stuart Russell. Policy invariance under reward transformations:
Theory and application to reward shaping. In Ivan Bratko and Saso Dzeroski (eds.), Proceedings
of the Sixteenth International Conference on Machine Learning (ICML 1999), Bled, Slovenia,
June 27 - 30, 1999, pp. 278–287. Morgan Kaufmann, 1999.

OpenAI. Openai o1 system card. ArXiv, abs/2412.16720, 2024.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong
Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to fol-
low instructions with human feedback. Advances in neural information processing systems, 35:
27730–27744, 2022.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea
Finn. Direct preference optimization: Your language model is secretly a reward model. Advances
in Neural Information Processing Systems, 36, 2023.

13

Preprint

Rafael Rafailov, Joey Hejna, Ryan Park, and Chelsea Finn. From r to q∗: Your language model is

secretly a q-function. arXiv preprint arXiv:2404.12358, 2024.

Corby Rosset, Ching-An Cheng, Arindam Mitra, Michael Santacroce, Ahmed Awadallah, and
Tengyang Xie. Direct nash optimization: Teaching language models to self-improve with general
preferences. ArXiv, abs/2404.03715, 2024.

John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, and Pieter Abbeel. High-
In 4th International
dimensional continuous control using generalized advantage estimation.
Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016,
Conference Track Proceedings, 2016.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy

optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal,
Alekh Agarwal, Jonathan Berant, and Aviral Kumar. Rewarding progress: Scaling automated
process verifiers for llm reasoning. arXiv preprint arXiv:2410.08146, 2024.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathe-
matical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.
03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng,
Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint
arXiv: 2409.19256, 2024.

SkunkworksAI. reasoning-0.01, 2024.

Richard Sutton. The bitter lesson. Incomplete Ideas (blog), 13(1):38, 2019.

Richard S Sutton. Learning to predict by the methods of temporal differences. Machine learning,

3:9–44, 1988.

Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press, 2018.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun
Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with
llms. arXiv preprint arXiv:2501.12599, 2025.

Qwen Team. Qwq: Reflect deeply on the boundaries of the unknown, November 2024. URL

https://qwenlm.github.io/blog/qwq-32b-preview/.

Shubham Toshniwal, Wei Du, Ivan Moshkov, Branislav Kisacanin, Alexan Ayrapetyan, and Igor
Gitman. Openmathinstruct-2: Accelerating ai for math with massive open-source instruction
data. arXiv preprint arXiv:2410.01560, 2024.

Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia
Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process-and
outcome-based feedback. arXiv preprint arXiv:2211.14275, 2022.

Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Y.Wu, and Zhifang
Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. ArXiv,
abs/2312.08935, 2023.

Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, and Lingming Zhang. Magicoder: Empowering
code generation with oss-instruct. In Forty-first International Conference on Machine Learning,
2024.

Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement

learning. Machine learning, 8:229–256, 1992.

14

Preprint

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li,
Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang,
Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin
Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li,
Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan,
Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint
arXiv:2412.15115, 2024a.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu,
Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu,
Xingzhang Ren, and Zhenru Zhang. Qwen2.5-math technical report: Toward mathematical ex-
pert model via self-improvement, 2024b. URL https://arxiv.org/abs/2409.12122.

Lifan Yuan, Ganqu Cui, Hanbin Wang, Ning Ding, Xingyao Wang, Jia Deng, Boji Shan, Huimin
Chen, Ruobing Xie, Yankai Lin, Zhenghao Liu, Bowen Zhou, Hao Peng, Zhiyuan Liu, and
Maosong Sun. Advancing llm reasoning generalists with preference trees. ArXiv, 2024a.

Lifan Yuan, Wendi Li, Huayu Chen, Ganqu Cui, Ning Ding, Kaiyan Zhang, Bowen Zhou, Zhiyuan
Liu, and Hao Peng. Free process rewards without process labels, 2024b. URL https://
arxiv.org/abs/2412.01981.

Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen.
Mammoth: Building math generalist models through hybrid instruction tuning. arXiv preprint
arXiv:2309.05653, 2023.

Xiang Yue, Tuney Zheng, Ge Zhang, and Wenhu Chen. Mammoth2: Scaling instructions from the

web. ArXiv, abs/2405.03548, 2024.

Kaiyan Zhang, Sihang Zeng, Ermo Hua, Ning Ding, Zhang-Ren Chen, Zhiyuan Ma, Haoxin Li,
Ganqu Cui, Biqing Qi, Xuekai Zhu, Xingtai Lv, Hu Jinfang, Zhiyuan Liu, and Bowen Zhou.
Ultramedical: Building specialized generalists in biomedicine, 2024.

Tianyu Zheng, Ge Zhang, Tianhao Shen, Xueling Liu, Bill Yuchen Lin, Jie Fu, Wenhu Chen, and
Xiang Yue. Opencodeinterpreter: Integrating code generation with execution and refinement.
arXiv preprint arXiv:2402.14658, 2024.

Zhanhui Zhou, Zhixuan Liu, Jie Liu, Zhichen Dong, Chao Yang, and Yu Qiao. Weak-to-strong
search: Align large language models via searching over small language models. arXiv preprint
arXiv:2405.19262, 2024.

Brian D. Ziebart, Andrew L. Maas, J. Andrew Bagnell, and Anind K. Dey. Maximum entropy
In Dieter Fox and Carla P. Gomes (eds.), Proceedings of the
inverse reinforcement learning.
Twenty-Third AAAI Conference on Artificial Intelligence, AAAI 2008, Chicago, Illinois, USA, July
13-17, 2008, pp. 1433–1438. AAAI Press, 2008. URL http://www.aaai.org/Library/
AAAI/2008/aaai08-227.php.

15

Preprint

A LIMITATIONS

Due to resource constraints, we only conducted experiments on models up to 32B. Besides the
main experiments of PRIME, we ran fewer steps for other ablation experiments, while we conduct
comparison under the same step number for fairness.

B ADDITIONAL RESULTS

B.1 REFERENCE MODEL CHOICE IS FLEXIBLE

(a) Policy ref: We use the policy logprob as πref
for PRM.

(b) SFT ref: We retain the initial policy to provide πref for
PRM and KL.

Figure 9: Comparison of different reference policy implementations. One uses the running policy’s
old logprobs as reference (policy ref) while the other uses the initial SFT model as the reference
model (SFT ref).

We implement two variants of our algorithms
to explore the effect of reference model of im-
plicit PRM, one using the initial SFT model as
the reference model (SFT ref) while the other
using the running policy’s old logprobs as ref-
erence (policy ref), as shown in Figure 9a. The
policy ref simply adopts the old logprob of the
policy model as πref, while the SFT ref remains
the initial SFT model for an additional πref cal-
culation. We compare their performance in this
section.

From the training rewards in Figure 10, we find
the two strategies are close and have pros and
cons in different aspects: The Q value calcu-
lated by implicit PRM is the expectation under
the distribution of the reference model. So the
updating policy could natrually serve as the ref-
erence. On the other hand, KL divergence cal-
culation is only allowed when the initial SFT
model is retained.

Figure 10: Different reference model for PRM.
We compare two reference model selection strate-
gies for PRIME. Using the policy model as refer-
ence and using the initial SFT model as reference.
Their rewards are similar.

B.2 SINGLE-FORWARD V.S. DOUBLE-FORWARD

Since our implicit PRM is concurrently updated in training, for each rollout stage, we can update the
PRM before the policy model and use the updated PRM to re-calculate the process rewards, which

16

050100150200250300350Steps0.340.360.380.400.420.440.460.480.50Outcome Training RewardsPolicy refSFT refPreprint

(a) PRM classification accuracy on training samples.

(b) Training outcome rewards.

Figure 11: Single and double forward. While double forward methods obtain higher accuracy after
online update, the two variants achieve similar rewards during training.

Table 4: Testset results of different RL algorithms.

Method

Step AIME 2024 AMC MATH-500 MinervaMath OlympiadBench LeetCode LiveCodeBench Avg.

RLOO
RLOO w/ PRIME

240
240

REINFORCE
240
REINFORCE w/ PRIME 240

GRPO
GRPO w/ PRIME

PPO
PRIME as Value Model
PPO w/ PRIME

240
240

240
240
240

20.0
20.0

6.7
6.7

10.0
16.7

10.0
16.7
13.3

47.0
50.6

47.0
50.0

44.6
47.0

41.0
44.6
50.6

73.2
78.2

72.6
76.4

73.2
75.0

73.6
72.6
77.4

36.4
39.3

36.0
36.8

37.5
34.9

36.0
34.6
37.1

35.4
40.3

37.2
39.1

36.6
38.2

36.3
35.7
40.6

28.3
31.1

27.2
27.8

25.0
28.9

28.3
27.8
30.0

26.7
27.5

25.0
27.5

25.8
23.9

25.7
24.6
26.7

36.9
41.0

36.0
37.8

36.1
37.8

35.8
36.6
39.4

we call the double-forward setting. We investigate the impact of double-forward in both the training
and test phases. Our default setting applies single-forward, which uses process rewards from old
PRMs. We plot PRM accuracy on rollouts and training rewards in Figure 11.

Accordingly, we find that double-forward could increase PRM accuracy, but the training rewards
remain close between the two methods.

B.3 RESULTS OF DIFFERENT RL ALGORITHMS

Figure 12: Overall math performance. Eurus-2-7B-PRIME excels at competition-level mathematics
benchmarks, outperforming advanced math models and larger models. Notably, PRIME brings
substantial performance gain (+16.7%) over Eurus-2-7B-SFT.

17

050100150200250300350Steps0.450.500.550.600.650.700.750.80PRM AccuracyBefore-double forwardBefore-single forwardAfter-double forward050100150200250300350Steps0.360.380.400.420.440.460.480.500.52Outcome Training RewardsDouble forwardSingle forwardAIME 2024AMCMinerva MathOlympiadBenchMATH-500Average01020304050607080Accuracy (%)26.757.838.642.179.248.93.330.132.729.865.132.213.350.634.640.779.843.816.730.135.331.964.635.79.345.836.843.376.443.3Eurus-2-7B-PRIMEEurus-2-7B-SFTQwen-2.5-Math-7B-InstructLlama-3.1-70B-InstructGPT-4o-2024-08-06Preprint

We ablate PRIME and different
RL algorithms with their vari-
ants and find that the PRIME al-
gorithm achieves the best per-
formance for several reasons.

Table 3: The comparison of resource requirements between
Eurus-2-7B-PRIME and Qwen2.5-Math-7B-Instruct.

Model

Eurus-2-7B-PRIME

Qwen2.5-Math-7B-Instruct

Base Model Qwen2.5-Math-7B
230K (open-source)
SFT Data
RM Data
0
Eurus-2-7B-SFT
RM
150K queries × 4 samples
RL Data

First of all, We compare dif-
ferent REINFORCE-like advan-
tage estimators including RE-
INFORCE, GRPO, and RLOO,
toggling the existence of im-
plicit process reward. To make different algorithms compatible with the compound of outcome
verifier reward and process reward, we accordingly make adaptions similar to Eq. 5. For GRPO, we
have

Qwen2.5-Math-7B
2.5M (open-source & in-house)
618K (in-house)
Qwen2.5-Math-RM (72B)
66K queries × 32 samples

Ai

t =

ro

(cid:124)

(cid:0)yj(cid:1))

(cid:0)yi(cid:1) − mean(ro
std(ro (yj))
(cid:123)(cid:122)
GRPO with outcome rewards

(cid:125)

+

|yi|
(cid:88)

s=t

(cid:124)



γs−t ·





rϕ(yi

s) − mean

(cid:16) rϕ(yj )
|yj |

std

(cid:19)



(cid:18) rϕ(yj)
|yj |
(cid:17)

(cid:123)(cid:122)
GRPO with implicit process rewards

(7)

.





(cid:125)

For REINFORCE, we have

Ai

t =

(cid:0)yi(cid:1)
ro
(cid:124) (cid:123)(cid:122) (cid:125)
REINFORCE with outcome rewards

+

|yi|
(cid:88)

γs−t · rϕ(yi
s)

.

(8)

s=t
(cid:124)

(cid:123)(cid:122)
REINFORCE with implicit process rewards

(cid:125)

As shown in Table 4, PRIME contributes consistently regardless of the policy update method, mak-
ing it a generic algorithm.

Moreover, the PPO variant of PRIME provides no performance gain, demonstrating that the addi-
tional computation cost from the critic model is redundant. This makes it possible to compensate
for the expense of the process reward model by using REINFORCE-like algorithms with simpler
advantage estimators.

Finally, we choose the best-performing RLOO as the advantage estimator in our algorithm.

B.4

“ZERO” EXPERIMENTS

(a) Outcome training rewards (10-step mov-
ing).

(b) Math test accuracy across different gradient steps.

Figure 13: “Zero” RL from Qwen2.5-Math-7B. RL from the base model converges way faster
than the SFT model, surpassing the instruct version within 32 steps.

DeepSeek-AI et al. (2025) proposed DeepSeek-R1-Zero, which is directly trained from a base model
with reinforcement learning. To further investigate the “Zero” setting, we also perform RL from
Qwen2.5-Math-7B-Base and Qwen2.5-32B-Base (Yang et al., 2024a), skipping the SFT phase. We
present the experimental results in Figure 13 and Figure 14. The observations are as follows:

18

050100150200Steps0.3500.3750.4000.4250.4500.4750.5000.525Outcome Training RewardsPRIME-ZeroPRIME0326496128160192224Steps323436384042444648Avg. Test AccPRIME-ZeroPRIMEQwen2.5-Math-7B-InstructPreprint

(a) Outcome training rewards (10-step moving).

(b) Math test accuracy across different gradient steps.

Figure 14: “Zero” RL from Qwen2.5-32B-Base. RL from a 32B base model shows more promis-
ing gain, surpassing the instruct version within 16 steps.

Table 5: Performance comparison of different reward models (Qwen2.5-7B-Base as policy model).

Reward Model AIME 24 AIME 25 AMC MATH Minerva OlympiadBench Average

Qwen2.5-3B

Qwen2.5-7B

Qwen2.5-14B

10.7

13.2

10.8

4.8

6.4

4.8

44.0

42.9

44.1

73.2

73.4

73.2

26.1

26.5

25.4

33.0

33.1

32.7

32.0

32.6

31.8

(1) RL from base model is suprisingly efficient and effective. Comparing PRIME from Qwen2.5-
Math-7B and Eurus-2-7B-SFT, the “Zero” setting converges much faster. This indicates that directly
performing RL from a base model might be a strong alternative to the conventional SFT-RL pipeline.

(2) Larger models benefit more. Comparing 7B and 32B models, we see that the 32B model
gains more on both training rewards and test performance. This is aligned with the conclusion in
DeepSeek-AI et al. (2025).

(3) Saturation could be a potential issue. Although PRIME-Zero obtains impressive performance
gain, we find it quickly saturated at a very early stage (about 50 steps), which hinders further im-
provement like in DeepSeek-AI et al. (2025). This is possibly attributed to the decrease of response
diversity, and we leave this as future work.

B.5 EFFECT OF REWARD MODEL SIZE

In the main experiments, we set the reward model to be of the same size as the policy model by
default. To further investigate the influence of reward model capacity, we conduct comparative
experiments by fixing the policy model as Qwen2.5-7B-Base and varying the reward model among
Qwen2.5-3B-Base, Qwen2.5-7B-Base, and Qwen2.5-14B-Base, other settings are aligned with the
main experiment. The results are summarized in Table 5. Overall, the results suggest that reward
model size has limited influence: the 7B reward model achieves the best average performance, while
larger (14B) or smaller (3B) reward models do not yield clear advantages.

B.6 COMPARISON WITH VINEPPO

VinePPO (Kazemnejad et al., 2024) uses average return across trajectories to estimate the value
within policy gradient update, instead of using value model in algorithm like PPO. We adopt the
same setting as VinePPO, using RhoMath 1.1B (Lin et al., 2024) as the base model and MATH
dataset for training. We reproduced PRIME and VinePPO with 8 A800-80G for 96 steps with the
same hyperparameters. It reveals that

(1) PRIME is 11x more efficient than VinePPO. As shown in Figure 15, VinePPO takes 13.94
hours for training, while PRIME only needs 1.22 hours.

19

020406080Steps0.250.300.350.400.450.500.550.60Outcome Training RewardsPRIME-Zero0163248648096Steps424446485052Avg. Test AccQwen2.5-32B-InstructPRIME-ZeroPreprint

Table 6: Comparison between PRIME and VinePPO.

Steps

VinePPO Val Acc (%)

VinePPO Clock Time (Hours)

PRIME Val Acc (%)

PRIME Clock Time (Hours)

16

15.7

2.23

16.4

0.22

32

16.3

4.57

16.8

0.41

48

17.2

7.23

17.5

0.60

64

17.6

9.86

18.1

0.80

80

96

17.7

18.4

11.96

13.94

18.7

1.01

18.8

1.22

Figure 15: Validation accuracy curves of
PRIME and VinePPO.

Figure 16: Training reward curves of PRIME
and DeepScaleR.

(2) PRIME also consistently outperforms VinePPO on the validation set. As shown in the
Table 6, the validation accuracy of PRIME is higher than VinePPO at each validation step.

B.7 COMARISON WITH DEEPSCALER

DeepScaleR (Luo et al., 2025) introduces a three-stage training pipeline, with maximum allowed
response length iteratively increased, from 8k (Stage 1) to 16k (Stage 2), finally to 24k (Stage 3).
Through this iterative context lengthening, DeepScaleR achieve continuous performance gain. We
run PRIME under the setting of DeepScaleR (base model, data, hyperparameters strictly follow the
official settings). Due to resource limits, we only conduct the first (8k) stage training for 330 steps,
yet PRIME achieves impressive results.

Figure 17: The effect of dense reward. We compared PRIME and RLOO with outcome verifier
(OV). The figure depicts training reward curves across wall clock, revealing better sample efficiency
of PRIME.

20

01234Wall Clock (Hours)15.015.516.016.517.017.518.018.519.0AccuracyVal Accuracy on MATH500 ( Pass@1 )PRIME ValVinePPO Val02004006008001000Steps0.460.480.500.520.540.560.58Training RewardsPRIMEDeepScaleR (estimated)05101520253035Wall Clock (Hours)0.360.380.400.420.440.460.48Outcome Training RewardsTraining Rewards (10-iteration Moving Average)PRIMERLOO w/ OV OnlyPreprint

(1) Model Performance. Figure 16 presents the training accuracy curves of PRIME and Deep-
ScaleR1. PRIME achieves comparable training accuracy within 330 steps, which is only 1/3 the
steps of DeepScaleR stage 1 (1040 steps). On testsets, as shown in Table 7, PRIME consistently
improves the performance of DeepSeek-R1-Distill-Qwen-1.5B by 3.1 points. This validates the
effectiveness of PRIME on highly capable base models.

(2) Efficiency. PRIME consumed 446.7 A800 GPU hours for this experiment. In contrast, Deep-
ScaleR consumed 3800 A100 GPU hours in total, and the first stage roughly required ∼600 GPU
hours. This means PRIME is also 25% faster than DeepScaleR. Note that the advantage could be
higher considering hardware differences (A800/A100).

(3) Computation Overhead. Moreover, due to the long response length of the distill model, the
overhead would be attributed more to the generation phase, narrowing down the extra time PRIME
brings to about 18%. This means that PRIME would be more suitable for long reasoning models.

Table 7: Comparison between PRIME and DeepScaleR.

Model

Step GPU Hour AIME 2024 MATH-500 AMC MinervaMath OlympiadBench Avg.

DeepScaleR-1.5B-Preview

DeepScaleR-1.5B-Stage1

1750

1040

DeepSeek-R1-Distill-Qwen-1.5B

-

3800

∼ 600

-

PRIME-DeepScaleR-1.5B-Stage1

330

446.7

43.1

33.9

28.8

32.1

87.8

-

82.8

85.1

73.6

-

62.9

68.1

30.2

-

26.5

30.1

50.0

-

43.3

44.6

57.0

-

48.9

52.0

C DISCUSSION ON IMPLICIT PROCESS REWARD

C.1 FORMULATION VALIDITY

As shown in Yuan et al. (2024b), implicit process reward is a parameterization of reward model-
ing. With such parameterization, the expectation of cumulative reward starting from the yt (i.e.
ϕ(y<t, yt) = β log Eπref(y|y≤t)e
q-value), qt
ϕ(y<t, yt) =
(cid:80)t
πref(yi|y<i) . Therefore, despite the similarities between implicit process reward formu-
lation to those of DPO (Rafailov et al., 2023; 2024), it is not derived from the optimal policy of
entropy-regularized RL (Ziebart et al., 2008; Haarnoja et al., 2017).

β rϕ(y) would have closed-formed solution qt

i=1 β log πϕ(yi|y<i)

1

C.2 LOSS FUNCTION

Since πϕ and πref are language models which are inherently self-normalized, using cross-entropy
loss brings about a minor issue that the minimum loss 0 cannot be reached. Therefore, the optimal
π∗
ϕ(y)
πref(y) = ro. This discrepancy would
solution would satisfy β log
(cid:0)yj(cid:1), rather than the
(cid:0)yi(cid:1) − 1
not affect RL, because PRIME uses relative reward, rϕ
original reward from PRM. This means that even if the bias term is included, it would be canceled
out in calculation since the bias term is only related with prompt x.

π∗
ϕ(y)
πref(y) = ro + c rather than β log

j̸=i rϕ

K−1

(cid:80)

To solve this issue, we can simply eliminate this value term by using the DPO loss. We have a pilot
experiment comparing DPO and CE loss, as shown in Table 8 and Figure 18. DPO and CE achieve
similar results, and we chose CE for memory efficiency.

Table 8: Test accuracy of updating PRM with CE or DPO loss after training.

Method

Step AIME 2024 AMC MATH-500 MinervaMath OlympiadBench Avg.

PRIME w. DPO loss

PRIME w. CE loss

96

96

7.7

7.9

39.3

40.2

66.2

66.0

17.3

16.9

31.3

30.7

32.4

32.3

1Note that the provided training logs are broken according to this issue, so we estimated its training curve

from the figure in the blog.

21

Preprint

(a) Outcome training rewards.

(b) Math test accuracy across different gradient steps.

Figure 18: Outcome rewards and test accuracy of updating PRM with DPO or CE loss during
training.

Table 9: Actions in action-centric chain-of-thought reasoning framework.

Action Name

Description

ASSESS
Analyze current situation, identify key elements and goals
ADVANCE
Move forward with reasoning - calculate, conclude, or form hypothesis
VERIFY
Check accuracy of current approach, look for errors
SIMPLIFY
Break complex problems into simpler parts
SYNTHESIZE Combine multiple pieces of information into complete solution
PIVOT
OUTPUT

Change strategy when current approach isn’t working
Summarize thought process and present final answer

C.3 REWARD SHAPING

Another view of PRIME is potential-based reward shaping (Ng et al., 1999). If we view the q-
value as the potential function, the process reward exactly satisfies the definition of shaping reward
(γ = 1). PBRS does not affect the optimal policy, but does speed up learning, which is aligned with
our results.

D SFT DATA & TRAINING DETAILS

We first performed supervised finetuning for RL preparation. We focus on mathematical and coding
problems in this paper. For models, we start with Qwen2.5-Math-7B-Base (Yang et al., 2024b) for
its great mathematical capabilities.

Action-centric chain-of-thought reasoning. We apply imitation learning (supervised finetuning)
as a warmup stage to teach models to learn certain reasoning patterns. To this end, we first design an
action-centric chain-of-thought reasoning framework. Table 9 shows the actions in the action-centric
chain-of-thought reasoning framework. When the model generates answers, it conducts multi-step
reasoning and chooses one of the 7 actions at each step. The response begins with the ASSESS
action and ends with the OUTPUT action.

Construction of the SFT dataset. To construct the SFT dataset, we collect reasoning instructions
from several open-source datasets.
It is noteworthy that we did not include many datasets with
ground-truth answers in SFT, even though they are of higher quality. However, we reserve them for
later RL training. The reason is that we aim to use different datasets for SFT and RL to diversify the
exploration in RL, and we consider ground-truth more essential in RL than in SFT. For completion,
we employ LLaMA-3.1-70B-Instruct to answer the instructions, with a system prompt requesting
the model to perform an action-centric chain-of-thought. Table 10 summarizes the key statistics
of the datasets used for SFT. The datasets span mathematics, coding, and biomedicine. We finally
obtain 230K SFT data and the average response length is 1390 tokens.

SFT Training. During the SFT phase, we conduct full parameter fine-tuning with a learning rate
of 1e-05, utilizing the AdamW optimizer alongside a cosine annealing learning rate schedule and a

22

020406080100Steps0.300.350.400.450.500.55Outcome Training RewardsTraining Rewards (10-iteration Moving Average)DPO lossCE loss163248648096Steps26272829303132Avg. Test AccDPO lossCE lossPreprint

Table 10: Data statistics of SFT data.

Task

Dataset

Size

Avg. Response Length

Source

Math

Coding

MathInstruct-MATH (Yue et al., 2023)
OpenMathIns-2-Aug Math (Toshniwal et al., 2024)
Numina (Li et al., 2024)
Reasoning-001 (SkunkworksAI, 2024)

Code-Feedback (Zheng et al., 2024)
Magicoder (Wei et al., 2024)
Magicoder-OSS (Wei et al., 2024)

Biomedicine UltraMedical mc (Zhang et al., 2024)

Total / Avg.

-

12715
15086
55845
29831

27663
24480
28980

35163

229763

964.01
1202.25
1331.61
1316.49

1805.16
1828.72
1850.05

891.06

1390.75

https://huggingface.co/datasets/TIGER-Lab/MathInstruct
https://huggingface.co/datasets/nvidia/OpenMathInstruct-2
https://huggingface.co/datasets/AI-MO/NuminaMath-CoT
https://huggingface.co/datasets/SkunkworksAI/reasoning-0.01

https://huggingface.co/datasets/m-a-p/Code-Feedback
https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K
https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K

https://huggingface.co/datasets/TsinghuaC3I/UltraMedical

-

warmup ratio of 0.1. The batch size was set to 96, with a fixed random seed of 42. The model was
trained on 230K datasets for 3 epochs.

SFT Results. After finetuning, the performance of our SFT model is reported in Figure 12. Com-
pared to baselines, Eurus-2-7B-SFT lags Qwen2.5-Math-7B-Instruct on all mathematics bench-
marks.

E RL DATA PREPROCESSING

E.1 RL DATA COLLECTION AND PREPROCESSING

We curate a high-quality RL training dataset of mathematics and coding problems with outcome ver-
ifiers (LaTeX answers for math and test cases for coding). For math, we source from NuminaMath-
CoT (Li et al., 2024), which contains about 860K math problems. The problems span from Chinese
high school mathematics to International Mathematical Olympiad competition questions. For cod-
ing, we source from APPS (Hendrycks et al., 2021a), CodeContests (Li et al., 2022), TACO (Li
et al., 2023), and Codeforces2. To further increase data quality, we conduct detailed cleaning and
filtering. Finally, we retain 457k math problems and 27k coding problems.

E.2 DATA FILTERING AND QUESTION-TYPE CLASSIFICATION

The preprocessing pipeline employs a systematic rule-based approach to filter and classify math-
ematical problems to create a high-quality dataset with solvable problems, appropriate difficulty
levels, and correct solutions. We exclude problems containing figures or diagrams since they require
visual processing capabilities. We also remove proof questions due to difficulties in answer verifi-
cation. Based on specific patterns, the remaining problems are classified into question-answering,
multiple-choice, or fill-in-the-blank questions. Since fill-in-the-blank questions comprise less than
400 examples compared to the much larger set of multiple-choice questions, we focus solely on
multiple-choice questions for further processing.

E.3 CONVERTING TO DIRECT QUESTION-ANSWER FORMAT

We transform multiple-choice questions into a direct question-answer format through three sequen-
tial stages: rule-based filtering, LLM-based filtering, and LLM-based formatting.

We first identify and remove questions that inherently require multiple-choice options - specifically,
those where comparing specific statements or properties is essential to the problem-solving process.
These questions cannot be meaningfully converted to a direct question-answer format. The initial
filtering employs simple rule-based pattern matching, searching for keywords like ”following” and
”statement” that typically indicate option-dependent problems.

Following the rule-based filtering, we employ Llama-3.1-8B-Instruct to perform a more nuanced
classification of the remaining questions. Our pilot study revealed that while the LLM occasionally
misclassifies questions, it tends to err on the conservative side - marking potentially convertible
questions as requiring options rather than the reverse. Given our large dataset, we accepted this
conservative approach to maintain quality.

2https://huggingface.co/datasets/MatrixStudio/Codeforces-Python-Submissions

23

Preprint

Table 11: Data statistics of EurusPRM training dataset.

Dataset

Generator Model

Num. Inst Resp/Inst

Step-level/Response-level

UltraInteract

Numina-SynMath

Numina-Olympiads

Llama-3.1-8B-Inst
Llama-3.1-8B-Base
Qwen2.5-72B-Inst
Qwen2.5-Math-7B-Base
Llama-3.1-8B-Inst
Qwen2.5-Math-7B-Base
Llama-3.1-8B-Inst
Qwen2.5-Math-7B-Base

20177
13570
4758
25713
4783
5806
2909
4739

8
8
8
8
8
8
8
8

Response-level
Response-level
Response-level
Response-level
Response-level
Response-level
Response-level
Response-level

Table 12: Avg@16 results with temperature=0.3 of PRIME and RLOO w/ outcome verifier (OV).

Method

Step AIME 2024 AMC

Eurus-2-7B-SFT

RLOO w/ OV Only

Eurus-2-7B-PRIME

Eurus-2-7B-PRIME

0

240

240

592

4.4

15.4

17.3

24.2

21.4

43.8

49.2

54.5

For questions classified as convertible, we implement a two-phase reformatting process: 1) Question
Reformatting: Removing choice indicators and restructuring the question to elicit direct answers. 2)
Solution Reformatting: Converting multiple-choice solutions into step-by-step derivations, ensuring
all final answers are presented in standard LaTeX boxed format. This systematic approach maintains
mathematical rigor while creating a standardized format suitable for downstream applications.

E.4 PROBLEM AND SOLUTION VALIDATION

The final stage involves merging all question-answer pairs and performing LLM-based comprehen-
sive validation. We identify two key aspects in validation: solvability and correctness.

We leverage state-of-the-art mathematical reasoning models, including QwQ-32B-Preview (Team,
2024) and Qwen2.5-Math-72B-Instruct (Yang et al., 2024b), employing a self-consistency approach
to determine problem solvability, and if solvable, verify the correctness of solutions provided in the
original dataset.

To enhance validation accuracy, we first analyzed sample problems to identify characteristics of
solvable and unsolvable cases and created synthetic unsolvable problems featuring missing con-
ditions or logical contradictions. Based on these samples, we developed specialized prompts to
improve the models’ ability to distinguish solvability. Each problem undergoes five independent
validation attempts, where the LLM: 1) Provides step-by-step solutions using LaTeX formatting.
2) Identifies unsolvability due to missing conditions or logical contradictions. 3) Generates com-
plete reasoning traces for solvable problems. 4) Presents final answers in standardized LaTeX boxed
format (\boxed{...}). 5) Document any impediments to solution completion.

We evaluate two key consistency measures across multiple validation attempts: 1) Status Consis-
tency: agreement on problem solvability. 2) Answer Consistency: consistency of solutions across
different attempts and agreement between generated solutions and ground truth. The final dataset
retains only problems that demonstrate consistent solvability across validation attempts, agreement
in solutions across multiple attempts, and alignment with ground truth answers. This rigorous vali-
dation process ensures the resulting dataset comprises well-defined, solvable problems with verified,
accurate solutions.

E.5 PRM DATA

The dataset statistics of training EurusPRM are shown in Table 11.

24

