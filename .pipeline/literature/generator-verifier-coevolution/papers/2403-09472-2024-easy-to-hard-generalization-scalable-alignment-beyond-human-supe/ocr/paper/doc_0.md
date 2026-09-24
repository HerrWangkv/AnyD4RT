4
2
0
2

c
e
D
0
1

]

G
L
.
s
c
[

2
v
2
7
4
9
0
.
3
0
4
2
:
v
i
X
r
a

Easy-to-Hard Generalization:
Scalable Alignment Beyond Human Supervision

Zhiqing Sun1∗, Longhui Yu2∗, Yikang Shen3, Weiyang Liu4,5,
Yiming Yang1†, Sean Welleck1†, Chuang Gan3,6†
1Carnegie Mellon University, 2Peking University, 3MIT-IBM Watson AI Lab
4University of Cambridge, 5Max Planck Institute for Intelligent Systems, 6UMass Amherst
Code: Edward-Sun/easy-to-hard

Abstract

Current AI alignment methodologies rely on human-provided demonstrations or
judgments, and the learned capabilities of AI systems would be upper-bounded by
human capabilities as a result. This raises a challenging research question: How can
we keep improving the systems when their capabilities have surpassed the levels of
humans? This paper answers this question in the context of tackling hard reasoning
tasks (e.g., level 4-5 MATH problems) via learning from human annotations on
easier tasks (e.g., level 1-3 MATH problems), which we term as easy-to-hard
generalization. Our key insight is that an evaluator (reward model) trained on
supervisions for easier tasks can be effectively used for scoring candidate solutions
of harder tasks and hence facilitating easy-to-hard generalization over different
levels of tasks. Based on this insight, we propose a novel approach to scalable
alignment, which firstly trains the (process-supervised) reward models on easy
problems (e.g., level 1-3), and then uses them to evaluate the performance of policy
models on hard problems. We show that such easy-to-hard generalization from
evaluators can enable easy-to-hard generalizations in generators either through
re-ranking or reinforcement learning (RL). Notably, our process-supervised 7b
RL model and 34b model (reranking@1024) achieves an accuracy of 34.0% and
52.5% on MATH500, respectively, despite only using human supervision on easy
problems. Our approach suggests a promising path toward AI systems that advance
beyond the frontier of human supervision.

1

Introduction

Rapid advancements in large language models (LLMs) indicate that in the near future, highly
sophisticated AI systems could surpass human capabilities in certain areas, significantly enhancing
our capabilities in solving harder problems beyond the levels we can currently solve [47, 49]. Since
the current AI alignment methods mostly rely on either supervised fine-tuning (SFT) with human-
provided demonstrations [59, 78, 14] or reinforcement learning from human feedback (RLHF)
[97, 68, 50], their capabilities would be inherently limited as humans cannot always provide helpful
demonstrations or supervision on the hard tasks beyond their expertise [64].

In order to build future AI systems for tackling complex challenges, such as advancing scientific
knowledge, it is crucial to develop new approaches for scalable oversight challenge, i.e., to supervise
the AI systems that can potentially outperform humans in most skills [9]. The key question is:

• Can we limit human supervision to easier tasks, yet enable the model to excel in harder tasks?

*Equal contributions as leading authors.
†Equal contributions as senior authors.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).

 
 
 
 
 
 
Figure 1: Illustration of different alignment scenarios: traditional alignment relies on human demon-
strations or judgements [50]; scalable alignment [9] assumes that humans cannot reliably supervise
smarter-than-human models; weak-to-strong generalization [11] focuses on using weak models
with unreliable labels to supervise strong models; Our proposed easier-to-general generalization
focuses on the transfer of rewarding policies from weak models to harder tasks.

We refer to this scenario as Easy-to-Hard Generalization [63, 95, 11, 29]. This setting requires no
human supervision on the harder tasks, which differs from existing work that either enhances humans’
ability to verify the outputs of AI systems [81, 60, 9, 57] or enables weak-to-strong generalization
via a teacher that only offers unreliable or noisy supervision [11].

The most basic form of easy-to-hard generalization can be achieved by training the policy models
(i.e., generator) using supervised fine-tuning (SFT) or in-context learning (ICL) on easy tasks
[55, 10], and expect this will unlock the ability to perform well on hard tasks. However, it has
been observed that SFT or ICL training of generators on easy tasks often fails to generalize to hard
tasks [71, 24, 95]. We hypothesize and show that methods beyond these can enable stronger degrees
of easy-to-hard generalization. Our intuition is guided by the observation that evaluation is easier
than generation [34, 46], so an evaluator may offer a degree of easy-to-hard generalization that is
useful for improving a generator. If that is true, we can first train a verifier on easy tasks, then make
use of its generalization ability to supervise the generator on hard tasks.

Complex tasks can often be broken down into smaller steps [95] and verified by validating the
individual steps – a strategy that is commonly employed in solving mathematical problems [74, 40, 73].
Inspired by this, we train outcome-supervised and process-supervised reward models [74, 85, 75, 40]
as our easy-to-hard evaluators. The training dataset is often comprised of a set of labeled easy tasks,
each with a question and a high-quality solution1, paired with a set of unlabeled hard tasks that
are represented only by their questions. This simulates the practical setting of having numerous
problems with known solutions, as well as significant unresolved challenges, such as the Millennium
Prize Problems [12], which present challenging open problems. The pivotal aspect of easy-to-hard
generalization thus lies in how we effectively leverage the capabilities of easier-level models in
solving harder problems.

Our investigation includes to training policy and reward models on the easy (i.e., level 1-3) portion of
the PRM800K [40] dataset, and comparing the performance of majority voting with the policy model

1We assume that human supervision is of high quality on the easy tasks in general.

2

Traditional Alignment2x2=4humans supervise strong models on hard tasksScalable Alignment (Superalignment)3^3^3humans cannot reliably supervise
superhuman models on the hardest tasks3^3=?3x3=?Burns’ Analogy on Weak-to-Strong Generalization2x2=3?weak models unreliably supervise strong models
on hard tasks that humans can evaluateOur Analogy on Easy-to-Hard Generalization1+1=2humans reliably supervise strong models
on easy tasks and evaluate them on hard tasks3x3=?3x3=?5^5=?Figure 2: We first train the evaluator with process supervision or outcome supervision (which
simulates the process supervision) to enable easy-to-hard evaluation, and then use it to facilitate
easy-to-hard generation via re-ranking or RL.

only and weighted majority voting with the policy model and PRMs (Process-supervised Reward
Models). We also introduce the Outcome & Process Reward Model (OPRM), which harnesses the
complementary strengths of outcome reward models (ORMs) and process reward models (PRMs):
judging if each step in reasoning is correct (like PRMs do) and deciding if the final answer is right
(like ORMs do). Our findings reveal a marked performance improvement with the inclusion of reward
models, especially on the hard (i.e., level 4-5) portion of the MATH500 test set. This improvement
indicates that easier-level evaluators can maintain their effectiveness on harder tasks. We have similar
observations in our experiments on the MetaMath dataset [86] and the Math-Shepherd dataset [75].

We further investigate the use of the easy-to-hard evaluator as a reward model in reinforcement
learning, where the evaluator provides targeted, step-by-step guidance in solving hard problems. We
have an intriguing finding that training with human supervision only on the easy tasks (i.e., training
with Level 1-3 problems and answers) can outperform both SFT and Final-Answer RL training on
the full dataset (Level 1-5). This finding underscores the potential of using easy-to-hard evaluation to
improve easy-to-hard generators, particularly when dealing with varied levels of task complexity.

2 Related Work

2.1 Scalable Oversight

While present-day models operate within the scope of human assessment, future, more advanced
models may engage in tasks that are beyond human evaluation capabilities. This raises a concern
that such models might prioritize objectives other than maintaining accuracy (Andreas 3, Perez et al.
53, Sharma et al. 64, Wei et al. 80). To address this, a branch of research develops techniques to
enhance the human capacity to supervise such models, such as via using AI to evaluate the work of
other AIs [1, 38, 60, 9]. Our setting differs from enhancing human oversight; instead, we focus on
enabling models to excel in hard tasks where human supervision may not be available. This also
differs from weak-to-strong generalization [11], where human supervision may be available, but not
reliable, on hard tasks. However, our framework aligns with the “sandwiching” concept proposed for
measuring progress in scalable oversight, which involves domain experts evaluating the outputs of
AI-assisted non-experts [18, 9, 57].

3

Easy-to-Hard GeneratorsEasy-to-Hard EvaluatorsEasy (train) : Domain of Human SupervisionHard (test): Domain Beyond Human SupervisionSampling SolutionsVerifying Solutionstrained on process supervisionoptimized againsteasy-to-hard evaluatorsSuccessful Generalization inEasy-to-Hard EvaluationFailed Generalization inEasy-to-Hard GenerationFigure 3: The overview diagram of our methods: the different components of modeling and training
and how they are interconnected.

2.2 Compositional Generalization

Compositional generalization is a fundamental aspect of how language works [13]. It refers to the
ability to understand and utilize novel combinations based on the understanding of basic concepts
and a limited number of their combinations [23]. Recently, least-to-most prompting [95, 20] teaches
language models how to solve a complex problem by reducing it to a series of easier sub-problems,
achieving easy-to-hard generalization on semantic parsing tasks like SCAN [37] and CFQ [35] with
perfect generalization accuracy. In addition, least-to-most prompting has also been successful in
mathematical reasoning tasks, specifically in datasets like GSM8K [16] and DROP [21], by teaching
language models to solve problems more difficult than those seen in the prompts. This success not
only underscores the capacity of language models to effectively break down complex tasks into
simpler sub-tasks Perez et al. [51], but also demonstrates their generalization capability in solving
these sub-problems.

2.3 Easy-to-Hard Generalization

Past work has evaluated easy-to-hard generalization by training easy-to-hard generators on easy
tasks using supervised finetune-tuning (SFT) or in-context learning (ICL) [55, 10]. Nevertheless,
Swayamdipta et al. [71] showed that the BERT model performs poorly on common-sense reasoning
when only trained on easy data. Fu et al. [24] showed similar results for ICL on reasoning tasks
like GSM8K [17]. In concurrent work, Hase et al. [29] evaluate the performance of easy-to-hard
generators on more datasets and models, and find that ICL or SFT on easy tasks is a strong baseline
for multiple-choice tasks like ARC [15] and MMLU [30]. In contrast, we evaluate the easy-to-hard
generation performance on the more challenging MATH dataset [32], and show that easy-to-hard
evaluation can improve a generator’s easy-to-hard generalization beyond ICL and SFT. Iterative
machine teaching [43] gives theoretical justification to show that training classifiers from easy to hard
examples yield better generalization.

3 Methodology

We study the easy-to-hard generalization problem: how can we enable capabilities beyond human su-
pervision? Specifically, we explore the efficacy and scalability of various easy-to-hard methodologies
on competition-level mathematical problem-solving problems (MATH; Hendrycks et al. 32). This
dataset is suitable for our study since it explicitly categorizes problems across five difficulty levels.
We consider levels 1-3 as “easy” tasks, encompassing both the problems and their respective solution
demonstrations, along with the correct answers. Conversely, levels 4-5, characterized by their more
complex nature, are treated as “hard” tasks and are represented solely by their questions. The MATH

4

MATH-Easy (Level 1-3)Problems and Final AnswersMATH-Hard (Level 4-5)Problemsand Final Answers*BaseLanguage ModelEasy-to-HardSFT ModelIn-Context Learning(ICL) ModelSolution SamplesMATH-Easy (Level 1-3)SolutionsMATH-Easy (Level 1-3)Solution Samples andProcess LabelsEasy-to-HardReward ModelMajority VotingReranking:Weighted Voting,Best-of-NReinforcement Learning:ReST-EM, DPO,PPOdataset’s difficulty distribution roughly follows a 1 : 2 : 2 : 3 : 3 ratio across levels 1 to 5. So our
division maintains a balanced number of easy and hard tasks.

The remainder of the paper aims to answer following research questions:

RQ1: How do generators generalize from easy to hard?

RQ2: How do evaluators generalize from easy to hard?

RQ3: If evaluators generalize better than generators, how can we take advantage of this to enable
stronger easy-to-hard generalization in generators?

3.1 Setup

Dataset MATH [32] is a dataset of 12,500 challenging competition mathematics problems, where
7,500 of them are training problems and 5,000 are originally used for testing. Following Lightman et al.
[40], Wang et al. [75], we use the identical subset of 500 representative problems (i.e., MATH500) as
our test set, uniformly sample another 500 problems for validation, across all five difficulty levels,
and leave the rest 4,000 MATH test split problems combined with the original 7,500 MATH training
split problems as our training set.

Simulated Human Demonstrations While the original MATH dataset provides full step-by-step
solutions, these solutions typically skip many chain-of-thought steps [79], which can be hard for
language models to directly imitate2. Instead, we consider filtered PRM800K [40] and MetaMATH
[86] as our SFT training data: the former is generated by a Minerva-style base GPT-4 model using
few-shot prompting after filtering the correct answers [39, 48], while the latter is generated by
ChatGPT [47]. We keep all the GSM8K data in the MetaMATH dataset since they are typically
easier than the problems in MATH. PRM800K comes with human annotated process labels, while for
MetaMath, we use Math-Shepherd as the corresponding process labels [75].

3.2 Generators

For a given dataset (e.g., a variant of MATH), we consider the following generator models:

Full & Hard ICL Full in-context learning (ICL) is a base model prompted with exemplars sampled
from all difficulty levels, or only from the level 5 [24].

Easy-to-Hard ICL This model is prompted with exemplars from easy problems. This baseline
evaluates the degree to which a model can solve problems more difficult than those seen in the
prompts [95].

Full SFT As prior work suggests that finetuning should outperform prompting alone [68, 52, 50], the
full supervised fine-tuning (SFT) model is typically considered as a ceiling that a model can achieve
on a type of task [11, 29].

Easy-to-Hard SFT This generator model is trained only on the easy tasks. Prior work suggests that
it can generalize to hard tasks but with some degeneration in performance [71].

The generator models are evaluated in greedy decoding and self-consistency (also known as majority
voting) settings [76].

3.3 Evaluators

Similarly, we consider the following evaluator models that can be trained either on the easy tasks
only, or on the full dataset. Notably, unlike final-answer rewards, reward models trained on easy tasks
can be applied to evaluate solutions to hard problems.

Final-Answer Reward is a symbolic reward that provides a binary reward based on the accuracy of
the model’s final answer. The matching is performed after normalization3.

2Hendrycks et al. [32] found that having models generate MATH-style step-by-step solutions before producing

an answer actually decreased accuracy.

3https://github.com/openai/prm800k/blob/main/prm800k/grading/grader.py

5

Table 1: Easy-to-hard generalization of generators. We compare generator performance under various
decoding settings. PRM800K and METAMATH indicate the SFT training data and ICL exemplars.
Evaluations are performed on the same MATH500 test set.

PRM800K

METAMATH

GREEDY MAJ@16 MAJ@256 GREEDY MAJ@16 MAJ@256

LLEMMA-7B

LLEMMA-34B

FULL ICL
HARD ICL
EASY-TO-HARD ICL
FULL SFT
EASY-TO-HARD SFT

FULL ICL
HARD ICL
EASY-TO-HARD ICL
FULL SFT
EASY-TO-HARD SFT

12.8
12.6
14.0
20.6
19.8

18.6
15.8
18.2
25.6
24.8

15.6
18.0
17.6
32.0
31.6

23.6
21.4
25.2
41.8
40.8

20.8
27.0
24.4
36.2
36.0

36.0
34.2
36.8
46.4
46.0

16.4
16.6
14.2
31.4
30.0

20.6
21.8
19.8
35.4
32.2

18.4
19.0
17.4
40.2
38.6

28.8
26.4
26.8
44.2
42.6

25.6
27.0
26.8
41.6
42.4

39.2
38.6
37.2
45.6
43.4

Outcome Reward Model (ORM) is trained on the Final-Answer rewards. Following Cobbe et al.
[16], Uesato et al. [74], Lightman et al. [40], we train the reward head to predict on every token
whether the solution is correct, in a similar sense to a value model [85]. At inference time, we use the
ORM’s prediction at the final token as the reward of the solution.

Process Reward Model (PRM) is trained to predict whether each step (delimited by newlines) in
the chain-of-thought reasoning path is correct. The labels are usually labeled by humans [74, 40] or
estimated with rollouts [65, 75].

Outcome & Process Reward Model (OPRM) Building on the distinct advantages of ORMs and
PRMs, we introduce the Outcome & Process Reward Model (OPRM), which harnesses the comple-
mentary strengths of both. OPRM is trained on the mixed data of ORMs and PRMs. Specifically, it
evaluates the correctness of each intermediate reasoning step, akin to PRMs, while also assesses the
overall solution’s accuracy at the final answer stage, mirroring the functionality of ORMs.

3.4 Optimizing Generators Against Evaluators

Finally, given a generator model (i.e., policy model) and a evaluator model (i.e., reward model; RM),
we optimize the generator against the evaluator using either re-ranking or reinforcement learning.

Best-of-n (BoN), also known as rejection sampling, is a reranking approach that sample multiple
solutions from the generator and selects one with the highest RM score.

Weighted Voting is similar to majority voting or self-consistency [76], but weights each solution
according to its RM score [74].

Reinforcement Learning (RL) We consider three online/offline RL variants, Reinforced Self-
Training (ReST) [28, 67], Direct Policy Optimization (DPO) [56], and Proximal Policy Optimization
(PPO) [62]. Due to the space limit, please find their detailed description in Appendix B.

3.5 Evaluation Metrics

In this study, we have chosen not to establish terms analogous to the weak-to-strong performance gap
recovery (PGR) as discussed in Burns et al. [11] or the easy-to-hard supervision gap recovery (SGR)
highlighted by Hase et al. [29]. This decision is based on our observations that sometimes, models
trained exclusively on simpler tasks—particularly when employing RL training—can outperform
those trained across the entire spectrum of problem difficulties. Therefore, we mainly focus on
the absolute and relative performance of generators (optionally optimized by the evaluator) on the
MATH500 test set [40].

3.6

Implementation Details

Base Language Model Llemma is a large language model for mathematics [6], which is continue pre-
trained from Code Llama [58] / LlaMA-2 [72]. We use both 7b and 34b variants in our experiments.

6

Figure 4: Easy-to-hard generalization of 7b (upper) and 34b (lower) evaluators. Both SFTs and RMs
are trained on the easy data. We found that PRMs trained on easy tasks can significantly improve the
re-ranking (i.e., weighted voting) performance on hard tasks. The shaded margin of the curve plot in
this paper represents the performance variance.

SFT / RL / Reward Model We fine-tune all models in full fine-tuning with frozen input-output
embedding layers and normalization layers. RMs are initialized from the base model, and have an
added scalar head to output the reward. In PPO training, we initialize the value model from the
reward model.

Hyper-parameters Due to the space limit, our training hyper-parameters can be found in Ap-
pendix. C.

4 Main Results

4.1 Easy-to-Hard Generalization of Generators

In Table 1, we compare the easy-to-hard generalization performance of the generators under various
decoding settings:

Supervised Fine-Tuning (SFT) outperforms In-Context Learning (ICL): This is consistent with
prior work [68, 50, 74]. We also find that the performance of ICL has larger variance than SFT with
respect to data ordering (or random seeds) [19, 93].

SFT data quality impacts easy-to-hard generalization: PRM800K data is generated by a base
(unaligned) GPT-4 model through few-shot prompting and is thus of lower quality than well-aligned
ChatGPT-generated MetaMATH data. We find that only MetaMath-trained models have certain
easy-to-hard gaps (e.g., 16.6 v.s. 14.2 in MetaMath-7b-ICL), while such gaps in PRM800K-trained
models are very small (less than 1%), or even inverted in the ICL setting. We hypothesize that
low-quality SFT data may only teach the model the format of the task [59, 78, 76], while high-quality
(imitation) SFT data can teach the model the principles of solving the task [70, 27]. Nevertheless, the
strongest performance is achieved by full SFT on the high-quality MetaMath data (35.4), showing an
unignorable difference, with a gap of up to 3.2, compared to its easy-to-hard SFT counterpart (32.2).

4.2 Easy-to-Hard Generalization of Evaluators

The primary metric we use to assess the effectiveness of our process reward model is not the average
accuracy of verifying each step in a solution but rather the overall performance achieved through
re-ranking methods (See discussion in Sec. 3.5). We first use re-ranking to evaluate the easy-to-hard
generalization performance of evaluators.

7

12481632641282565121024N = number of solutions per problem253035404550556065% Problems Solved27.8429.0236.1943.8849.8453.2155.3656.5857.0357.1756.7628.2335.3141.6347.0751.7255.3658.2960.6362.0762.8063.2027.8735.6641.6946.0749.2351.3852.8653.8754.6555.0355.06Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-7b (on PRM800K)Accuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem5.07.510.012.515.017.520.022.525.0% Problems Solved6.847.569.2011.6014.1415.7517.0317.9418.3818.9219.196.939.3211.4113.6915.6917.7019.2520.8922.3923.3324.126.809.4811.8413.9215.2616.5017.6618.7019.3819.9519.91Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-7b (on PRM800K)Accuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem15202530354045% Problems Solved16.8717.7822.0726.9631.0933.5635.2736.3336.7837.1237.1017.0721.6725.8429.5732.8535.6037.8639.7541.2742.0942.7416.8121.9326.0429.2031.4133.0834.4235.4336.1736.6236.66Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-7b (on PRM800K)Accuracy on All (Level 1-5) Problems12481632641282565121024N = number of solutions per problem354045505560657075% Problems Solved37.5238.9347.5956.0661.9665.6967.5268.5668.8968.8568.5137.4746.8854.2560.1464.9668.3671.0072.6473.6174.0874.3737.7946.9754.1659.8363.4065.8867.5768.1968.2468.0467.44Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-34b (on PRM800K)Accuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem5101520253035% Problems Solved10.1910.8613.6917.2520.5923.0924.7525.6026.0926.3726.4110.3413.8717.1820.2723.0425.4227.3429.1630.5031.6132.5810.3114.1117.4020.3122.3024.0424.9725.7326.1626.6827.02Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-34b (on PRM800K)Accuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem2025303540455055% Problems Solved23.2124.2229.8035.7140.2843.3445.1646.0746.5246.6246.5323.2629.5834.8339.2142.9945.8448.1749.8750.9851.8152.4523.4129.7634.8839.1041.8643.9445.2645.9746.2546.3746.31Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-34b (on PRM800K)Accuracy on All (Level 1-5) ProblemsFigure 5: Easy-to-hard generalization of evaluators applied to generators of different sizes. We
evaluated 7b generator + 34b evaluator (upper) and 34b generator + 7b evaluator (lower). Both SFTs
and RMs are trained on the easy data.

4.2.1 Re-ranking

We consider two re-ranking strategies: Best-of-n (or rejection sampling) and Weighted Voting. In our
easy-to-hard generalization setting, both SFT models and Reward Models (RMs) are trained on easier
tasks (levels 1-3), but evaluated on all difficulty levels (1-5). We compare the performance between
majority voting (SFT only) and re-ranking (SFT + OPRM) on the PRM800K dataset in Figure 4-5,
and the performance of different reward models (PRMs, ORMs, & OPRMs) on the PRM800K dataset
in Figure 8-9. Specifically, we use min as the reward aggregation function for best-of-n and prod for
weighted voting4. The figures illustrate the performance of different decoding strategies or reward
models under the same number of sampled solutions per problem. We have the following findings:

OPRMs outperforms ORMs and PRMs This confirms our hypothesis that Process Reward
Models (PRMs) and Outcome Reward Models (ORMs) capture different aspects of task-solving
processes. By integrating the strengths of both PRMs and ORMs, Outcome & Process Reward Models
(OPRMs) demonstrate superior performance. However, follow-up experiments conducted on the
MetaMath/Math-Shepherd datasets do not demonstrate significant improvements from incorporating
additional ORM training examples. This lack of enhancement may be attributed to the fact that
Math-Shepherd is already generated from final-answer rewards. This suggests that there remains
a substantial difference between process rewards labeled by humans (e.g., PRM800K) and those
generated automatically (e.g., Math-Shepherd).

Weighted voting outshines Best-of-n This finding diverges from past research where minimal
performance differences were observed between weighted voting and Best-of-n [40, 74]. Our
hypothesis is that this discrepancy arises from our specific experiment, which involves training a
less powerful base model (Llemma; Azerbayev et al. 6) on more difficult tasks (MATH; Hendrycks
et al. 32). This setup might diminish the effectiveness of the reward model, potentially leading
to an over-optimization of rewards [25]. Given these insights, weighted voting is preferred as the
primary re-ranking method for further discussions. Nevertheless, Best-of-n still achieves competitive
performance to majority voting when producing only one full solution. In Figure 5, we also find that
the 34b evaluator can significantly improve the 7b generator, while the 7b evaluator can still improve
the performance of the 34b generator.

Greater effectiveness of re-ranking on harder tasks: Weighted voting not only consistently
surpasses majority voting but also shows a more pronounced advantage on harder tasks. This

4See more detailed analysis of reward aggregation functions in Appendix. L.

8

12481632641282565121024N = number of solutions per problem25303540455055606570% Problems Solved27.8429.0236.1943.8849.8453.2155.3656.5857.0357.1756.7628.2236.1843.2049.0954.0957.7460.7763.0764.6566.0566.5427.9737.0044.0149.8054.4757.6259.9061.6862.6663.5264.38Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-34b (on PRM800K)Accuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem51015202530% Problems Solved6.847.569.2011.6014.1415.7517.0317.9418.3818.9219.196.839.5212.2214.8217.3519.4021.5023.5425.2526.8327.816.849.8312.9015.6517.9119.8921.5023.1924.4525.4226.00Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-34b (on PRM800K)Accuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem1520253035404550% Problems Solved16.8717.7822.0726.9631.0933.5635.2736.3336.7837.1237.1017.0322.2026.9731.1334.8337.6740.1642.3244.0445.4946.2716.9322.7427.7031.8635.3437.8439.7741.5042.6343.5544.27Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-34b (on PRM800K)Accuracy on All (Level 1-5) Problems12481632641282565121024N = number of solutions per problem354045505560657075% Problems Solved37.5238.9347.5956.0661.9665.6967.5268.5668.8968.8568.5137.7245.9952.5858.0462.6366.0368.5770.4271.4272.2872.6837.5046.3152.5856.8459.2960.9262.1362.8763.2763.6063.46Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-7b (on PRM800K)Accuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem5101520253035% Problems Solved10.1910.8613.6917.2520.5923.0924.7525.6026.0926.3726.4110.2313.3316.3318.9921.4423.5825.6427.4128.8030.0630.6810.3913.5916.6118.7120.1721.4422.5923.3224.0424.3324.38Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-7b (on PRM800K)Accuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem2025303540455055% Problems Solved23.2124.2229.8035.7140.2843.3445.1646.0746.5246.6246.5323.3128.8633.5837.5941.0443.7746.1247.9349.1050.1550.6723.2729.1633.6936.8638.7540.2041.4142.1142.6943.0242.96Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-7b (on PRM800K)Accuracy on All (Level 1-5) ProblemsTable 2: Comparing reinforcement learning (RL) approaches for easy-to-hard generalization. All
methods are of 7b size and evaluated with greedy decoding.

RL DATA

REWARD

ACCURACY

FINAL-ANSWER

PROCESS RM EASY (LEVEL 1-3) HARD (LEVEL 4-5) ALL

(SFT / PRM trained on level 1-3 of PRM800K)
SFT
REST-EM
ITERATIVE DPO
PPO
PPO

EASY
EASY
EASY
ALL

EASY
EASY
EASY
EASY

×
√

×
√

(SFT / PRM trained on level 1-5 of MetaMath / Math-Shepherd)
LLEMMA-BASED SFT SOTA (OURS)
PREVIOUS RL SOTA [75]

(SFT / PRM trained on level 1-3 of MetaMath / Math-Shepherd)
SFT
REST-EM
ITERATIVE DPO
ITERATIVE DPO
PPO
PPO

EASY
EASY
ALL
EASY
ALL

EASY
EASY
EASY
EASY
EASY

×
√
√

×
√

28.2
33.2
42.0
42.0
45.4

51.7
-

44.1
50.4
53.8
49.6
50.8
53.8

12.2
12.6
12.2
14.1
14.9

13.7
-

14.9
14.5
16.0
10.7
15.3
16.0

19.8
22.4
26.4
27.4
29.4

31.4
33.0

28.8
31.6
34.0
29.2
32.2
34.0

observation leads to the conclusion that evaluators demonstrate better easy-to-hard generalization
capabilities in comparison to generators. This motivates us to explore RL approaches that optimize
the generator against the evaluator to further improve the performance of easy-to-hard generation.

4.2.2 Reinforcement Learning (RL)

Given the conclusion above, an important question arises: how can evaluators once again assist
generators in achieving enhanced easy-to-hard generalization capabilities? We further investigate
the enhancement of policy models through RL, utilizing easy-to-hard evaluators as reward models.
Similar to re-ranking, SFT and PRM are only trained on easy data. For a fair comparison between
PRM800K and MetaMath, we only use vanilla PRMs in the RL training. All the RL methods use the
validation accuracy for selecting the best checkpoint5. Our comparison spans offline (ReST & DPO)
and online (PPO) RL algorithms under two training conditions:

Easy Questions & Easy Final Answers. The SFT model samples from easy questions and receives
the corresponding Final-Answer and optional PRM rewards.

All Questions & Easy Final Answers. This assumes access to a range of easy and hard problems for
RL training, with rewards for hard tasks solely provided by the easy-to-hard evaluator.

Based on the results reported in Table 2, we have the following findings:

DPO and PPO excel over ReST. Among the RL algorithms trained on the PRM800K dataset,
PPO emerges as the most effective, significantly surpassing both ReST and DPO. On the MetaMATH
dataset, PPO and DPO achieve top performance, while ReST shows only marginal improvements
over the SFT baseline. The comparative analysis between DPO and PPO across the PRM800K and
MetaMATH datasets indicates that while DPO’s efficacy is on par with PPO given a high-quality SFT
model as initialization, PPO’s effectiveness is less contingent on the quality of the underlying SFT
model [50, 56].

PRM rewards are more beneficial than Final-Answer rewards for hard tasks. Notably, models
trained with PRM rewards with human supervision on the easy tasks (achieving a top performance
of 34.0) outperform the previous state-of-the-art model trained across all task levels (33.0). This
highlights the effectiveness of leveraging easy-to-hard evaluations to improve generator performance
across varying task difficulties.

5This includes stopping iterations in ReST-EM and iterative DPO, and stopping online steps in PPO.

9

Table 3: Easy-to-hard generalization of evaluators on coding problems (APPS). Both SFTs and RMs
are trained on the easy (Introductory) data. We found that ORMs trained on easy tasks can improve
the re-ranking (Best-of-N) performance on hard (Interview & Competition) coding problems.

CODE LLAMA - 7B

CODE LLAMA - 34B

SFT / ORM
TRAIN DATA

ALL
EASY
EASY
EASY
EASY

ALL
EASY
EASY
EASY
EASY

DECODING

GREEDY
GREEDY
BEST-OF-1
BEST-OF-4
BEST-OF-16

GREEDY
GREEDY
BEST-OF-1
BEST-OF-4
BEST-OF-16

AVERAGE ACCURACY (%)

STRICT ACCURACY (%)

INTRO.

INTER.

COMP. ALL

INTRO.

INTER.

COMP. ALL

31.4
26.8
25.4
27.1
29.7

37.6
33.9
28.5
36.3
45.9

15.5
14.1
12.0
13.8
16.3

19.9
19.4
14.5
21.3
25.8

12.2
9.5
0.1
8.1
11.3

11.3
8.5
4.4
10.5
10.0

18.0
15.7
13.5
15.3
18.0

21.7
20.1
15.3
22.1
26.6

17.0
11.0
16.0
14.0
19.0

22.0
21.0
21.0
24.0
30.0

2.3
3.0
2.7
4.0
5.0

5.0
6.0
3.3
8.7
10.7

2.0
0.0
0.0
0.0
3.0

2.0
1.0
0.0
1.0
3.0

5.2
4.0
4.8
5.2
7.4

7.8
8.0
6.2
10.2
13.0

4.3 Easy-to-Hard Generalization on the Coding Domain

We conduct further experiments in the coding domain with the APPS dataset [31]. Similarly to
Lightman et al. [40], we sub-sampled 500 questions from the original test set of APPS as our test
set. Specifically, we sub-sampled 100 Introductory questions, 300 Interview questions, and 100
Competition questions, following the original distribution in the test set.

In Table 3, we compare the performance of SFT-trained Code Llama [58] (7b & 34b) with greedy
decoding and best-of-N approach. In the latter, an Outcome Reward Model (ORM) of the same
model size is trained to select the best coding one from N sampled solutions.

We found that while the reward model is only trained on the outcome supervision of easy (Introduc-
tory) data, it significantly improves the model performance on hard (Interview & Competition) data.
These findings extend the premise of easy-to-hard generalization beyond the confines of mathematical
reasoning, suggesting its applicability across diverse domains.

5 Conclusion

Our study advances the field of AI alignment by demonstrating the potential of easy-to-hard gen-
eralization, where models trained on simpler tasks can be guided to solve more complex problems
without direct human supervision on these harder tasks. Through the use of (process-supervised)
reward models for evaluating and enhancing policy models, we show that evaluators can facilitate
this form of generalization, outperforming traditional training methods. Our findings highlight the
effectiveness of re-ranking strategies and reinforcement learning (RL) in leveraging evaluators for
performance gains on difficult tasks. This approach presents a promising direction for developing AI
systems capable of surpassing human problem-solving capabilities, suggesting a scalable alignment
method that could enable AI to independently advance knowledge in complex domains.

While our study provides valuable insights into easy-to-hard generalization and the potential of
process-supervised reward models, there are limitations to consider. These include the focus on
specific model sizes and datasets, the domain specificity of reasoning tasks, and the need for further
research on the long-term implications and robustness of the method.

6 Acknowledgement

This work is supported by OpenAI Superalignment Fast Grants and Microsoft Accelerate Foundation
Models Research (AFMR) Initiative. Additionally, ZS thanks Google PhD Fellowship; SW thanks
NSF SCALE (NSF DMS 2134012) and Convergent Research.

References

[1] Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané.

Concrete problems in ai safety. arXiv preprint arXiv:1606.06565, 2016. 3

10

[2] Shengnan An, Zexiong Ma, Zeqi Lin, Nanning Zheng, Jian-Guang Lou, and Weizhu Chen.
Learning from mistakes makes llm better reasoner. arXiv preprint arXiv:2310.20689, 2023. 18

[3] Jacob Andreas. Language models as agent models. In Findings of the Association for Computa-

tional Linguistics: EMNLP 2022, pages 5769–5779, 2022. 3

[4] Thomas Anthony, Zheng Tian, and David Barber. Thinking fast and slow with deep learning

and tree search. Advances in neural information processing systems, 30, 2017. 18

[5] Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello,
Michal Valko, and Rémi Munos. A general theoretical paradigm to understand learning from
human preferences. arXiv preprint arXiv:2310.12036, 2023. 18

[6] Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer,
Albert Q Jiang, Jia Deng, Stella Biderman, and Sean Welleck. Llemma: An open language
model for mathematics. arXiv preprint arXiv:2310.10631, 2023. 6, 8, 18, 24

[7] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn
Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless
assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862,
2022. 18

[8] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones,
Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine
Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli
Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal
Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer,
Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston,
Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton,
Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben
Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan.
Constitutional ai: Harmlessness from ai feedback, 2022. 18

[9] Samuel R Bowman, Jeeyoon Hyun, Ethan Perez, Edwin Chen, Craig Pettit, Scott Heiner, Kamile
Lukosuite, Amanda Askell, Andy Jones, Anna Chen, et al. Measuring progress on scalable
oversight for large language models. arXiv preprint arXiv:2211.03540, 2022. 1, 2, 3

[10] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D. Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners. Advances in Neural Information Processing Systems, 33:1877–1901, 2020.
2, 4

[11] Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschen-
brenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, et al. Weak-to-strong gener-
alization: Eliciting strong capabilities with weak supervision. arXiv preprint arXiv:2312.09390,
2023. 2, 3, 5, 6

[12] James A Carlson, Arthur Jaffe, and Andrew Wiles. The millennium prize problems. American

Mathematical Soc., 2006. 2

[13] Noam Chomsky. On the representation of form and function. 1981. 4

[14] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan
Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu,
Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie
Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent
Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob
Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. Scaling instruction-finetuned
language models. arXiv preprint arXiv:2210.11416, 2022. 1

[15] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick,
and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning
challenge. arXiv preprint arXiv:1803.05457, 2018. 4

11

[16] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John
Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168,
2021. 4, 6, 18

[17] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to
solve math word problems. arXiv preprint arXiv:2110.14168, 2021. 4

[18] Ajeya Cotra. The case for aligning narrowly superhuman models. In AI Alignment Forum, 2021.

3

[19] Jesse Dodge, Gabriel Ilharco, Roy Schwartz, Ali Farhadi, Hannaneh Hajishirzi, and Noah Smith.
Fine-tuning pretrained language models: Weight initializations, data orders, and early stopping.
arXiv preprint arXiv:2002.06305, 2020. 7

[20] Andrew Drozdov, Nathanael Schärli, Ekin Akyürek, Nathan Scales, Xinying Song, Xinyun
Chen, Olivier Bousquet, and Denny Zhou. Compositional semantic parsing with large language
models. In The Eleventh International Conference on Learning Representations, 2022. 4

[21] Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt
Gardner. DROP: A reading comprehension benchmark requiring discrete reasoning over
paragraphs. In Proceedings of the 2019 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long
and Short Papers), pages 2368–2378, Minneapolis, Minnesota, June 2019. Association for
Computational Linguistics. doi: 10.18653/v1/N19-1246. URL https://aclanthology.
org/N19-1246. 4

[22] Yann Dubois, Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos
Guestrin, Percy Liang, and Tatsunori B Hashimoto. Alpacafarm: A simulation framework for
methods that learn from human feedback. arXiv preprint arXiv:2305.14387, 2023. 20

[23] Jerry A Fodor and Ernest Lepore. The compositionality papers. Oxford University Press, 2002.

4

[24] Yao Fu, Hao Peng, Ashish Sabharwal, Peter Clark, and Tushar Khot. Complexity-based
prompting for multi-step reasoning. In The Eleventh International Conference on Learning
Representations, 2022. 2, 4, 5, 18

[25] Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization.
In International Conference on Machine Learning, pages 10835–10866. PMLR, 2023. 8

[26] Zhibin Gou, Zhihong Shao, Yeyun Gong, Yujiu Yang, Minlie Huang, Nan Duan, Weizhu Chen,
et al. Tora: A tool-integrated reasoning agent for mathematical problem solving. arXiv preprint
arXiv:2309.17452, 2023. 18

[27] Arnav Gudibande, Eric Wallace, Charlie Snell, Xinyang Geng, Hao Liu, Pieter Abbeel, Sergey
Levine, and Dawn Song. The false promise of imitating proprietary llms. arXiv preprint
arXiv:2305.15717, 2023. 7

[28] Caglar Gulcehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, Lotte Weerts,
Abhishek Sharma, Aditya Siddhant, Alex Ahern, Miaosen Wang, Chenjie Gu, et al. Reinforced
self-training (rest) for language modeling. arXiv preprint arXiv:2308.08998, 2023. 6, 18

[29] Peter Hase, Mohit Bansal, Peter Clark, and Sarah Wiegreffe. The unreasonable effectiveness of

easy training data for hard tasks. arXiv preprint arXiv:2401.06751, 2024. 2, 4, 5, 6

[30] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and
In International

Jacob Steinhardt. Measuring massive multitask language understanding.
Conference on Learning Representations, 2020. 4

[31] Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo,
Collin Burns, Samir Puranik, Horace He, Dawn Song, et al. Measuring coding challenge
competence with apps. arXiv preprint arXiv:2105.09938, 2021. 10

12

[32] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn
Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset.
arXiv preprint arXiv:2103.03874, 2021. 4, 5, 8, 18, 24

[33] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik
Narasimhan. Swe-bench: Can language models resolve real-world github issues? arXiv preprint
arXiv:2310.06770, 2023. 18

[34] Richard M Karp. On the computational complexity of combinatorial problems. Networks, 5(1):

45–68, 1975. 2

[35] Daniel Keysers, Nathanael Schärli, Nathan Scales, Hylke Buisman, Daniel Furrer, Sergii Kashu-
bin, Nikola Momchev, Danila Sinopalnikov, Lukasz Stafiniak, Tibor Tihon, et al. Measuring
compositional generalization: A comprehensive method on realistic data. In International
Conference on Learning Representations, 2019. 4

[36] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large

language models are zero-shot reasoners. arXiv preprint arXiv:2205.11916, 2022. 18

[37] Brenden Lake and Marco Baroni. Generalization without systematicity: On the compositional
skills of sequence-to-sequence recurrent networks. In International conference on machine
learning, pages 2873–2882. PMLR, 2018. 4

[38] Jan Leike, David Krueger, Tom Everitt, Miljan Martic, Vishal Maini, and Shane Legg. Scalable
agent alignment via reward modeling: a research direction. arXiv preprint arXiv:1811.07871,
2018. 3

[39] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay
Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving
quantitative reasoning problems with language models. arXiv preprint arXiv:2206.14858, 2022.
5, 18

[40] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan
Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. arXiv preprint
arXiv:2305.20050, 2023. 2, 5, 6, 8, 10, 18, 19, 30

[41] Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale
generation: Learning to solve and explain algebraic word problems. In Proceedings of the 55th
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
pages 158–167, 2017. 18

[42] Bingbin Liu, Sebastien Bubeck, Ronen Eldan, Janardhan Kulkarni, Yuanzhi Li, Anh Nguyen,
Rachel Ward, and Yi Zhang. Tinygsm: achieving> 80% on gsm8k with small language models.
arXiv preprint arXiv:2312.09241, 2023. 18

[43] Weiyang Liu, Bo Dai, Ahmad Humayun, Charlene Tay, Chen Yu, Linda B Smith, James M
Rehg, and Le Song. Iterative machine teaching. In International Conference on Machine
Learning, pages 2149–2158. PMLR, 2017. 4

[44] Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng,
Qingwei Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical rea-
soning for large language models via reinforced evol-instruct. arXiv preprint arXiv:2308.09583,
2023. 18

[45] Rémi Munos, Michal Valko, Daniele Calandriello, Mohammad Gheshlaghi Azar, Mark Rowland,
Zhaohan Daniel Guo, Yunhao Tang, Matthieu Geist, Thomas Mesnard, Andrea Michi, et al.
Nash learning from human feedback. arXiv preprint arXiv:2312.00886, 2023. 18

[46] Moni Naor. Evaluation may be easier than generation. In Proceedings of the twenty-eighth

annual ACM symposium on Theory of computing, pages 74–83, 1996. 2

[47] OpenAI. OpenAI: Introducing ChatGPT, 2022. URL https://openai.com/blog/chatgpt.

1, 5

13

[48] OpenAI. Gpt-4 technical report, 2023. 5, 18

[49] OpenAI. OpenAI: GPT-4, 2023. URL https://openai.com/research/gpt-4. 1

[50] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L Wainwright, Pamela Mishkin,
Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to
follow instructions with human feedback. arXiv preprint arXiv:2203.02155, 2022. 1, 2, 5, 7, 9,
18, 20

[51] Ethan Perez, Patrick Lewis, Wen-tau Yih, Kyunghyun Cho, and Douwe Kiela. Unsupervised
question decomposition for question answering. In Proceedings of the 2020 Conference on
Empirical Methods in Natural Language Processing (EMNLP), pages 8864–8880, 2020. 4

[52] Ethan Perez, Douwe Kiela, and Kyunghyun Cho. True few-shot learning with language models.

Advances in neural information processing systems, 34:11054–11070, 2021. 5

[53] Ethan Perez, Sam Ringer, Kamil˙e Lukoši¯ut˙e, Karina Nguyen, Edwin Chen, Scott Heiner, Craig
Pettit, Catherine Olsson, Sandipan Kundu, Saurav Kadavath, et al. Discovering language model
behaviors with model-written evaluations. arXiv preprint arXiv:2212.09251, 2022. 3

[54] Stanislas Polu and Ilya Sutskever. Generative language modeling for automated theorem proving.

arXiv preprint arXiv:2009.03393, 2020. 18

[55] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language

understanding by generative pre-training. 2018. 2, 4

[56] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and
Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model.
arXiv preprint arXiv:2305.18290, 2023. 6, 9, 18, 20

[57] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien
Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a
benchmark. arXiv preprint arXiv:2311.12022, 2023. 2, 3

[58] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan,
Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. Code llama: Open foundation models
for code. arXiv preprint arXiv:2308.12950, 2023. 6, 10

[59] Victor Sanh, Albert Webson, Colin Raffel, Stephen Bach, Lintang Sutawika, Zaid Alyafeai,
Antoine Chaffin, Arnaud Stiegler, Arun Raja, Manan Dey, et al. Multitask prompted training
enables zero-shot task generalization. In International Conference on Learning Representations,
2021. 1, 7

[60] William Saunders, Catherine Yeh, Jeff Wu, Steven Bills, Long Ouyang, Jonathan Ward, and Jan
Leike. Self-critiquing models for assisting human evaluators. arXiv preprint arXiv:2206.05802,
2022. 2, 3

[61] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-
dimensional continuous control using generalized advantage estimation. arXiv preprint
arXiv:1506.02438, 2015. 20

[62] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal

policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 6, 18

[63] Avi Schwarzschild, Eitan Borgnia, Arjun Gupta, Furong Huang, Uzi Vishkin, Micah Goldblum,
and Tom Goldstein. Can you learn an algorithm? generalizing from easy to hard problems with
recurrent networks. Advances in Neural Information Processing Systems, 34:6695–6706, 2021.
2

[64] Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R
Bowman, Newton Cheng, Esin Durmus, Zac Hatfield-Dodds, Scott R Johnston, et al. Towards
understanding sycophancy in language models. arXiv preprint arXiv:2310.13548, 2023. 1, 3

14

[65] David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driess-
che, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mas-
tering the game of Go with deep neural networks and tree search. Nature, 529(7587):484–489,
2016. 6

[66] David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur
Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of
go without human knowledge. nature, 550(7676):354–359, 2017. 18

[67] Avi Singh, John D Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Peter J Liu, James
Harrison, Jaehoon Lee, Kelvin Xu, Aaron Parisi, et al. Beyond human data: Scaling self-training
for problem-solving with language models. arXiv preprint arXiv:2312.06585, 2023. 6, 18, 19,
20

[68] Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec
Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback.
Advances in Neural Information Processing Systems, 33:3008–3021, 2020. 1, 5, 7, 18

[69] Zhiqing Sun, Yikang Shen, Hongxin Zhang, Qinhong Zhou, Zhenfang Chen, David Cox, Yiming
Yang, and Chuang Gan. Salmon: Self-alignment with principle-following reward models. arXiv
preprint arXiv:2310.05910, 2023. 18

[70] Zhiqing Sun, Yikang Shen, Qinhong Zhou, Hongxin Zhang, Zhenfang Chen, David Cox, Yiming
Yang, and Chuang Gan. Principle-driven self-alignment of language models from scratch with
minimal human supervision. arXiv preprint arXiv:2305.03047, 2023. 7

[71] Swabha Swayamdipta, Roy Schwartz, Nicholas Lourie, Yizhong Wang, Hannaneh Hajishirzi,
Noah A Smith, and Yejin Choi. Dataset cartography: Mapping and diagnosing datasets with
training dynamics. In Proceedings of the 2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 9275–9293, 2020. 2, 4, 5

[72] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei,
Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open
foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 6, 18

[73] Trieu H Trinh, Yuhuai Wu, Quoc V Le, He He, and Thang Luong. Solving olympiad geometry

without human demonstrations. Nature, 625(7995):476–482, 2024. 2

[74] Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang,
Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with
process- and outcome-based feedback. arXiv preprint arXiv:2211.14275, 2022. 2, 6, 7, 8, 18,
29

[75] Peiyi Wang, Lei Li, Zhihong Shao, RX Xu, Damai Dai, Yifei Li, Deli Chen, Y Wu, and Zhifang
Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. CoRR,
abs/2312.08935, 2023. 2, 3, 5, 6, 9, 19, 20, 26

[76] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi,
and Hannaneh Hajishirzi. Self-instruct: Aligning language model with self generated instruc-
tions. arXiv preprint arXiv:2212.10560, 2022. 5, 6, 7

[77] Zihan Wang, Yunxuan Li, Yuexin Wu, Liangchen Luo, Le Hou, Hongkun Yu, and Jingbo Shang.
Multi-step problem solving through a verifier: An empirical analysis on model-induced process
supervision. arXiv preprint arXiv:2402.02658, 2024. 30

[78] Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan
Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In
International Conference on Learning Representations, 2021. 1, 7

[79] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny
Zhou. Chain-of-thought prompting elicits reasoning in large language models. NeurIPS, 2022.
5, 18

15

[80] Jerry Wei, Da Huang, Yifeng Lu, Denny Zhou, and Quoc V Le. Simple synthetic data reduces

sycophancy in large language models. arXiv preprint arXiv:2308.03958, 2023. 3

[81] Jeff Wu, Long Ouyang, Daniel M Ziegler, Nisan Stiennon, Ryan Lowe, Jan Leike, and
Paul Christiano. Recursively summarizing books with human feedback. arXiv preprint
arXiv:2109.10862, 2021. 2

[82] Jing Xu, Andrew Lee, Sainbayar Sukhbaatar, and Jason Weston. Some things are more
cringe than others: Preference optimization with the pairwise cringe loss. arXiv preprint
arXiv:2312.16682, 2023. 18

[83] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao.
React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629,
2022. 18

[84] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik
Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. arXiv
preprint arXiv:2305.10601, 2023. 18

[85] Fei Yu, Anningzhe Gao, and Benyou Wang. Outcome-supervised verifiers for planning in

mathematical reasoning. arXiv preprint arXiv:2311.09724, 2023. 2, 6

[86] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok,
Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical
questions for large language models. arXiv preprint arXiv:2309.12284, 2023. 3, 5, 18, 19, 20,
26

[87] Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Chuanqi Tan, and Chang Zhou.
Scaling relationship on learning mathematical reasoning with large language models. arXiv
preprint arXiv:2308.01825, 2023. 18

[88] Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu
Chen. Mammoth: Building math generalist models through hybrid instruction tuning. arXiv
preprint arXiv:2309.05653, 2023. 18

[89] Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. Star: Bootstrapping reasoning with
reasoning. Advances in Neural Information Processing Systems, 35:15476–15488, 2022. 18, 19

[90] Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. Automatic chain of thought prompting

in large language models. arXiv preprint arXiv:2210.03493, 2022. 18

[91] Yao Zhao, Mikhail Khalman, Rishabh Joshi, Shashi Narayan, Mohammad Saleh, and Peter J
Liu. Calibrating sequence likelihood improves conditional language generation. In The Eleventh
International Conference on Learning Representations, 2022. 18

[92] Yao Zhao, Rishabh Joshi, Tianqi Liu, Misha Khalman, Mohammad Saleh, and Peter J Liu. Slic-
hf: Sequence likelihood calibration with human feedback. arXiv preprint arXiv:2305.10425,
2023. 18

[93] Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. Calibrate before use:
Improving few-shot performance of language models. In International Conference on Machine
Learning, pages 12697–12706. PMLR, 2021. 7

[94] Chuanyang Zheng, Zhengying Liu, Enze Xie, Zhenguo Li, and Yu Li. Progressive-hint prompt-
ing improves reasoning in large language models. arXiv preprint arXiv:2304.09797, 2023.
18

[95] Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale
Schuurmans, Claire Cui, Olivier Bousquet, Quoc V Le, et al. Least-to-most prompting enables
complex reasoning in large language models. In The Eleventh International Conference on
Learning Representations, 2022. 2, 4, 5, 18

[96] Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan,
and Jimmy Ba. Large language models are human-level prompt engineers. arXiv preprint
arXiv:2211.01910, 2022. 18

16

[97] Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei,
Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences.
arXiv preprint arXiv:1909.08593, 2019. 1

17

A Additional Related Work

A.1 Rationale-Augmented (Mathematical) Reasoning

Ling et al. [41] pioneer the work of solving math word problems by generating step-by-step solutions
before the final answer. Cobbe et al. [16] extend this work by constructing a much larger dataset to
finetune a pre-trained large language model to solve math word problems, and a outcome-supervised
verifier to rank candidate solutions. Wei et al. [79] demonstrate that the reasoning ability of a language
model can be elicited through the use of prefixed rationales. Subsequent research [36, 83, 39, 96,
84] in tasks requiring human-level reasoning skills has also highlighted the efficacy of rationale
augmentation.

Among all the reasoning tasks, we select mathematical reasoning to evaluate easy-to-hard generaliza-
tion ability, given that mathematical reasoning serves as a valuable assessment for complex reasoning
abilities and features a clear delineation of difficulty levels. Recent research efforts focus on prompt
design [79, 95, 24, 90, 94] to elicit the intrinsic reasoning capabilities of models, or data engineering
for fine-tuning [44, 87, 88, 86, 26, 42, 2, 6], which draws on experts to provide high-quality training
datasets. Our work is categorized as fine-tuning based work. However, unlike previous work, our
focus lies in exploring how to generalize to more challenging mathematical problems when only
provided with easy mathematical data.

A.2 Outcome Reward Models & Process Reward Models

For some multi-step complex reasoning tasks, such as generating highly complex code, it may be
challenging for humans to fully grasp the outputs produced by an advanced AI system. In such
scenarios, process-supervised reward models (PRMs) present a promising solution [74, 40]. These
models operate by supervising each step in the reasoning or generation process, rather than focusing
solely on the end result. They are particularly effective in tasks where the reasoning process itself is
as important as the final outcome [32, 33].

Uesato et al. [74] find that process-supervised reward models (PRMs) achieve better performance
than outcome-supervised reward models (ORMs) when re-ranking sampled solutions from the policy
model, but their performance is similar during reinforcement learning (RL) via expert iteration
[66, 4, 54, 89, 28, 67]. Lightman et al. [40] compare ORMs and PRMs with a more capable base
model [48] and significantly more human-labeled process feedback on the more challenging MATH
dataset, and also find that PRMs significantly outperform ORMs in the reranking setting. In contrast
to these works, which only study the effectivenss of PRM in an independent and identically distributed
(IID) domain, we study the utilization of PRMs in the easy-to-hard generalization scenario, and
show that easy-to-hard evaluators instantiated by PRMs can enable easy-to-hard generation of policy
models.

B Reinforcement Learning Algorithms

Reinforced Self-Training (ReST) is an offline RL algorithm, which alternates between generating
samples from the policy, which are then used to improve the LLM policy with RM-weighted SFT
[28, 67]. Its variants include expert iteration [4] and rejection sampling fine-tuning [72, 87].

Direct Policy Optimization (DPO) is a class of offline RL algorithms [56] that consider both positive
and negative gradient updates. It fine-tunes the policy model on a preference dataset consisting of
paired positive and negative samples. The variants include NLHF [45], IPO [5], and SLiC [91, 92].
Recent work shows that iteratively applying DPO leads to improved performance [82].

Proximal Policy Optimization (PPO) is an online RL algorithm which samples from the policy
during fine-tuning [62]. It is widely used in RLHF [68, 7, 50] and RLAIF [8, 69].

18

C Hyper-parameters

C.1 Supervised Fine-Tuning & Reward Modeling

For the PRM800K dataset [40], the SFT model is trained using steps that are labeled as correct. For
the MetaMath dataset [86], given that the original dataset can contain upwards of ten solutions for
the same question, potentially leading to over-fitting, we implement a filtering process. This process
ensures that, during any given epoch, no more than three solutions per question are retained, thereby
mitigating the risk of over-fitting.

The PRMs are trained on the corresponding released dataset [40, 75]. For generating solutions to train
ORMs, we sample 32 solutions for each question from the language model using top-K sampling with
K=20 and temperature of 0.7. We also ensure that the ratio between positive and negative samples for
each question is between 1:3 to 3:1.

See Table 4 for a list of training hyper-parameters used in the training jobs. We use full fine-tuning
for all SFT/RM training.

Table 4: Hyper-parameters in our SFT/RM training jobs

PRM800K

SFT

PRM ORM OPRM

METAMATH
PRM
SFT

LEARNING RATE
EPOCHS
BATCH SIZE
MAX SEQ LEN
DTYPE

LEARNING RATE
EPOCHS
BATCH SIZE
MAX SEQ LEN
DTYPE

2E-5
3
128
768
BF16

1E-5
3
128
768
BF16

2E-5
2
128
768
BF16

1E-5
2
128
768
BF16

2E-5
2
128
1024
BF16

1E-5
2
128
1024
BF16

2E-5
2
128
1024
BF16

1E-5
2
128
1024
BF16

8E-6
3
128
1024
FP32

5E-6
3
128
768
FP32

2E-5
2
128
768
BF16

-
-
-
-
-

LLEMMA-7B

LLEMMA-34B

C.2 Re-Ranking

For majority voting, weighted voting, and best-of-n, we sample from the language model using top-K
sampling with K=20 and temperature of 0.7. At test time, we use the ORM’s prediction at the final
token as the overall score for the solution, and use the PRM’s prediction at each intermediate step
(denoted by the new line symbol) and the final token as the process reward scores.

C.3 Reinforcement Learning

We use full fine-tuning during the RL stage.

ReST-EM Following Singh et al. [67], we sample 32 solutions for each question from the language
model using top-K sampling with K=40. We also used a cut-off threshold of 10 for the maximum
number of solutions per problem [89, 67]. We performed iterative ReST training for two epochs,
and observed performance degeneration starting from the third epoch. For PRM800K, we used a
temperature of 1.0, while for MetaMath, we used a temperature of 1.2. The rest training hyper-
parameters are the same as in SFT training.

Iterative DPO We sample 8 solutions for each question from the language model using top-K
sampling with K=20 and temperature of 1.0. We use the process reward model to assign a score
between 0 and 1 to each solution, and use final-answer reward to assign an additional 0/1 score to
each solution. A preference training pair is constructed only when the score difference between
positive and negative solutions is greater than 1.0. We used a cut-off threshold of 3 for the maximum
number of preference pairs per problem.

19

Table 5: Full results of comparing reinforcement learning (RL) approaches for easy-to-hard gener-
alization. All methods are of 7b size and evaluated with greedy decoding. † indicates the model is
trained with additional final-answer labels on hard tasks (similar to Singh et al. [67]), which is not
strictly a easy-to-hard generalization setup.

RL DATA

REWARD

ACCURACY

FINAL-ANSWER

PROCESS RM EASY (LEVEL 1-3) HARD (LEVEL 4-5) ALL

(SFT / PRM trained on level 1-3 of PRM800K)
SFT
REST-EM
REST-EM
REST-EM†
ITERATIVE DPO
ITERATIVE DPO†
PPO
PPO
PPO†
PPO

EASY
HARD
ALL
EASY
ALL
EASY
HARD
ALL
ALL

EASY
HARD
ALL
EASY
ALL
EASY
HARD
ALL
EASY

×
×
×
√
√

×
×
×
√

(SFT / PRM trained on level 1-5 of MetaMath / Math-Shepherd)
LLEMMA-BASED SFT SOTA (OURS)
PREVIOUS RL SOTA [75]

(SFT / PRM trained on level 1-3 of MetaMath / Math-Shepherd)
SFT
REST-EM
ITERATIVE DPO
ITERATIVE DPO
ITERATIVE DPO†
PPO
PPO†
PPO

EASY
EASY
ALL
ALL
EASY
ALL
ALL

EASY
EASY
EASY
ALL
EASY
ALL
EASY

×
√
√
√

×
×
√

28.2
33.2
31.9
35.7
42.0
38.2
42.0
34.0
42.0
45.4

51.7
-

44.1
50.4
53.8
49.6
47.9
50.8
50.8
53.8

12.2
12.6
8.0
8.8
12.2
11.5
14.1
9.2
10.7
14.9

13.7
-

14.9
14.5
16.0
10.7
12.2
15.3
13.4
16.0

19.8
22.4
19.4
21.6
26.4
24.2
27.4
21.0
25.6
29.4

31.4
33.0

28.8
31.6
34.0
29.2
29.2
32.2
31.2
34.0

For all DPO training [56], we used a learning rate of 2 × 10−6, a batch size of 64, and a DPO training
epoch of 1. We set β = 0.1 for all DPO experiments, and performed at most 5 DPO iterations (i.e.,
sampling new solutions and performing one DPO epoch).

PPO We follow Dubois et al. [22] on the implementation of the PPO algorithm, which is a variant
of [50]6. Specifically, we normalize the advantage across the entire batch of rollouts obtained for
each PPO step and initialize the value model from the reward model.

We clipped the gradient by its Euclidean norm at a limit of 1. Our training spanned 500 PPO steps on
the RL data (MATH questions except MATH500 and our 500 validation questions). For generalized
advantage estimation (GAE; Schulman et al. [61]), both λ and γ were set at 1.

For PRM800K, we used a batch size of 512 for each PPO step. This comprised 8 epochs of gradient
steps, each having 64 rollouts. We applied a peak learning rate of 2 × 10−5 with cosine decay. We
opted for a constant KL regularizer coefficient of 0.01, and a sampling temperature of 0.7.

For MetaMath/Math-Shepherd, we used a batch size of 512 for each PPO step. This comprised 2
epochs of gradient steps, each having 256 rollouts. We applied a peak learning rate of 5 × 10−6
with cosine decay. We opted for a constant KL regularizer coefficient of 0.002, and a sampling
temperature of 1.2.

D Re-ranking Results on MetaMath

Similar to Sec. 4.2.1, we assess the effectiveness of process reward models on the MetaMath/Math-
Shepherd dataset [86, 75]. From Figure 6, we can see that PRMs are also more effective on harder
tasks when trained on MetaMath/Math-Shepherd.

6https://github.com/openai/lm-human-preferences

20

Table 6: Pass@N scores (upper bound of Best-of-N) on coding problems (APPS).

CODE LLAMA - 7B

CODE LLAMA - 34B

SFT / ORM
TRAIN DATA

EASY
EASY
EASY

EASY
EASY
EASY

DECODING

PASS @ 1
PASS @ 4
PASS @ 16

PASS @ 1
PASS @ 4
PASS @ 16

INTRO.

ACCURACY (%)
INTER.

COMP. ALL

11.0
20.0
39.0

19.0
32.0
44.0

1.6
4.7
9.3

5.3
10.7
18.3

0.0
2.0
6.0

2.0
3.0
8.0

3.2
7.2
14.6

7.3
13.4
21.4

Figure 6: Easy-to-hard generalization of evaluators on MetaMath. Both SFTs and RMs are trained on
the easy data.

21

12481632641282565121024N = number of solutions per problem4550556065% Problems Solved44.8444.8451.5256.4559.6061.7863.0863.8364.1164.2164.1744.6250.0154.7958.6361.3863.1464.5665.1765.4965.5665.6344.7550.1853.1854.7355.9256.6957.0957.3857.4357.1356.93Majority VotingWeighted Voting w/ RMBest-of-N w/ RMMetaMath-7b + Math-Shepherd-7bAccuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem1012141618202224% Problems Solved12.1112.2614.5717.2019.0620.2020.8021.2521.7221.9422.3212.1414.5016.5118.5920.3021.4522.0522.5722.7223.0923.3712.1214.3416.1217.1718.0818.6919.2919.6119.6619.6620.54Majority VotingWeighted Voting w/ RMBest-of-N w/ RMMetaMath-7b + Math-Shepherd-7bAccuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem27.530.032.535.037.540.042.5% Problems Solved27.6827.7732.1135.8738.3740.0040.9041.5341.8842.0942.1927.5731.3934.7037.6539.8241.2542.2742.7843.0143.2343.3927.6531.3833.7535.0536.1436.7937.3237.6037.6637.5237.89Majority VotingWeighted Voting w/ RMBest-of-N w/ RMMetaMath-7b + Math-Shepherd-7bAccuracy on All (Level 1-5) ProblemsE Qualitative Comparison

QUESTION

METAMATH-SFT

METAMATH-
PPO

√

x and when x = 24, y = 15. What is x when

The value of y varies inversely as
y = 3?

If y varies inversely as

x, then y = k√

√

x .

x for some constant k.
√

√

Plugging in the given values, we have 15 = k√
24

, or k = 15

24

24 = 15

4

6

.

Now we can find x when y = 3 by plugging in these values
into the equation y = k√

√

6

15
4√

√
6
x .
√
√

4

2

6.

√

√

√

√

(cid:1)2

x = 15
√

x = 15
= 5
2 .

6, or 12
√
x = 15
6
√
12
6
= 6.25 .

x, then we can write the equation as y = k√
x

x , or 3 = 15
We have 3 =
Cross-multiplying, we get 3 · 4
Dividing both sides by 12, we find
Squaring both sides, we get x = (cid:0) 5
If y varies inversely as
for some constant k.
We are given that when x = 24, y = 15, so we can substitute these values
into the equation to solve for k: 15 = k√
√
24
24 gives 15
Multiplying both sides by
Simplifying the left side gives 15 · 2
Now we can use this value of k in the equation to find x when y = 3:
3 = 30
Multiplying both sides by
Dividing both sides by 3 gives
Squaring both sides gives x = (10

6)2 = 100(6) = 600

x = 30
√
6

x gives 3
√

x = 10
√

6 = k, which further simplifies to 30

√
6√
x

24 = k

√

√

√

√

√

6

√

6 = k

Table 7: Qualitative comparison between MetaMath-7b-SFT and MetaMath-7b-PPO, as well as the
first error detected in MetaMath-SFT greedy decoding by Math-Shepherd. Examples are cherry-
picked for better illustration.

F Performance of DPO Iterations

Figure 7: Test performance as a function of DPO iterations.

22

QUESTION

PRM800K-SFT

PRM800K-PPO

What is the total surface area, in square inches, of a cube with a volume of 1
cubic foot?
The volume of a cube is given by V = s3, where s is the length of one side.
We are given that V = 1 cubic foot, so s3 = 1.
Taking the cube root of both sides, we find that s = 1.
The surface area of a cube is given by A = 6s2,
so the total surface area of this cube is A = 6(12) = 6 square inches.
# Answer
6
So the volume of a cube is s3.
And we know the volume is 1.
So s3 = 1.
So s = 1.
The surface area of a cube is 6s2.
So the surface area is 6 · 12 = 6.
So the answer is 6 .
# Answer
6

Table 8: The distribution shift before and after PPO training of a Llemma-7B model trained on
PRM800K. We believe such a big shift could explain why does PPO work exceptionally well on
PRM800K, where the policy is initialized from a weak SFT model.

23

G Re-ranking Performance Analysis of PRMs, ORMs & OPRMs

We compare the re-ranking performance of Process Reward Models (PRMs), Outcome Reward
Models (ORMs), and our proposed Outcome & Process Reward Models (OPRMs). Figure 8 shows
the results on 7b models and Figure 9 is on 34b models. We find that in our setting of Llemma [6] +
MATH [32], PRMs and ORMs perform similarly, with PRMs slightly outperforming ORMs on hard
tasks. But the OPRMs that trained on the mixed data of PRMs and ORMs significantly outperforms
both of them.

Figure 8: Comparing process reward models (PRMs, left), outcome reward models (ORMs, middle),
and outcome & process reward models (OPRMs, right) on 7b models trained on the PRM800K
dataset. Both SFTs and RMs are trained on the easy data.

24

12481632641282565121024N = number of solutions per problem253035404550556065% Problems Solved27.8429.0236.1943.8849.8453.2155.3656.5857.0357.1756.7628.0234.5940.4845.7750.4253.8156.4058.1159.1459.8260.1928.0234.2638.8542.2545.5447.6149.0649.8850.2149.8149.09Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + PRM-7b (on PRM800K)Accuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem253035404550556065% Problems Solved27.8429.0236.1943.8849.8453.2155.3656.5857.0357.1756.7628.0036.0842.8948.7053.1656.0858.0459.5560.2560.7760.8128.2136.1641.7646.0748.5150.2250.9051.1651.2950.9250.91Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + ORM-7b (on PRM800K)Accuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem253035404550556065% Problems Solved27.8429.0236.1943.8849.8453.2155.3656.5857.0357.1756.7628.2335.3141.6347.0751.7255.3658.2960.6362.0762.8063.2027.8735.6641.6946.0749.2351.3852.8653.8754.6555.0355.06Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-7b (on PRM800K)Accuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem5.07.510.012.515.017.520.022.525.0% Problems Solved6.847.569.2011.6014.1415.7517.0317.9418.3818.9219.196.908.8210.7912.9314.9516.6618.1219.6220.8921.9822.776.938.9310.3511.8212.8814.0514.7015.2815.6715.7915.66Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + PRM-7b (on PRM800K)Accuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem5.07.510.012.515.017.520.022.525.0% Problems Solved6.847.569.2011.6014.1415.7517.0317.9418.3818.9219.196.929.6211.9514.1915.8717.4118.6919.8620.5121.0121.236.889.4411.5013.0914.0614.8515.6015.9616.2516.3816.34Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + ORM-7b (on PRM800K)Accuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem5.07.510.012.515.017.520.022.525.0% Problems Solved6.847.569.2011.6014.1415.7517.0317.9418.3818.9219.196.939.3211.4113.6915.6917.7019.2520.8922.3923.3324.126.809.4811.8413.9215.2616.5017.6618.7019.3819.9519.91Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-7b (on PRM800K)Accuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem15202530354045% Problems Solved16.8717.7822.0726.9631.0933.5635.2736.3336.7837.1237.1016.9921.1024.9528.5931.8334.3236.3537.9339.0539.9340.4717.0020.9523.9526.3128.4029.9931.0531.7232.0931.9531.55Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + PRM-7b (on PRM800K)Accuracy on All (Level 1-5) Problems12481632641282565121024N = number of solutions per problem15202530354045% Problems Solved16.8717.7822.0726.9631.0933.5635.2736.3336.7837.1237.1016.9722.2326.6630.6433.6135.8037.4138.7439.4239.9039.9817.0622.1725.9128.7530.4431.6632.3832.6732.8832.8032.77Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + ORM-7b (on PRM800K)Accuracy on All (Level 1-5) Problems12481632641282565121024N = number of solutions per problem15202530354045% Problems Solved16.8717.7822.0726.9631.0933.5635.2736.3336.7837.1237.1017.0721.6725.8429.5732.8535.6037.8639.7541.2742.0942.7416.8121.9326.0429.2031.4133.0834.4235.4336.1736.6236.66Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-7b (on PRM800K)Accuracy on All (Level 1-5) ProblemsFigure 9: Comparing process reward models (PRMs, left), outcome reward models (ORMs, middle),
and outcome & process reward models (OPRMs, right) on 34b models trained on the PRM800K
dataset. Both SFTs and RMs are trained on the easy data.

25

12481632641282565121024N = number of solutions per problem354045505560657075% Problems Solved37.5238.9347.5956.0661.9665.6967.5268.5668.8968.8568.5137.6545.4452.3757.8462.7166.3168.4169.7770.5970.6970.4837.6045.2050.7455.0658.4860.8262.4063.6464.0964.5164.57Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + PRM-34b (on PRM800K)Accuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem354045505560657075% Problems Solved37.5238.9347.5956.0661.9665.6967.5268.5668.8968.8568.5137.8446.7154.4260.6665.2968.3870.2871.1871.5471.7171.5337.6546.7253.4157.5160.2562.0562.8363.5064.2164.5364.99Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + ORM-34b (on PRM800K)Accuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem354045505560657075% Problems Solved37.5238.9347.5956.0661.9665.6967.5268.5668.8968.8568.5137.4746.8854.2560.1464.9668.3671.0072.6473.6174.0874.3737.7946.9754.1659.8363.4065.8867.5768.1968.2468.0467.44Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-34b (on PRM800K)Accuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem5101520253035% Problems Solved10.1910.8613.6917.2520.5923.0924.7525.6026.0926.3726.4110.3613.3216.1418.8821.2023.4025.4927.0727.9328.7029.3810.1613.1115.5117.5218.8620.1621.3522.4023.0923.4323.67Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + PRM-34b (on PRM800K)Accuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem5101520253035% Problems Solved10.1910.8613.6917.2520.5923.0924.7525.6026.0926.3726.4110.2713.7317.0319.9022.5624.8826.6227.8729.0129.7830.1910.2313.7516.5618.5320.1721.2421.9522.4522.7423.0023.70Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + ORM-34b (on PRM800K)Accuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem5101520253035% Problems Solved10.1910.8613.6917.2520.5923.0924.7525.6026.0926.3726.4110.3413.8717.1820.2723.0425.4227.3429.1630.5031.6132.5810.3114.1117.4020.3122.3024.0424.9725.7326.1626.6827.02Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-34b (on PRM800K)Accuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem2025303540455055% Problems Solved23.2124.2229.8035.7140.2843.3445.1646.0746.5246.6246.5323.3828.5833.3737.4440.9543.8045.9847.4348.2448.7248.9123.2428.4132.2735.3937.7539.5140.8942.0242.6142.9643.12Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + PRM-34b (on PRM800K)Accuracy on All (Level 1-5) Problems12481632641282565121024N = number of solutions per problem2025303540455055% Problems Solved23.2124.2229.8035.7140.2843.3445.1646.0746.5246.6246.5323.3829.4234.8239.3042.9045.5847.4548.4949.2849.7349.8823.2629.4534.0937.1039.2440.6441.3841.9642.4542.7343.32Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + ORM-34b (on PRM800K)Accuracy on All (Level 1-5) Problems12481632641282565121024N = number of solutions per problem2025303540455055% Problems Solved23.2124.2229.8035.7140.2843.3445.1646.0746.5246.6246.5323.2629.5834.8339.2142.9945.8448.1749.8750.9851.8152.4523.4129.7634.8839.1041.8643.9445.2645.9746.2546.3746.31Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-34b (on PRM800K)Accuracy on All (Level 1-5) ProblemsH Re-ranking Results on MetaMath

Similar to Sec. 4.2.1, we assess the effectiveness of process reward models on the MetaMath/Math-
Shepherd dataset [86, 75]. From Figure 10, we can see that PRMs are also more effective on harder
tasks when trained on MetaMath/Math-Shepherd.

Figure 10: Comparing process reward models (PRMs, left, trained on Meth-Shepherd), outcome
reward models (ORMs, middle), and outcome & process reward models (OPRMs, right) on 7b models
trained on the MetaMath dataset. Both SFTs and RMs are trained on the easy data.

26

12481632641282565121024N = number of solutions per problem40455055606570% Problems Solved44.8444.8451.5256.4559.6061.7863.0863.8364.1164.2164.1744.7550.2254.6458.5361.2263.2264.5564.9865.1665.1765.2744.6950.3353.7155.9557.7958.9759.7360.3960.6260.7860.80Majority VotingWeighted Voting w/ RMBest-of-N w/ RMMetaMath-7b + Math-Shepherd-7bAccuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem40455055606570% Problems Solved44.8444.8451.5256.4559.6061.7863.0863.8364.1164.2164.1744.4149.8754.3557.9860.8462.9264.1064.8265.4065.6165.7244.7249.6652.6454.4855.5956.2656.8957.3957.8557.7757.45Majority VotingWeighted Voting w/ RMBest-of-N w/ RMMetaMath-7b + ORM-7bAccuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem40455055606570% Problems Solved44.8444.8451.5256.4559.6061.7863.0863.8364.1164.2164.1744.7650.1855.1859.2062.1164.2966.0767.1367.6767.8767.8044.7850.3553.4655.6156.4657.2057.2156.9257.0656.9256.58Majority VotingWeighted Voting w/ RMBest-of-N w/ RMMetaMath-7b + Math-Shepherd-OPRM-7bAccuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem1012141618202224% Problems Solved12.1112.2614.5717.2019.0620.2020.8021.2521.7221.9422.3212.0814.3016.6818.5520.3821.4122.1122.5122.7222.8823.0911.9814.4816.0917.3017.9218.2818.6418.8018.8218.3317.54Majority VotingWeighted Voting w/ RMBest-of-N w/ RMMetaMath-7b + Math-Shepherd-7bAccuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem1012141618202224% Problems Solved12.1112.2614.5717.2019.0620.2020.8021.2521.7221.9422.3212.0514.3716.3718.0719.5820.7621.2521.5921.6021.7121.7711.9514.2515.9117.3118.5519.0819.2319.3319.4319.5319.40Majority VotingWeighted Voting w/ RMBest-of-N w/ RMMetaMath-7b + ORM-7bAccuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem1012141618202224% Problems Solved12.1112.2614.5717.2019.0620.2020.8021.2521.7221.9422.3212.1714.4216.4218.3019.8320.9121.5221.8322.0522.2022.3112.0214.3916.0317.0717.6217.8517.9118.3219.0419.7319.48Majority VotingWeighted Voting w/ RMBest-of-N w/ RMMetaMath-7b + Math-Shepherd-OPRM-7bAccuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem25.027.530.032.535.037.540.042.545.0% Problems Solved27.6827.7732.1135.8738.3740.0040.9041.5341.8842.0942.1927.6331.3834.7437.5939.8141.3242.2942.6742.8942.9343.0827.5731.5233.9435.6936.8937.6238.2038.6038.7338.4938.10Majority VotingWeighted Voting w/ RMBest-of-N w/ RMMetaMath-7b + Math-Shepherd-7bAccuracy on All (Level 1-5) Problems12481632641282565121024N = number of solutions per problem25.027.530.032.535.037.540.042.545.0% Problems Solved27.6827.7732.1135.8738.3740.0040.9041.5341.8842.0942.1927.4631.2534.4737.0739.2240.8741.6642.1442.4142.5642.6327.5231.1233.3835.0036.1836.7537.1737.4437.7137.7537.50Majority VotingWeighted Voting w/ RMBest-of-N w/ RMMetaMath-7b + ORM-7bAccuracy on All (Level 1-5) Problems12481632641282565121024N = number of solutions per problem25.027.530.032.535.037.540.042.545.0% Problems Solved27.6827.7732.1135.8738.3740.0040.9041.5341.8842.0942.1927.6631.4434.8337.7539.9841.5442.6943.4043.7643.9543.9027.6031.4633.8235.4136.1136.5836.6336.7137.1637.4337.16Majority VotingWeighted Voting w/ RMBest-of-N w/ RMMetaMath-7b + Math-Shepherd-OPRM-7bAccuracy on All (Level 1-5) ProblemsI More Comparisons

Figure 11: Comparing different reward models with Weighted Voting (upper) and Best-of-N (lower)
on 7b models trained on the PRM800K dataset. Both SFTs and RMs are trained on the easy data.

Figure 12: Comparing different reward models with Weighted Voting (upper) and Best-of-N (lower)
on 34b models trained on the PRM800K dataset. Both SFTs and RMs are trained on the easy data.

27

12481632641282565121024N = number of solutions per problem253035404550556065% Problems Solved28.0234.5940.4845.7750.4253.8156.4058.1159.1459.8260.1928.0036.0842.8948.7053.1656.0858.0459.5560.2560.7760.8128.2335.3141.6347.0751.7255.3658.2960.6362.0762.8063.20Process RMOutcome RMOutcome \& Process RMSFT-7b + RM-7b (Weighted Voting on PRM800K)Accuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem5.07.510.012.515.017.520.022.525.0% Problems Solved6.908.8210.7912.9314.9516.6618.1219.6220.8921.9822.776.929.6211.9514.1915.8717.4118.6919.8620.5121.0121.236.939.3211.4113.6915.6917.7019.2520.8922.3923.3324.12Process RMOutcome RMOutcome \& Process RMSFT-7b + RM-7b (Weighted Voting on PRM800K)Accuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem15202530354045% Problems Solved16.9921.1024.9528.5931.8334.3236.3537.9339.0539.9340.4716.9722.2326.6630.6433.6135.8037.4138.7439.4239.9039.9817.0721.6725.8429.5732.8535.6037.8639.7541.2742.0942.74Process RMOutcome RMOutcome \& Process RMSFT-7b + RM-7b (Weighted Voting on PRM800K)Accuracy on All (Level 1-5) Problems12481632641282565121024N = number of solutions per problem253035404550556065% Problems Solved28.0234.2638.8542.2545.5447.6149.0649.8850.2149.8149.0928.2136.1641.7646.0748.5150.2250.9051.1651.2950.9250.9127.8735.6641.6946.0749.2351.3852.8653.8754.6555.0355.06Process RMOutcome RMOutcome \& Process RMSFT-7b + RM-7b (Best-of-N on PRM800K)Accuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem5.07.510.012.515.017.520.022.525.0% Problems Solved6.938.9310.3511.8212.8814.0514.7015.2815.6715.7915.666.889.4411.5013.0914.0614.8515.6015.9616.2516.3816.346.809.4811.8413.9215.2616.5017.6618.7019.3819.9519.91Process RMOutcome RMOutcome \& Process RMSFT-7b + RM-7b (Best-of-N on PRM800K)Accuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem15202530354045% Problems Solved17.0020.9523.9526.3128.4029.9931.0531.7232.0931.9531.5517.0622.1725.9128.7530.4431.6632.3832.6732.8832.8032.7716.8121.9326.0429.2031.4133.0834.4235.4336.1736.6236.66Process RMOutcome RMOutcome \& Process RMSFT-7b + RM-7b (Best-of-N on PRM800K)Accuracy on All (Level 1-5) Problems12481632641282565121024N = number of solutions per problem354045505560657075% Problems Solved37.6545.4452.3757.8462.7166.3168.4169.7770.5970.6970.4837.8446.7154.4260.6665.2968.3870.2871.1871.5471.7171.5337.4746.8854.2560.1464.9668.3671.0072.6473.6174.0874.37Process RMOutcome RMOutcome \& Process RMSFT-34b + RM-34b (Weighted Voting on PRM800K)Accuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem5101520253035% Problems Solved10.3613.3216.1418.8821.2023.4025.4927.0727.9328.7029.3810.2713.7317.0319.9022.5624.8826.6227.8729.0129.7830.1910.3413.8717.1820.2723.0425.4227.3429.1630.5031.6132.58Process RMOutcome RMOutcome \& Process RMSFT-34b + RM-34b (Weighted Voting on PRM800K)Accuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem2025303540455055% Problems Solved23.3828.5833.3737.4440.9543.8045.9847.4348.2448.7248.9123.3829.4234.8239.3042.9045.5847.4548.4949.2849.7349.8823.2629.5834.8339.2142.9945.8448.1749.8750.9851.8152.45Process RMOutcome RMOutcome \& Process RMSFT-34b + RM-34b (Weighted Voting on PRM800K)Accuracy on All (Level 1-5) Problems12481632641282565121024N = number of solutions per problem354045505560657075% Problems Solved37.6045.2050.7455.0658.4860.8262.4063.6464.0964.5164.5737.6546.7253.4157.5160.2562.0562.8363.5064.2164.5364.9937.7946.9754.1659.8363.4065.8867.5768.1968.2468.0467.44Process RMOutcome RMOutcome \& Process RMSFT-34b + RM-34b (Best-of-N on PRM800K)Accuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem5101520253035% Problems Solved10.1613.1115.5117.5218.8620.1621.3522.4023.0923.4323.6710.2313.7516.5618.5320.1721.2421.9522.4522.7423.0023.7010.3114.1117.4020.3122.3024.0424.9725.7326.1626.6827.02Process RMOutcome RMOutcome \& Process RMSFT-34b + RM-34b (Best-of-N on PRM800K)Accuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem2025303540455055% Problems Solved23.2428.4132.2735.3937.7539.5140.8942.0242.6142.9643.1223.2629.4534.0937.1039.2440.6441.3841.9642.4542.7343.3223.4129.7634.8839.1041.8643.9445.2645.9746.2546.3746.31Process RMOutcome RMOutcome \& Process RMSFT-34b + RM-34b (Best-of-N on PRM800K)Accuracy on All (Level 1-5) ProblemsFigure 13: Comparing different reward models with Weighted Voting (upper) and Best-of-N (lower)
on 34b SFT model and 7b reward model trained on the PRM800K dataset. Both SFTs and RMs are
trained on the easy data.

Figure 14: Comparing different reward models with Weighted Voting (upper) and Best-of-N (lower)
on 7b models trained on the MetaMath dataset. Both SFTs and RMs are trained on the easy data.

28

12481632641282565121024N = number of solutions per problem354045505560657075% Problems Solved37.6145.0151.3856.6861.5164.9967.3268.9869.8570.3270.4137.6246.6254.0060.1964.7767.9869.7170.4170.7270.8871.0137.7245.9952.5858.0462.6366.0368.5770.4271.4272.2872.68Process RMOutcome RMOutcome \& Process RMSFT-34b + RM-7b (Weighted Voting on PRM800K)Accuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem5101520253035% Problems Solved10.2612.9615.7518.2520.5322.6924.6326.1127.2327.7528.1210.3513.4816.5319.6022.0924.1325.9627.0627.8828.6029.0910.2313.3316.3318.9921.4423.5825.6427.4128.8030.0630.68Process RMOutcome RMOutcome \& Process RMSFT-34b + RM-7b (Weighted Voting on PRM800K)Accuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem2025303540455055% Problems Solved23.2928.1932.6436.5440.0142.8144.9946.5647.5348.0248.2523.3429.2534.3538.9642.3845.0546.7947.7248.3348.7649.0623.3128.8633.5837.5941.0443.7746.1247.9349.1050.1550.67Process RMOutcome RMOutcome \& Process RMSFT-34b + RM-7b (Weighted Voting on PRM800K)Accuracy on All (Level 1-5) Problems12481632641282565121024N = number of solutions per problem354045505560657075% Problems Solved37.5244.7949.9353.8056.5258.5860.0161.3562.4163.3163.8937.7146.3152.5756.5958.4259.4559.6659.7459.7059.0957.7237.5046.3152.5856.8459.2960.9262.1362.8763.2763.6063.46Process RMOutcome RMOutcome \& Process RMSFT-34b + RM-7b (Best-of-N on PRM800K)Accuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem5101520253035% Problems Solved10.2913.2615.1616.9118.3919.5320.4120.7821.0021.1121.1610.3113.4915.7716.9618.0718.5219.0519.0018.9718.9018.8610.3913.5916.6118.7120.1721.4422.5923.3224.0424.3324.38Process RMOutcome RMOutcome \& Process RMSFT-34b + RM-7b (Best-of-N on PRM800K)Accuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem2025303540455055% Problems Solved23.2428.2231.6734.4636.5538.1139.2640.0940.7241.2041.5123.3729.1333.2835.8137.2738.0038.3838.3738.3338.0237.3323.2729.1633.6936.8638.7540.2041.4142.1142.6943.0242.96Process RMOutcome RMOutcome \& Process RMSFT-34b + RM-7b (Best-of-N on PRM800K)Accuracy on All (Level 1-5) Problems12481632641282565121024N = number of solutions per problem40455055606570% Problems Solved44.7550.2254.6458.5361.2263.2264.5564.9865.1665.1765.2744.4149.8754.3557.9860.8462.9264.1064.8265.4065.6165.7244.7650.1855.1859.2062.1164.2966.0767.1367.6767.8767.80Process RMOutcome RMOutcome \& Process RMSFT-7b + RM-7b (Weighted Voting on MetaMath)Accuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem1012141618202224% Problems Solved12.0814.3016.6818.5520.3821.4122.1122.5122.7222.8823.0912.0514.3716.3718.0719.5820.7621.2521.5921.6021.7121.7712.1714.4216.4218.3019.8320.9121.5221.8322.0522.2022.31Process RMOutcome RMOutcome \& Process RMSFT-7b + RM-7b (Weighted Voting on MetaMath)Accuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem25.027.530.032.535.037.540.042.545.0% Problems Solved27.6331.3834.7437.5939.8141.3242.2942.6742.8942.9343.0827.4631.2534.4737.0739.2240.8741.6642.1442.4142.5642.6327.6631.4434.8337.7539.9841.5442.6943.4043.7643.9543.90Process RMOutcome RMOutcome \& Process RMSFT-7b + RM-7b (Weighted Voting on MetaMath)Accuracy on All (Level 1-5) Problems12481632641282565121024N = number of solutions per problem40455055606570% Problems Solved44.6950.3353.7155.9557.7958.9759.7360.3960.6260.7860.8044.7249.6652.6454.4855.5956.2656.8957.3957.8557.7757.4544.7850.3553.4655.6156.4657.2057.2156.9257.0656.9256.58Process RMOutcome RMOutcome \& Process RMSFT-7b + RM-7b (Best-of-N on MetaMath)Accuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem1012141618202224% Problems Solved11.9814.4816.0917.3017.9218.2818.6418.8018.8218.3317.5411.9514.2515.9117.3118.5519.0819.2319.3319.4319.5319.4012.0214.3916.0317.0717.6217.8517.9118.3219.0419.7319.48Process RMOutcome RMOutcome \& Process RMSFT-7b + RM-7b (Best-of-N on MetaMath)Accuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem25.027.530.032.535.037.540.042.545.0% Problems Solved27.5731.5233.9435.6936.8937.6238.2038.6038.7338.4938.1027.5231.1233.3835.0036.1836.7537.1737.4437.7137.7537.5027.6031.4633.8235.4136.1136.5836.6336.7137.1637.4337.16Process RMOutcome RMOutcome \& Process RMSFT-7b + RM-7b (Best-of-N on MetaMath)Accuracy on All (Level 1-5) ProblemsTable 9: Results of Full, Easy-to-Hard, & Hard-to-Easy SFT training of the Llemma-7b model

TRAINING DATA

FULL SFT
EASY-TO-HARD SFT
HARD-TO-EASY SFT

ALL
EASY
HARD

PRM800K

METAMATH
ALL HARD ALL HARD

20.6
19.8
18.4

9.9
12.2
13.0

31.4
30.0
30.4

13.7
14.9
15.3

Figure 15: The agreement between the prediction from the Llemma-7b-based reward model when
trained on ORM and PRM data, and their agreement to ground-truth final-answer labels.

J Hard-to-Easy Generalization

From Table 5, it is evident that reinforcement learning training on hard tasks alone significantly
underperforms compared to training the model on easy tasks or on all tasks. This difference is
especially pronounced for PPO on the PRM800K dataset. This raises a crucial question: does training
on hard tasks only generalize to easy tasks?

To address this, we fine-tuned the Llemma-7b model using all data (easy and hard), only easy data,
and only hard data. As shown in Table 9, we found that training on all data consistently yields the
best performance. Conversely, the generator’s performance deteriorates when transitioning from easy-
to-hard and hard-to-easy tasks. This suggests that language models face difficulties in generalizing in
both directions.

It is also worth noting that while Full SFT underperforms Easy-to-Hard SFT and Hard-to-Easy SFT
on hard test questions, it eventually outperforms Easy-to-Hard SFT and Hard-to-Easy SFT when
evaluated on all test questions. We believe that this is because by exposing the model to a wider
variety of unique questions and difficulties, it gains a better understanding of the problem space in
general, as measured by the accuracy on the full distribution.

K On ORM’s Approximation of PRM Labels

From Sec. G, we observe that in PRM800K, PRMs and ORMs exhibit similar performance levels,
with OPRMs outperforming both. This raises the question of why ORMs also demonstrate strong
easy-to-hard generalization ability. A straightforward explanation is that ORMs are trained to
approximate PRM labels [74]. Specifically, ORMs are trained to predict the correctness of the entire
solution through value estimation. As Uesato et al. [74] state, “it is simpler for the ORM to learn
to recognize when steps are correct than it is to check the answer by internally computing the final
answer itself.”

Nevertheless, people may argue that the conclusion from Uesato et al. [74] is based on GSM8K’s
experimental results, so the conclusion may not transfer to the more challenging Hendrick’s MATH
dataset. To show the universal existence of “ORM’s approximation of PRM labels”, we further
conduct evaluation of agreement between different rewards on two variants of the MATH dataset:
PRM800K and MetaMath.

The results are shown in Figure 15. Similarly to the findings from Uesato et al. [74], we see that the
ORM has higher agreement with the PRM, despite being trained to predict the Final-Answer rewards.

29

Thus, “this result indicates that the ORM tends more towards predicting whether the full trace is
correct, and not just whether the final answer is correct.”

Overall, this shows easy-to-hard generalization is not exclusively linked to reward models trained on
explicit step-wise annotations. It also applies to ORMs that are trained to perform value estimation
and practically evaluates each solution step.

We also perform DPO training on a MetaMath-initialized Llemma-7b model. We find that in this
RL setting, re-ranking the output pairs with ORM also gives similar performance to re-ranking with
PRM (29.2 v.s. 30.4 & 31.2 v.s. 34.0).

L Analysis of Aggregation Functions in PRMs & OPRMs

We explored different methods to consolidate step-wise prediction scores into a single score value, a
process we describe as employing an aggregation function, during the use of the evaluator. Lightman
et al. [40] report comparable performance when using min (minimum) and prod (product) as the
aggregation function to reduce multiple scores into a single solution-level score. Note that when
training PRMs on PRM800K [40], we have already considered neutral steps to be positive as training
labels.

Following Wang et al. [77], given {p1, p2, . . . , pn} as a list of predicted correctness probability of
each step (including the final answer), we considered the following aggregation functions:

min = min{p1, p2, . . . , pn}

max = max{p1, p2, . . . , pn}

prod =

(cid:89)

pi

mean =

mean_logit = σ

i
(cid:80)

i pi
n
(cid:32) (cid:80)

i log pi
1−pi
n

(cid:33)

mean_odd = ReLU

(cid:32) (cid:80)
i

(cid:33)

pi
1−pi
n

(1)

(2)

(3)

(4)

(5)

(6)

last = pn
(7)
In Figure 16-18, we perform analysis of aggregation functions on PRM800K and Math-Shepherd
(from MetaMath) datasets with weighted voting and best-of-n decoding and PRMs or OPRMs. In
general, we find prod works universally well in weighted voting and min works well in best-of-n.
So we adopt these two strategies in our main experiments.

One interesting finding is that for reward models trained on the human annotated process reward (e.g.,
PRM800K), the last strategy does not perform very well, but last works much better on OPRMs
and pseudo PRMs (e.g., Math-Shepherd). This could partially explain why OPRMs does not further
improve the performance on the Math-Shepherd dataset.

30

Figure 16: Analysis of aggregation functions in process reward models (PRMs) on the PRM800K
dataset with Weighted Voting and Best-of-N. Both SFTs and RMs are trained on the easy data.

31

12481632641282565121024N = number of solutions per problem30354045505560% Problems Solvedprodminmaxmeanmean_logitmean_oddlastSFT-7b + PRM-7b + Weighted VotingAccuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem6810121416182022% Problems Solvedprodminmaxmeanmean_logitmean_oddlastSFT-7b + PRM-7b + Weighted VotingAccuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem2025303540% Problems Solvedprodminmaxmeanmean_logitmean_oddlastSFT-7b + PRM-7b + Weighted VotingAccuracy on All (Level 1-5) Problems12481632641282565121024N = number of solutions per problem3035404550% Problems Solvedprodminmaxmeanmean_logitmean_oddlastSFT-7b + PRM-7b + Best-of-NAccuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem810121416% Problems Solvedprodminmaxmeanmean_logitmean_oddlastSFT-7b + PRM-7b + Best-of-NAccuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem1820222426283032% Problems Solvedprodminmaxmeanmean_logitmean_oddlastSFT-7b + PRM-7b + Best-of-NAccuracy on All (Level 1-5) Problems12481632641282565121024N = number of solutions per problem40455055606570% Problems Solvedprodminmaxmeanmean_logitmean_oddlastSFT-34b + PRM-34b + Weighted VotingAccuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem10.012.515.017.520.022.525.027.530.0% Problems Solvedprodminmaxmeanmean_logitmean_oddlastSFT-34b + PRM-34b + Weighted VotingAccuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem253035404550% Problems Solvedprodminmaxmeanmean_logitmean_oddlastSFT-34b + PRM-34b + Weighted VotingAccuracy on All (Level 1-5) Problems12481632641282565121024N = number of solutions per problem404550556065% Problems Solvedprodminmaxmeanmean_logitmean_oddlastSFT-34b + PRM-34b + Best-of-NAccuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem1012141618202224% Problems Solvedprodminmaxmeanmean_logitmean_oddlastSFT-34b + PRM-34b + Best-of-NAccuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem25303540% Problems Solvedprodminmaxmeanmean_logitmean_oddlastSFT-34b + PRM-34b + Best-of-NAccuracy on All (Level 1-5) ProblemsFigure 17: Analysis of aggregation functions in outcome & process reward models (OPRMs) on the
PRM800K dataset with Weighted Voting and Best-of-N. Both SFTs and RMs are trained on the easy
data.

32

12481632641282565121024N = number of solutions per problem3035404550556065% Problems Solvedprodminmaxmeanmean_logitmean_oddlastSFT-7b + OPRM-7b + Weighted VotingAccuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem7.510.012.515.017.520.022.5% Problems Solvedprodminmaxmeanmean_logitmean_oddlastSFT-7b + OPRM-7b + Weighted VotingAccuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem2025303540% Problems Solvedprodminmaxmeanmean_logitmean_oddlastSFT-7b + OPRM-7b + Weighted VotingAccuracy on All (Level 1-5) Problems12481632641282565121024N = number of solutions per problem303540455055% Problems Solvedprodminmaxmeanmean_logitmean_oddlastSFT-7b + OPRM-7b + Best-of-NAccuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem8101214161820% Problems Solvedprodminmaxmeanmean_logitmean_oddlastSFT-7b + OPRM-7b + Best-of-NAccuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem17.520.022.525.027.530.032.535.037.5% Problems Solvedprodminmaxmeanmean_logitmean_oddlastSFT-7b + OPRM-7b + Best-of-NAccuracy on All (Level 1-5) Problems12481632641282565121024N = number of solutions per problem4045505560657075% Problems Solvedprodminmaxmeanmean_logitmean_oddlastSFT-34b + OPRM-34b + Weighted VotingAccuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem1015202530% Problems Solvedprodminmaxmeanmean_logitmean_oddlastSFT-34b + OPRM-34b + Weighted VotingAccuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem253035404550% Problems Solvedprodminmaxmeanmean_logitmean_oddlastSFT-34b + OPRM-34b + Weighted VotingAccuracy on All (Level 1-5) Problems12481632641282565121024N = number of solutions per problem40455055606570% Problems Solvedprodminmaxmeanmean_logitmean_oddlastSFT-34b + OPRM-34b + Best-of-NAccuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem10.012.515.017.520.022.525.027.5% Problems Solvedprodminmaxmeanmean_logitmean_oddlastSFT-34b + OPRM-34b + Best-of-NAccuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem2530354045% Problems Solvedprodminmaxmeanmean_logitmean_oddlastSFT-34b + OPRM-34b + Best-of-NAccuracy on All (Level 1-5) ProblemsFigure 18: Analysis of aggregation functions in psuedo process reward models (PRMs) on the
Math-Shepherd (from MetaMath) dataset with Weighted Voting and Best-of-N. Both SFTs and RMs
are trained on the easy data.

M Societal Impact

Our work on easy-to-hard generalization has the potential for both positive and negative societal
impacts. On the positive side, this approach could enable AI systems to tackle increasingly complex
problems in domains such as scientific discovery, healthcare, and education, potentially leading to
groundbreaking advancements that benefit society. However, the development of AI systems that
can operate beyond human supervision also raises concerns about the transparency, accountability,
and potential misuse of such systems. It is crucial to carefully consider the ethical implications and
establish robust safeguards to mitigate the risks of unintended consequences or malicious applications.
Ongoing research and public discourse on the responsible development and deployment of these
technologies will be essential to ensure that their societal benefits outweigh the potential drawbacks.

33

12481632641282565121024N = number of solutions per problem4550556065% Problems SolvedprodminmeanlastMetaMath-7b + Math-Shepherd-7b + Weighted VotingAccuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem121416182022% Problems SolvedprodminmeanlastMetaMath-7b + Math-Shepherd-7b + Weighted VotingAccuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem283032343638404244% Problems SolvedprodminmeanlastMetaMath-7b + Math-Shepherd-7b + Weighted VotingAccuracy on All (Level 1-5) Problems12481632641282565121024N = number of solutions per problem4446485052545658% Problems SolvedprodminmeanlastMetaMath-7b + Math-Shepherd-7b + Best-of-NAccuracy on Easy (Level 1-3) Problems12481632641282565121024N = number of solutions per problem1214161820% Problems SolvedprodminmeanlastMetaMath-7b + Math-Shepherd-7b + Best-of-NAccuracy on Hard (Level 4-5) Problems12481632641282565121024N = number of solutions per problem283032343638% Problems SolvedprodminmeanlastMetaMath-7b + Math-Shepherd-7b + Best-of-NAccuracy on All (Level 1-5) ProblemsN Fine-Grained Analysis of OPRMs’ Re-ranking strategies

Figure 19: Easy-to-hard generalization for different difficulty levels’ data. Both SFTs and OPRMs
are trained on the level1-3 data. Each row compares the performance of different OPRMs’ reranking
strategies across different levels’ data.

As shown in Figure 19, both the Best-of-N and Weighted Voting strategies demonstrate strong perfor-
mance across all levels, which leverage the advantages of OPRM methods and thereby underscoring
OPRM’s effectiveness. Furthermore, despite the SFT models and OPRM models being trained on
level 1-3 data, re-ranking strategies enhanced by OPRMs continue to perform well on the unseen and
more challenging level 4-5 data. This indicates the feasibility of generalizing from easier to harder
tasks using OPRMs.

34

12481632641282565121024N = number of solutions per problem4050607080% Problems Solved41.7542.6552.2761.0766.6769.9871.5372.5572.4772.4272.1341.3450.9259.3165.4970.4974.5678.0480.6582.2182.9583.4541.4852.0859.3765.1670.2872.9675.0675.6975.9775.3073.19Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-7b (on PRM800K)Accuracy on Level1 Problems12481632641282565121024N = number of solutions per problem4050607080% Problems Solved41.7542.6552.2761.0766.6769.9871.5372.5572.4772.4272.1341.6752.0960.7966.6470.4273.2875.4677.0378.4279.5280.0241.6053.0962.0668.5273.9176.9779.4581.1782.0282.6884.05Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-34b (on PRM800K)Accuracy on Level1 Problems12481632641282565121024N = number of solutions per problem5060708090% Problems Solved53.0654.3265.2974.0480.4184.6687.4289.0289.6789.6789.7553.0962.6470.4476.2880.3683.5685.2286.3286.9187.1287.3652.9463.8969.8474.4177.5779.6180.8481.3981.9782.7583.75Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-7b (on PRM800K)Accuracy on Level1 Problems12481632641282565121024N = number of solutions per problem5060708090% Problems Solved53.0654.3265.2974.0480.4184.6687.4289.0289.6789.6789.7553.2663.8172.6778.1882.1184.5086.1886.6686.6486.3586.0353.1564.9972.1177.2880.4682.6983.9483.9883.9483.3282.90Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-34b (on PRM800K)Accuracy on Level1 Problems12481632641282565121024N = number of solutions per problem303540455055606570% Problems Solved33.0033.9941.9850.2356.1259.7961.9863.7164.7465.0764.5833.1441.0247.7252.8857.5960.9763.1965.1566.4967.4067.9133.4941.4447.0650.5353.6555.2857.2358.7860.0361.1162.45Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-7b (on PRM800K)Accuracy on Level2 Problems12481632641282565121024N = number of solutions per problem304050607080% Problems Solved33.0033.9941.9850.2356.1259.7961.9863.7164.7465.0764.5833.3341.7749.3555.3560.6964.9268.2370.9473.0174.6476.0633.1342.9550.0855.6960.1463.2265.7167.1468.0668.8469.38Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-34b (on PRM800K)Accuracy on Level2 Problems12481632641282565121024N = number of solutions per problem404550556065707580% Problems Solved43.2443.7753.9362.4168.5271.5373.1774.1074.8374.8874.7543.4551.6258.6063.9368.3371.5574.0775.9777.3478.0478.2543.0252.2358.5961.6163.2164.3064.8165.4065.4565.2564.28Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-7b (on PRM800K)Accuracy on Level2 Problems12481632641282565121024N = number of solutions per problem404550556065707580% Problems Solved43.2443.7753.9362.4168.5271.5373.1774.1074.8374.8874.7543.5953.3460.5667.0271.3174.6377.1678.3878.8878.9978.8343.2453.7560.8166.0669.5471.8672.9873.5073.2272.5271.62Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-34b (on PRM800K)Accuracy on Level2 Problems12481632641282565121024N = number of solutions per problem1520253035404550% Problems Solved18.2718.8824.4531.3737.0241.0543.2043.7544.1344.2944.0718.0823.6029.4033.9739.0043.3646.3548.5549.9150.8350.9917.8024.2230.0033.8336.9338.7940.0140.6441.2341.2441.16Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-7b (on PRM800K)Accuracy on Level3 Problems12481632641282565121024N = number of solutions per problem152025303540455055% Problems Solved18.2718.8824.4531.3737.0241.0543.2043.7544.1344.2944.0718.1824.4330.7336.0741.2845.3348.2750.3951.5852.3752.5917.9925.8431.4837.2441.5845.1047.1948.8950.2551.1551.52Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-34b (on PRM800K)Accuracy on Level3 Problems12481632641282565121024N = number of solutions per problem253035404550556065% Problems Solved26.7727.8534.4442.7649.3052.8054.4754.9255.1355.0654.5926.5233.3739.7445.7450.3054.2256.9058.9660.4261.1762.0326.6633.8740.2844.8448.1650.2351.9752.8453.3754.1353.96Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-7b (on PRM800K)Accuracy on Level3 Problems12481632641282565121024N = number of solutions per problem3040506070% Problems Solved26.7727.8534.4442.7649.3052.8054.4754.9255.1355.0654.5926.7433.9241.2347.2852.2556.5660.0562.2263.7665.1766.1926.5834.4941.3147.0551.3253.5655.7057.3557.7358.2357.12Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-34b (on PRM800K)Accuracy on Level3 Problems12481632641282565121024N = number of solutions per problem101520253035% Problems Solved10.3210.9313.7017.8821.6223.8225.4726.3926.8327.2127.5610.3914.0117.5420.5623.7626.8929.0731.0332.8833.8434.4910.6414.3017.8020.5722.7224.6726.3427.6028.3128.8029.52Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-7b (on PRM800K)Accuracy on Level4 Problems12481632641282565121024N = number of solutions per problem10152025303540% Problems Solved10.3210.9313.7017.8821.6223.8225.4726.3926.8327.2127.5610.3514.6218.7522.5725.6628.7531.6634.4436.4138.7040.5110.4414.8219.5523.3926.5529.4631.8633.9436.0737.1037.59Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-34b (on PRM800K)Accuracy on Level4 Problems12481632641282565121024N = number of solutions per problem15202530354045% Problems Solved15.4016.2620.2225.9230.0033.4335.9437.8338.5438.8138.6915.3719.9523.8427.9631.6935.0938.2540.6142.6944.3945.6115.2920.0124.1927.3229.6031.9533.0534.3835.5036.5036.41Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-7b (on PRM800K)Accuracy on Level4 Problems12481632641282565121024N = number of solutions per problem15202530354045% Problems Solved15.4016.2620.2225.9230.0033.4335.9437.8338.5438.8138.6915.1720.3125.3029.5333.3036.8139.5842.2543.9145.3146.0815.1820.7725.6329.1532.5034.4836.5037.9538.2239.3040.37Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-34b (on PRM800K)Accuracy on Level4 Problems12481632641282565121024N = number of solutions per problem2468101214% Problems Solved3.453.854.615.656.908.128.839.6610.4910.7911.193.474.645.776.847.948.789.9511.0212.2313.2614.103.454.806.257.348.028.849.5010.0210.7411.0010.75Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-7b (on PRM800K)Accuracy on Level5 Problems12481632641282565121024N = number of solutions per problem246810121416% Problems Solved3.453.854.615.656.908.128.839.6610.4910.7911.193.404.666.087.378.9010.3311.8912.9014.5415.6115.783.434.996.618.119.3210.6611.7112.5213.4214.2715.02Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-34b (on PRM800K)Accuracy on Level5 Problems12481632641282565121024N = number of solutions per problem4681012141618% Problems Solved5.276.047.139.3711.4312.8913.8514.2214.3214.4614.735.367.108.9410.4011.6612.8313.8414.8515.6616.1916.365.227.349.0910.2911.3511.9412.2312.6112.9213.0112.74Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-7b (on PRM800K)Accuracy on Level5 Problems12481632641282565121024N = number of solutions per problem2.55.07.510.012.515.017.520.0% Problems Solved5.276.047.139.3711.4312.8913.8514.2214.3214.4614.735.147.349.5311.1612.9214.3715.3816.4817.5518.4519.775.417.629.8211.6712.8813.8314.1814.3214.6114.5814.47Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-34b (on PRM800K)Accuracy on Level5 ProblemsFigure 20: Easy-to-hard generalization for different type’s data. Both SFTs and OPRMs are trained
on the easy data. Each row compares the performance of different OPRMs’ reranking strategies
across different types’ data.

To determine which types of data benefit more from OPRM’s easy-to-hard generalization and which
types still struggle with this challenging generalization, we compare OPRM’s generalization abilities
on different problem types in Figure 20. Among Algebra, Counting & Probability, Geometry,
Intermediate Algebra, Number Theory, Prealgebra, and Precalculus problems, OPRMs generalize
best on Algebra, Intermediate Algebra, and Precalculus problems. Conversely, OPRMs generalize
worst on Counting & Probability problems. These findings are highly valuable for practical system
design, allowing us to decide when to use OPRMs to enhance performance based on the downstream
data type.

35

12481632641282565121024N = number of solutions per problem253035404550556065% Problems Solved27.6728.1534.8242.2047.7551.4253.8655.2356.3557.0757.1027.5234.3040.3145.1949.8053.3856.4258.8561.0562.7863.5727.6234.8240.0243.8446.5948.5250.3051.5852.7653.2854.02Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-7b (on PRM800K)Accuracy on Algebra Problems12481632641282565121024N = number of solutions per problem3040506070% Problems Solved27.6728.1534.8242.2047.7551.4253.8655.2356.3557.0757.1027.7435.2942.0947.6152.1755.9659.3961.6964.1966.3167.3627.6836.1242.6648.3052.0655.3357.6859.3361.0062.0363.11Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-34b (on PRM800K)Accuracy on Algebra Problems12481632641282565121024N = number of solutions per problem3540455055606570% Problems Solved35.8036.6643.8651.2856.3159.5061.3161.9862.4562.4562.5435.9243.1049.1454.4458.6262.1664.8466.6167.7468.7569.4635.8043.5749.1453.3455.7157.6858.8759.8460.7161.2660.84Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-7b (on PRM800K)Accuracy on Algebra Problems12481632641282565121024N = number of solutions per problem3540455055606570% Problems Solved35.8036.6643.8651.2856.3159.5061.3161.9862.4562.4562.5435.8743.6250.4255.8460.4063.8566.3568.1669.4470.3871.4635.9044.4850.5655.4959.2161.3262.7363.6863.7863.9363.78Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-34b (on PRM800K)Accuracy on Algebra Problems12481632641282565121024N = number of solutions per problem101520253035% Problems Solved13.6613.8517.4020.9623.5325.3926.2327.4428.8930.6932.5313.2616.4219.2621.4322.7423.6424.2125.3025.5925.2924.9313.4816.4819.1220.5221.6622.1722.8923.5824.1024.7824.24Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-7b (on PRM800K)Accuracy on Counting & Probability Problems12481632641282565121024N = number of solutions per problem101520253035% Problems Solved13.6613.8517.4020.9623.5325.3926.2327.4428.8930.6932.5313.8017.7220.7123.2825.0126.4927.5528.9730.4231.4834.2113.4317.8222.0724.4825.6927.3628.5428.4429.2828.2226.94Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-34b (on PRM800K)Accuracy on Counting & Probability Problems12481632641282565121024N = number of solutions per problem15202530354045% Problems Solved19.6020.0424.8028.8832.0134.4536.5838.5139.0839.6739.9719.7223.2426.5930.1832.9135.6638.3941.0343.3644.4845.2919.5123.6626.4927.7428.4829.1129.6730.2730.3831.4131.75Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-7b (on PRM800K)Accuracy on Counting & Probability Problems12481632641282565121024N = number of solutions per problem15202530354045% Problems Solved19.6020.0424.8028.8832.0134.4536.5838.5139.0839.6739.9719.7124.7028.3631.6734.3036.1237.9739.8641.9343.2844.3119.8624.8327.8130.0731.9532.7333.5033.8433.8934.7535.44Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-34b (on PRM800K)Accuracy on Counting & Probability Problems12481632641282565121024N = number of solutions per problem1015202530354045% Problems Solved13.4614.2617.6322.2226.0229.1231.1632.4933.2232.6131.9213.7318.2922.8025.4129.6633.3936.8439.9142.3744.2245.2213.7318.5422.9625.9628.4930.9632.9234.8536.5837.2539.34Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-7b (on PRM800K)Accuracy on Geometry Problems12481632641282565121024N = number of solutions per problem101520253035404550% Problems Solved13.4614.2617.6322.2226.0229.1231.1632.4933.2232.6131.9213.8118.5622.5026.6930.4434.5237.7340.8243.3944.9245.5813.3119.3624.2527.2330.9033.5535.9438.7740.4042.0542.91Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-34b (on PRM800K)Accuracy on Geometry Problems12481632641282565121024N = number of solutions per problem1520253035404550% Problems Solved19.2020.3524.7731.6337.4641.9244.5346.1746.2045.7345.2419.5825.3630.3834.8638.8142.4745.2146.9747.6547.6047.4818.9125.3031.3534.3637.0039.0439.7039.9640.7141.6841.96Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-7b (on PRM800K)Accuracy on Geometry Problems12481632641282565121024N = number of solutions per problem20304050% Problems Solved19.2020.3524.7731.6337.4641.9244.5346.1746.2045.7345.2419.3825.2231.1935.7140.0943.3547.0850.1751.7652.8053.2519.3925.4830.7335.2738.0940.9042.3942.6342.9342.1743.00Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-34b (on PRM800K)Accuracy on Geometry Problems12481632641282565121024N = number of solutions per problem510152025% Problems Solved6.607.439.3312.1814.6917.4419.2020.3021.0421.5621.836.438.9211.3113.7516.2318.9420.3421.9622.9323.4023.766.449.0211.7914.2516.5018.2719.3920.5221.3221.6721.63Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-7b (on PRM800K)Accuracy on Intermediate Algebra Problems12481632641282565121024N = number of solutions per problem51015202530% Problems Solved6.607.439.3312.1814.6917.4419.2020.3021.0421.5621.836.458.9611.2614.1216.4318.8320.7122.5023.8324.9525.286.629.3712.1915.0617.9020.2821.8623.3524.5825.8327.07Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-34b (on PRM800K)Accuracy on Intermediate Algebra Problems12481632641282565121024N = number of solutions per problem51015202530% Problems Solved8.149.6512.1015.4819.4021.5122.8723.2723.6523.7823.458.4111.1514.0917.1219.8822.1724.3226.2627.9029.0429.648.3811.6014.8117.6219.9321.3822.7924.1225.1326.1527.15Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-7b (on PRM800K)Accuracy on Intermediate Algebra Problems12481632641282565121024N = number of solutions per problem51015202530% Problems Solved8.149.6512.1015.4819.4021.5122.8723.2723.6523.7823.458.4411.5814.7918.2121.3623.7125.8427.5229.0630.5131.478.3111.7815.4118.8121.9724.3926.0628.0229.3030.2730.52Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-34b (on PRM800K)Accuracy on Intermediate Algebra Problems12481632641282565121024N = number of solutions per problem101520253035% Problems Solved11.3312.4115.2518.7621.7023.7624.5524.6524.4324.2324.1211.5414.2717.6920.6223.7226.5829.0731.0032.8033.5934.6511.6714.9817.6719.7322.4523.5624.8025.6626.7427.1228.14Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-7b (on PRM800K)Accuracy on Number Theory Problems12481632641282565121024N = number of solutions per problem10152025303540% Problems Solved11.3312.4115.2518.7621.7023.7624.5524.6524.4324.2324.1211.3714.8319.1722.2825.6328.0631.2633.6834.8636.7538.9111.6115.8619.3223.1225.6027.8530.4132.6734.5936.1836.87Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-34b (on PRM800K)Accuracy on Number Theory Problems12481632641282565121024N = number of solutions per problem1520253035404550% Problems Solved19.0320.0325.1130.9536.2339.9242.6344.7745.9246.6346.8519.0223.3227.5931.5535.2538.4740.3242.3843.7644.8146.5318.8323.6227.4329.9532.2333.5234.3635.1935.5736.4237.31Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-7b (on PRM800K)Accuracy on Number Theory Problems12481632641282565121024N = number of solutions per problem1520253035404550% Problems Solved19.0320.0325.1130.9536.2339.9242.6344.7745.9246.6346.8518.8924.1829.4433.7037.7241.8844.3046.4447.0647.7248.0118.8624.2429.3132.9636.3838.2539.9741.4940.7340.8038.78Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-34b (on PRM800K)Accuracy on Number Theory Problems12481632641282565121024N = number of solutions per problem25303540455055% Problems Solved27.1627.7434.9443.4549.7252.3953.7354.3654.1953.3652.8027.1734.6241.1246.1750.0553.0754.8655.4155.8356.1056.0027.4735.2841.8645.6347.8648.9949.5549.3949.0648.7947.50Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-7b (on PRM800K)Accuracy on Prealgebra Problems12481632641282565121024N = number of solutions per problem30405060% Problems Solved27.1627.7434.9443.4549.7252.3953.7354.3654.1953.3652.8026.9735.1443.4249.1554.1058.2660.7362.5763.6964.3664.2726.8836.2743.8849.8454.5057.9859.7760.9761.9062.3962.81Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-34b (on PRM800K)Accuracy on Prealgebra Problems12481632641282565121024N = number of solutions per problem3540455055606570% Problems Solved37.8337.8847.3356.7562.5566.1467.4167.7967.6567.3467.2336.9445.8752.3157.5361.1164.1065.7466.6867.9268.5668.6837.0146.5652.3355.5057.2858.5458.4058.0657.5156.7055.27Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-7b (on PRM800K)Accuracy on Prealgebra Problems12481632641282565121024N = number of solutions per problem3540455055606570% Problems Solved37.8337.8847.3356.7562.5566.1467.4167.7967.6567.3467.2337.0446.8855.0260.7764.5567.4169.1169.9670.2570.4570.9836.9047.8755.5360.1862.7663.8064.1963.6463.1062.3661.47Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-34b (on PRM800K)Accuracy on Prealgebra Problems12481632641282565121024N = number of solutions per problem51015202530% Problems Solved7.137.589.3411.3813.6515.1216.3017.3617.9118.3818.507.329.5912.4415.1018.2820.6422.9625.2626.9128.5429.957.2210.0412.9116.0918.7221.0023.3024.4824.8725.1324.81Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-7b (on PRM800K)Accuracy on Precalculus Problems12481632641282565121024N = number of solutions per problem51015202530% Problems Solved7.137.589.3411.3813.6515.1216.3017.3617.9118.3818.507.439.9312.6715.2518.0620.1222.0023.9725.8027.0527.097.3010.4313.7116.8020.0022.0323.8224.7025.5725.9125.90Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-7b + OPRM-34b (on PRM800K)Accuracy on Precalculus Problems12481632641282565121024N = number of solutions per problem1015202530% Problems Solved10.5311.3714.5218.3021.2723.3124.6125.6626.3226.5726.2710.8014.1917.7820.4223.0725.0726.9928.2329.3329.8929.8110.8514.3817.7320.3822.9925.0126.7528.2329.0029.0026.91Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-7b (on PRM800K)Accuracy on Precalculus Problems12481632641282565121024N = number of solutions per problem101520253035% Problems Solved10.5311.3714.5218.3021.2723.3124.6125.6626.3226.5726.2710.8615.0318.3421.4123.9326.6228.4429.2429.8130.2330.3510.8615.3318.8022.0124.7326.4828.1528.9029.3430.0730.94Majority VotingWeighted Voting w/ RMBest-of-N w/ RMSFT-34b + OPRM-34b (on PRM800K)Accuracy on Precalculus ProblemsO Pass@N Analysis for Different Math Questions

Figure 21: The increase of Pass@N metrics with the number of samples, divided by level and category.
As the number of samples increases, the Pass@N rate tends to improve. However, not all problems
will eventually yield a correct solution, meaning that even as N approaches infinity, the Pass@N
metric may not reach 100%.

Figure 21 shows the changes in Pass@N as the number of samples increases. As shown in the
figure, not all problems reach a 100% Pass@N rate, even with a larger N. Specifically, three key
observations can be made: First, Pass@N is strongly correlated with problem difficulty. As depicted
in Figure 21, for Level 1 problems, the Pass@N rate nearly saturates with more samples, while for
Level 5 problems, many instances still do not achieve a correct solution. Second, larger models (e.g.,
34b) perform significantly better on Pass@1 compared to smaller models (e.g., 7b). However, as
N increases, the advantage of the larger model diminishes on metrics like Pass@1024. Third, the
variations in Pass@N across different categories of math questions are different, which is expected,
as the difficulty of different types of math questions varies for large language models.

36

12481632641282565121024N = number of solutions per problem20406080100% Problems Solved41.8457.8972.5283.9391.4095.1097.2098.3698.8599.4499.8833.4147.2159.6071.7180.5887.5191.2793.7895.9297.1498.0518.0628.8441.9355.3468.4478.2485.5990.0892.9095.0296.2010.5117.2026.7537.1747.7557.8567.0374.2279.7583.9987.173.486.4811.1818.3227.8838.8349.8960.2669.6476.6981.75Level 1Level 2Level 3Level 4Level 5SFT-7b (on PRM800K)Pass@N on Problems by Level12481632641282565121024N = number of solutions per problem20406080100% Problems Solved52.9969.6582.4490.3894.7797.3898.6699.4199.89100.00100.0043.4958.3870.7980.1086.2989.8492.4694.9196.9398.2698.9026.5239.2252.9966.0676.0083.8188.9991.5593.6695.4796.9515.2024.1834.2844.6155.3965.3873.5380.0385.0288.1490.325.349.6416.0924.9734.9245.6655.6665.0872.4878.3682.71Level 1Level 2Level 3Level 4Level 5SFT-34b (on PRM800K)Pass@N on Problems by Level12481632641282565121024N = number of solutions per problem20406080100% Problems Solved27.7440.0152.3664.8174.9282.6488.0791.4293.9295.5596.7327.1340.1855.0768.5980.6989.5094.8097.5998.7699.1899.5013.5221.3330.1939.0948.9360.6171.4181.0088.8195.2698.6911.6519.1229.6341.8053.2164.4974.1782.3088.4193.1196.416.6210.7317.0524.7133.1242.3452.0961.0468.5874.4278.497.1911.6416.7824.0131.9039.5547.0455.2962.5968.7473.7914.0221.0529.6940.5952.1562.1371.9578.4883.8886.9489.50PrealgebraCounting & ProbabilityAlgebraNumber TheoryGeometryIntermediate AlgebraPrecalculusSFT-7b (on PRM800K)Pass@N on Problems by Category12481632641282565121024N = number of solutions per problem20406080100% Problems Solved35.5148.8060.9072.0281.0487.6792.1194.5996.1297.0997.7037.5552.5366.9379.3988.4994.3496.7398.2299.1099.7699.9819.4828.6837.4848.6862.2775.6786.1692.6596.3698.4699.7518.5729.6442.9654.3365.7774.5782.9989.5693.6695.9696.928.3813.4620.6428.1635.8944.3954.5163.1970.3275.1978.4311.1217.0823.6429.7335.4842.4749.8757.9065.8773.3480.5319.5428.8839.5149.2658.0366.0772.2678.7283.7988.5892.11PrealgebraCounting & ProbabilityAlgebraNumber TheoryGeometryPrecalculusIntermediate AlgebraSFT-34b (on PRM800K)Pass@N on Problems by CategoryP Case Studies for OPRM

Figure 22: Case Study 1. The evaluator can help identify and reduce the confidence of hallucinations
caused by misleading information.

Figure 23: Case Study 2. The evaluator can assist in reducing the confidence of solutions that misuse
mathematical theorems or formulas.

We have included more case studies in Figures 22 and 23. We find that evaluator can help generalize
to harder ones in the following ways:

• The evaluator can help identify and reduce the confidence of hallucinations caused by misleading
information in problems. As demonstrated in Case Study 1, the solution selected by majority voting
with an answer of 36 is misled by the different units of measurement in the problem (2.5 hours
and 90 seconds), resulting in an incorrect solution. Then, the ORPM model successfully gives this
solution a low score.

• The evaluator can assist in reducing the confidence of solutions that misuse mathematical theorems.
In Case Study 2, the majority solution incorrectly applies the theorem "the sum of the exterior
angles of a polygon is 360°", leading to erroneous reasoning, and low confidence by the ORPM
model.

37

Case Study 1: A robot moving forward at a constant speed takes 2.5 hours to travel 1 kilometer. Moving forward at this same constant speed, it takes the robot 90 seconds to travel the length of a particular hallway. How many meters long is the hallway?  Level: 5 GT: 10Majority Solution:OPRM Solution:Voting V.S. Reward: {Answer: 36, Voting times: 139, OPRM reward: 11.3}, {Answer: 10, Voting times: 99, OPRM reward: 23.7}Let's call the number of meters in the hallway x.           0.97We know the robot moved 1 km in 2.5 hours.           0.97That means it moved x meters in 90 seconds.            0.28We can write that as the equation $\\frac{1}{2.5}=\\frac{x}{90}$.           0.11To solve for x, we can multiply both sides by 90 to get $36=x$.           0.07So the hallway is 36 meters long.           0.03# Answer36To solve this problem, I need to find the relationship between the robot's speed, distance, and time.                    0.98I can use the formula distance = speed * time to write an equation for each case.                    0.98Let s be the robot's speed in meters per hour.                    0.98Then the first case gives me 1000 = s * 2.5, or s = 1000 / 2.5 = 400 meters per hour.                    0.98The second case gives me the length of the hallway = s * 90 / 3600, or length = 400 * 90 / 3600 = 10 meters.                    0.79# Answer10#Reward#Reward#Final Reward: 6.7e-5#Final Reward: 0.74Case Study 2: In a regular polygon, the measure of an interior angle is 6.5 times the measure of an exterior angle. How many sides does the polygon have?                            Level: 4 GT: 15Majority Solution:OPRM Solution:Voting V.S. Reward: {Answer: 0, Voting times: 132, OPRM reward: 0.03}, {Answer: 15, Voting times: 94, OPRM reward: 17.3}So we know that the measure of an interior angle is 6.5 times the measure of an exterior angle.                      0.98And we know that the sum of the measures of the interior angles of a polygon is given by the formula (n-2)180, where n is the number of sides.               0.89Right. So we can say that 6.5(n-2)=(n-2)180.                0.22And that simplifies to 6.5=180, which is false.                0.12So there is no such polygon.                   0.03# Answer0I know that the sum of the interior angles of any polygon with n sides is (n-2) times 180 degrees, and the sum of the exterior angles is 360 degrees.                        0.98I can write an equation relating the interior and exterior angles of a regular polygon: (n-2) * 180 = 6.5 * 360, where n is the number of sides.                     0.84I can simplify this equation by dividing everything by 180: n - 2 = 6.5 * 2, which gives me n - 2 = 13.                              0.97I can add 2 to both sides to isolate n: n = 13 + 2, which gives me n = 15.                       0.95# Answer15#Reward#Reward#Final Reward: 6e-4#Final Reward: 0.75Q Step and Outcome ROC Curve

Model

71.7
63.5
74.4

64.3
80.4
79.8

Step ACC (%) Outcome ACC (%)

ORM-PRM800K-7B
ORM-PRM800K-7B
OPRM-PRM800K-7B

To assess the accuracy of reasoning step
judgments of different reward models, we
conducted additional experiments using
the PRM800K-test data, which includes
correctness annotations for each step, to
test our model’s ability to distinguish cor-
rect reasoning steps. We randomly se-
lected a portion of PRM800K-test data
to balance positive and negative samples.
The accuracy of the reasoning steps for the
three models is shown in Table 10. This
table demonstrates the effectiveness of our trained PRM, showing that PRM has a significantly
greater ability to distinguish steps compared to ORM. Additionally, in Figure 24, we present the
Step ROC curves of three models, where PRM and OPRM exhibit better step discrimination abilities
compared to ORM. However, it is important to note that a stronger ability to distinguish steps does
not necessarily indicate that the evaluator is more helpful for generation. We then also present the
Outcome ROC curves of three models on discriminating the final outcome. We collect data generated
on MATH500 test set from our 7B policy model. According to the final outcome and groundtruth, we
label each data and select a positive-negative balanced set to plot the Outcome ROC curves, where
OPRM exhibits better outcome discrimination abilities compared to ORM and PRM. The above table
also shows the effectiveness of OPRM on Outcome discrimination ability.

Table 10: The accuracy of the reasoning steps for differ-
ent models.

Figure 24: The Step and Outcome ROC Curve show the ability of discriminate the reasoning step
and the whole reasoning solution, illustrating OPRM’s effectiveness on identifying both steps and
outcomes.

38

0.00.20.40.60.81.0False Positive Rate0.00.20.40.60.81.0True Positive RateStep ROC CurveRandom Guess (AUC = 0.50)ORM-PRM800K (AUC = 0.55)PRM-PRM800K (AUC = 0.79)OPRM-PRM800K (AUC = 0.76)0.00.20.40.60.81.0False Positive Rate0.00.20.40.60.81.0True Positive RateOutcome ROC CurveRandom Guess (AUC = 0.50)ORM-PRM800K (AUC = 0.67)PRM-PRM800K (AUC = 0.62)OPRM-PRM800K (AUC = 0.71)R Few-Shot Prompt in In-Context Learning

We sample with temperature T = 0.9 for ICL-related experiments. We list our few-shot sample for
the In-context learning experiments in Table 1.

4-shot example for PRM800K Full ICL:

Problem:
In right triangle $PQR$, we have $\\angle Q = \\angle R$ and $PR = 6\\sqrt{2}$.

What is the area of $\\triangle PQR$?

Solution:
I see that this is an isosceles right triangle, since $\\angle Q = \\angle R =

45^\\circ$.\n\nThat means that $PQ = PR = 6\\sqrt{2}$.\n\nTo find the area of a

triangle, I need to multiply the base and the height and divide by 2.\n\nIn
this case, the base and the height are both $6\\sqrt{2}$, so the area is $(6\\
sqrt{2})(6\\sqrt{2})/2$.\n\nI can simplify this expression by using the
property that $\\sqrt{a}\\sqrt{b} = \\sqrt{ab}$.\n\nSo, the area is $(\\sqrt
{2}\\sqrt{2})(6\\cdot 6)/2 = 2\\cdot 36/2 = 36$.

Final Answer: The final answer is $36$. I hope it is correct.

Problem:
The length of the longer side of rectangle $R$ is $10$ percent more than the length

of a side of square $S.$ The length of the shorter side of rectangle $R$ is

$10$ percent less than the length of a side of square $S.$ What is the ratio of
the area of rectangle $R$ to the area of square $S?$ Express your answer as a

common fraction.

Solution:
Let the side of the square be $s$.\n\nThen the longer side of the rectangle is $1.1
s$.\n\nAnd the shorter side of the rectangle is $0.9s$.\n\nThe area of the
square is $s^2$.\n\nThe area of the rectangle is $1.1s * 0.9s = 0.99s^2$.\n\nSo

the ratio of the area of rectangle $R$ to the area of square $S$ is $\\frac

{0.99s^2}{s^2} = 0.99$.\n\nTherefore, the answer is $\\boxed{\\frac{99}{100}}$.

Final Answer: The final answer is $\\frac{99}{100}$. I hope it is correct.

Problem:
Compute $\\sin(-60^\\circ)$.

Solution:
I know the sine function is periodic with period $360^\\circ$, so $\\sin(-60^\\circ
) = \\sin(-60^\\circ + 360^\\circ) = \\sin(300^\\circ)$.\n\nI also know that
the sine function is symmetrical about the origin, so $\\sin(-x) = -\\sin(x)$
for any angle $x$.\n\nTherefore, $\\sin(-60^\\circ) = -\\sin(60^\\circ)$.\n\nTo
find $\\sin(60^\\circ)$, I can use the special right triangle with angles $30
^\\circ$, $60^\\circ$, and $90^\\circ$ and sides in the ratio $1:\\sqrt{3}:2$.\
n\nThe sine of an angle is the ratio of the opposite side to the hypotenuse, so
$\\sin(60^\\circ) = \\frac{\\sqrt{3}}{2}$.\n\nTherefore, $\\sin(-60^\\circ) =

-\\frac{\\sqrt{3}}{2}$.

Final Answer: The final answer is $-\\frac{\\sqrt{3}}{2}$. I hope it is correct.

Problem:
Simplify $\\sin (x - y) \\cos y + \\cos (x - y) \\sin y.$

Solution:
’Ok, so we have $\\sin (x - y) \\cos y + \\cos (x - y) \\sin y$\n\nNotice that $\\
sin (x - y) \\cos y + \\cos (x - y) \\sin y$ is in the form of $\\sin(\\alpha)
\\cos(\\beta)+\\cos(\\alpha)\\sin(\\beta)$.\n\nYes, and we know that $\\sin(\\
alpha)\\cos(\\beta)+\\cos(\\alpha)\\sin(\\beta)=\\sin(\\alpha+\\beta)$\n\nSo, $
\\sin (x - y) \\cos y + \\cos (x - y) \\sin y=\\sin((x-y)+y)$\n\n$\\sin((x-y)+y
)=\\sin(x)$

Final Answer: The final answer is $\\sinx$. I hope it is correct.

39

4-shot example for PRM800K Hard ICL:

Problem:
In right triangle $PQR$, we have $\\angle Q = \\angle R$ and $PR = 6\\sqrt{2}$.

What is the area of $\\triangle PQR$?

Solution:
I see that this is an isosceles right triangle, since $\\angle Q = \\angle R =

45^\\circ$.\n\nThat means that $PQ = PR = 6\\sqrt{2}$.\n\nTo find the area of a

triangle, I need to multiply the base and the height and divide by 2.\n\nIn
this case, the base and the height are both $6\\sqrt{2}$, so the area is $(6\\
sqrt{2})(6\\sqrt{2})/2$.\n\nI can simplify this expression by using the
property that $\\sqrt{a}\\sqrt{b} = \\sqrt{ab}$.\n\nSo, the area is $(\\sqrt
{2}\\sqrt{2})(6\\cdot 6)/2 = 2\\cdot 36/2 = 36$.

Final Answer: The final answer is $36$. I hope it is correct.

Problem:
The length of the longer side of rectangle $R$ is $10$ percent more than the length

of a side of square $S.$ The length of the shorter side of rectangle $R$ is

$10$ percent less than the length of a side of square $S.$ What is the ratio of
the area of rectangle $R$ to the area of square $S?$ Express your answer as a

common fraction.

Solution:
Let the side of the square be $s$.\n\nThen the longer side of the rectangle is $1.1
s$.\n\nAnd the shorter side of the rectangle is $0.9s$.\n\nThe area of the
square is $s^2$.\n\nThe area of the rectangle is $1.1s * 0.9s = 0.99s^2$.\n\nSo

the ratio of the area of rectangle $R$ to the area of square $S$ is $\\frac

{0.99s^2}{s^2} = 0.99$.\n\nTherefore, the answer is $\\boxed{\\frac{99}{100}}$.

Final Answer: The final answer is $\\frac{99}{100}$. I hope it is correct.

Problem:
Suppose that $y^3$ varies inversely with $\\sqrt[3]{z}$. If $y=2$ when $z=1$, find
the value of $z$ when $y=4$. Express your answer in simplest fractional form.

Solution:
I know that inverse variation means that the product of the two quantities is

constant, so I can write an equation of the form $y^3\\cdot\\sqrt[3]{z}=k$,
where $k$ is some constant.\n\nTo find $k$, I can plug in the given values of
$y$ and $z$: $2^3\\cdot\\sqrt[3]{1}=k$, which simplifies to $8=k$.\n\nNow I can

use this equation to find $z$ when $y=4$: $4^3\\cdot\\sqrt[3]{z}=8$, which
implies that $\\sqrt[3]{z}=\\frac{8}{64}=\\frac{1}{8}$.\n\nTo get rid of the
cube root, I can cube both sides: $z=\\left(\\frac{1}{8}\\right)^3=\\frac
{1}{512}$.

Final Answer: The final answer is $\\frac{1}{512}$. I hope it is correct.

Problem:
Let $d$ be a positive number such that when $109$ is divided by $d$, the remainder

is $4.$ Compute the sum of all possible two-digit values of $d$.

Solution:
This problem involves finding the divisors of a given number, as well as using the
concept of remainders.\n\nOne way to approach this is to write the division as
a quotient and a remainder, like this: $109 = qd + 4$, where $q$ is the
quotient.\n\nThen, I can rearrange this equation to get $105 = qd$, which means

that $d$ is a divisor of $105$.\n\nNow, I need to find all the two-digit

divisors of $105$.\n\nI can use prime factorization to help me do this.\n\nI
notice that $105 = 3 \\times 5 \\times 7$, which are all prime numbers.\n\
nTherefore, any divisor of $105$ must be a product of some combination of these
three factors.\n\nThe possible products are $1, 3, 5, 7, 15, 21, 35, 105$.\n\

nHowever, not all of these are two-digit numbers. Only $15, 21, 35$ are.\n\
nThese are the only possible values of $d$ that satisfy the given condition.\n\
nTo find their sum, I just add them up: $15 + 21 + 35 = 71$.

Final Answer: The final answer is $71$. I hope it is correct.

40

4-shot example for PRM800K Easy-to-Hard ICL:

Problem:
How many 4-letter words with at least one consonant can be constructed from the
letters $A$, $B$, $C$, $D$, and $E$? (Note that $B$, $C$, and $D$ are
consonants, any word is valid, not just English language words, and letters may

be used more than once.)

Solution:
To count the number of 4-letter words with at least one consonant, I can use the
complement principle and subtract the number of 4-letter words with no
consonants from the total number of 4-letter words.\n\nThe total number of 4-
letter words is simply $5^4$, since each letter has 5 choices and the order
matters.\n\nThe number of 4-letter words with no consonants is just the number
of ways to choose 4 letters from the 2 vowels $A$ and $E$, which is $2^4$.\n\
nTherefore, the number of 4-letter words with at least one consonant is $5^4 -
2^4$. 625 - 16 = 609

Final Answer: The final answer is $609$. I hope it is correct.

Problem:
Compute the integer $k > 2$ for which\n\\[\\log_{10} (k - 2)! + \\log_{10} (k - 1)!

+ 2 = 2 \\log_{10} k!.\\]

Solution:
I recognize that this equation involves logarithms of factorials, which are

products of consecutive integers.\n\nI also know that logarithms have some
useful properties, such as $\\log_{10} a + \\log_{10} b = \\log_{10} (ab)$ and
$c \\log_{10} d = \\log_{10} d^c$.\n\nUsing these properties, I can simplify
the equation as follows: \\[\\log_{10} (k - 2)! + \\log_{10} (k - 1)! + 2 = 2
\\log_{10} k! \\implies \\log_{10} \\left[ (k - 2)! (k - 1)! 100 \\right] = \\
log_{10} (k!)^2.\\]\n\nSince the bases of the logarithms are equal, I can
conclude that the arguments must also be equal, i.e., \\[(k - 2)! (k - 1)! 100
= (k!)^2.\\]\n\nNow I have a simpler equation to solve for $k$.\n\nI notice
that the left-hand side has a factor of $(k - 2)!$, which is also a factor of $
(k - 1)!$ and $(k!)^2$.\n\nTherefore, I can divide both sides by $(k - 2)!$ to
get \\[(k - 1)! 100 = k! (k - 1)!\\]\n\nSimplifying further, I get \\[100 = k
!.\\]\n\nThis means that $k$ is the smallest integer such that its factorial is

greater than or equal to 100.\n\nI can try some values of $k$ to find the

right one.\n\nIf $k = 3$, then $k! = 3! = 6 < 100$.\n\nIf $k = 4$, then $k! =
4! = 24 < 100$.\n\nIf $k = 5$, then $k! = 5! = 120 > 100$.\n\nTherefore, $k = 5
$ is the smallest integer that satisfies the equation.

Final Answer: The final answer is $5$. I hope it is correct.

Problem:
What is the largest perfect square factor of 1512?

Solution:
I know that 1512 is divisible by 2, so I can try to factor out powers of 2 until I
get an odd number.\n\nDividing by 2 once, I get 756.\n\nDividing by 2 again, I
get 378.\n\nDividing by 2 one more time, I get 189.\n\nNow 189 is odd, so I can
’t divide by 2 anymore.\n\nI notice that 189 is divisible by 3, so I can try to

factor out powers of 3.\n\nDividing by 3 once, I get 63.\n\nDividing by 3

again, I get 21.\n\nDividing by 3 one more time, I get 7.\n\nNow 7 is prime, so

I can’t factor it anymore.\n\nI have factored 1512 as 2^3 * 3^3 * 7.\n\nTo
find the largest perfect square factor, I need to use the highest even powers
of each prime factor.\n\nThe highest even power of 2 that I have is 2^2.\n\nThe

highest even power of 3 that I have is 3^2.\n\nThe highest even power of 7

that I have is 7^0, which is just 1.\n\nSo the largest perfect square factor is

2^2 * 3^2 * 7^0, which is 36.

Final Answer: The final answer is $36$. I hope it is correct.

Problem:
If $3p+4q=8$ and $4p+3q=13$, what is $q$ equal to?

Solution:

41

I notice that the coefficients of $p$ and $q$ are reversed in the two equations, so

I wonder if I can use elimination to get rid of one of them.\n\nI try

multiplying the first equation by 4 and the second equation by 3, and then
subtracting them to eliminate $p$.\n\nI get $12p+16q-12p-9q=32-39$, which
simplifies to $7q=-7$.\n\nDividing both sides by 7, I get $q=-1$.

Final Answer: The final answer is $-1$. I hope it is correct.

42

4-shot example for MetaMath Full ICL:

Problem:
In right triangle $PQR$, we have $\\angle Q = \\angle R$ and $PR = 6\\sqrt{2}$.

What is the area of $\\triangle PQR$?

Solution:
Since $\\angle Q = \\angle R$, we have $\\triangle PQR$ is an isosceles right

triangle.\nTherefore, $PQ = PR = 6\\sqrt{2}$.\nThe area of $\\triangle PQR$ is
$\\frac{1}{2} \\cdot PQ \\cdot PR = \\frac{1}{2} \\cdot 6\\sqrt{2} \\cdot 6\\
sqrt{2} = \\boxed{36}$.

Final Answer: The final answer is $36$. I hope it is correct.

Problem:
The length of the longer side of rectangle $R$ is $10$ percent more than the length

of a side of square $S.$ The length of the shorter side of rectangle $R$ is

$10$ percent less than the length of a side of square $S.$ What is the ratio of
the area of rectangle $R$ to the area of square $S?$ Express your answer as a

common fraction.

Solution:
Let the side length of square $S$ be $x$.\nThen the longer side of rectangle $R$ is

$1.1x$ and the shorter side is $0.9x$.\nThe area of square $S$ is $x^2$ and

the area of rectangle $R$ is $(1.1x)(0.9x) = 0.99x^2$.\nTherefore, the ratio of
the area of rectangle $R$ to the area of square $S$ is $\\frac{0.99x^2}{x^2} =
\\boxed{\\frac{99}{100}}$.

Final Answer: The final answer is $\\frac{99}{100}$. I hope it is correct.

Problem:
Compute the integer $k > 2$ for which\n\\[\\log_{10} (k - 2)! + \\log_{10} (k - 1)!

+ 2 = 2 \\log_{10} k!.\\]

Solution:
Using the properties of logarithms, we can rewrite the equation as\n\\[\\log_{10}

((k-2)! \\cdot (k-1)!) + 2 = \\log_{10} (k!)^2.\\]\nThen, using the fact that $
\\log_{a} b + \\log_{a} c = \\log_{a} (bc)$, we have\n\\[\\log_{10} ((k-2)! \\
cdot (k-1)! \\cdot 100) = \\log_{10} (k!)^2.\\]\nFinally, using the fact that $
\\log_{a} b = \\log_{a} c$ if and only if $b = c$, we have\n\\[(k-2)! \\cdot (k
-1)! \\cdot 100 = (k!)^2.\\]\nSimplifying, we have\n\\[(k-2)! \\cdot (k-1)! \\
cdot 100 = (k \\cdot (k-1) \\cdot (k-2))!.\\]\nCancelling out the common
factors, we have\n\\[100 = k \\cdot (k-1) \\cdot (k-2).\\]\nFactoring, we find
that $100 = 5 \\cdot 5 \\cdot 4 = 2^2 \\cdot 5^2$.\nSince $k > 2$, we see that
$k = \\boxed{5}$.

Final Answer: The final answer is $5$. I hope it is correct.

Problem:
If $3p+4q=8$ and $4p+3q=13$, what is $q$ equal to?

Solution:
We can solve this system of equations using the method of elimination.\nMultiplying

the first equation by 3 and the second equation by 4 gives us:\n$9p+12q=24$\
n$16p+12q=52$\nWe can subtract the first equation from the second equation to
eliminate the q term:\n$16p+12q-(9p+12q)=52-24$\n$7p=28$\nDividing both sides
by 7 gives us:\n$p=4$\nSubstituting this value of p into either of the original

equations gives us:\n$4(4)+3q=13$\n$16+3q=13$\n$3q=-3$\n$q=\\boxed{-1}$

Final Answer: The final answer is $-1$. I hope it is correct.

43

4-shot example for MetaMath Hard ICL:

Problem:
In right triangle $PQR$, we have $\\angle Q = \\angle R$ and $PR = 6\\sqrt{2}$.

What is the area of $\\triangle PQR$?

Solution:
Since $\\angle Q = \\angle R$, we have $\\triangle PQR$ is an isosceles right

triangle.\nTherefore, $PQ = PR = 6\\sqrt{2}$.\nThe area of $\\triangle PQR$ is
$\\frac{1}{2} \\cdot PQ \\cdot PR = \\frac{1}{2} \\cdot 6\\sqrt{2} \\cdot 6\\
sqrt{2} = \\boxed{36}$.

Final Answer: The final answer is $36$. I hope it is correct.

Problem:
The length of the longer side of rectangle $R$ is $10$ percent more than the length

of a side of square $S.$ The length of the shorter side of rectangle $R$ is

$10$ percent less than the length of a side of square $S.$ What is the ratio of
the area of rectangle $R$ to the area of square $S?$ Express your answer as a

common fraction.

Solution:
Let the side length of square $S$ be $x$.\nThen the longer side of rectangle $R$ is

$1.1x$ and the shorter side is $0.9x$.\nThe area of square $S$ is $x^2$ and

the area of rectangle $R$ is $(1.1x)(0.9x) = 0.99x^2$.\nTherefore, the ratio of
the area of rectangle $R$ to the area of square $S$ is $\\frac{0.99x^2}{x^2} =
\\boxed{\\frac{99}{100}}$.

Final Answer: The final answer is $\\frac{99}{100}$. I hope it is correct.

Problem:
Suppose that $y^3$ varies inversely with $\\sqrt[3]{z}$. If $y=2$ when $z=1$, find
the value of $z$ when $y=4$. Express your answer in simplest fractional form.

Solution:
Since $y^3$ varies inversely with $\\sqrt[3]{z}$, we can write the equation as $y

^3\\sqrt[3]{z}=k$, where $k$ is a constant.\nWe are given that $y=2$ when $z=1$
, so we can substitute these values into the equation to solve for $k$:\n$2^3\\
sqrt[3]{1}=k$\n$8\\cdot 1=k$\n$k=8$\nNow we can use this value of $k$ to find
$z$ when $y=4$:\n$4^3\\sqrt[3]{z}=8$\n$64\\sqrt[3]{z}=8$\n$\\sqrt[3]{z}=\\frac
{8}{64}=\\frac{1}{8}$\n$z=\\left(\\frac{1}{8}\\right)^3=\\frac{1}{512}$\nSo
when $y=4$, $z=\\boxed{\\frac{1}{512}}$.

Final Answer: The final answer is $\\frac{1}{512}$. I hope it is correct.

Problem:
Let $d$ be a positive number such that when $109$ is divided by $d$, the remainder

is $4.$ Compute the sum of all possible two-digit values of $d$.

Solution:
If $109$ is divided by $d$ and the remainder is $4$, then $d$ divides $109-4=105$.\
nThe prime factorization of $105$ is $3\\cdot5\\cdot7$.\nSince $d$ must be a
factor of $105$, the possible values of $d$ are $1, 3, 5, 7, 15, 21, 35, 105$.\
nOut of these, only the two-digit values are $15, 21, 35$.\nThe sum of these
values is $15+21+35=\\boxed{71}$.

Final Answer: The final answer is $71$. I hope it is correct.

44

4-shot example for MetaMath Easy-to-Hard ICL:

Problem:
How many 4-letter words with at least one consonant can be constructed from the
letters $A$, $B$, $C$, $D$, and $E$? (Note that $B$, $C$, and $D$ are
consonants, any word is valid, not just English language words, and letters may

be used more than once.)

Solution:
There are a total of $5^4 = 625$ possible 4-letter words that can be constructed
from the given letters.\nTo count the number of words with no consonants, we
can use the principle of complementary counting.\nSince there are 2 vowels ($A$
and $E$) and 3 consonants ($B$, $C$, and $D$), there are $2^4 = 16$ words with
only vowels.\nTherefore, the number of words with at least one consonant is

$625 - 16 = \\boxed{609}$.

Final Answer: The final answer is $609$. I hope it is correct.

Problem:
Compute the integer $k > 2$ for which\n\\[\\log_{10} (k - 2)! + \\log_{10} (k - 1)!

+ 2 = 2 \\log_{10} k!.\\]

Solution:
Using the properties of logarithms, we can rewrite the equation as\n\\[\\log_{10}

((k-2)! \\cdot (k-1)!) + 2 = \\log_{10} (k!)^2.\\]\nThen, using the fact that $
\\log_{a} b + \\log_{a} c = \\log_{a} (bc)$, we have\n\\[\\log_{10} ((k-2)! \\
cdot (k-1)! \\cdot 100) = \\log_{10} (k!)^2.\\]\nFinally, using the fact that $
\\log_{a} b = \\log_{a} c$ if and only if $b = c$, we have\n\\[(k-2)! \\cdot (k
-1)! \\cdot 100 = (k!)^2.\\]\nSimplifying, we have\n\\[(k-2)! \\cdot (k-1)! \\
cdot 100 = (k \\cdot (k-1) \\cdot (k-2))!.\\]\nCancelling out the common
factors, we have\n\\[100 = k \\cdot (k-1) \\cdot (k-2).\\]\nFactoring, we find
that $100 = 5 \\cdot 5 \\cdot 4 = 2^2 \\cdot 5^2$.\nSince $k > 2$, we see that
$k = \\boxed{5}$.

Final Answer: The final answer is $5$. I hope it is correct.

Problem:
If $n$ is $1$ less than a multiple of $50$, then what is the remainder when $n^2+2n

+3$ is divided by $50$?

Solution:
Since $n$ is $1$ less than a multiple of $50$, we can write $n=50k-1$ for some

integer $k$.\nThen \\[n^2+2n+3=(50k-1)^2+2(50k-1)+3=2500k^2-96k+2,\\] so the
remainder when $n^2+2n+3$ is divided by $50$ is $\\boxed{2}$.

Final Answer: The final answer is $2$. I hope it is correct.

Problem:
If $3p+4q=8$ and $4p+3q=13$, what is $q$ equal to?

Solution:
We can solve this system of equations using the method of elimination.\nMultiplying

the first equation by 3 and the second equation by 4 gives us:\n$9p+12q=24$\
n$16p+12q=52$\nWe can subtract the first equation from the second equation to
eliminate the q term:\n$16p+12q-(9p+12q)=52-24$\n$7p=28$\nDividing both sides
by 7 gives us:\n$p=4$\nSubstituting this value of p into either of the original

equations gives us:\n$4(4)+3q=13$\n$16+3q=13$\n$3q=-3$\n$q=\\boxed{-1}$

Final Answer: The final answer is $-1$. I hope it is correct.

45

