4
2
0
2

t
c
O
4

]

G
L
.
s
c
[

5
v
5
7
6
0
0
.
5
0
4
2
:
v
i
X
r
a

Self-Play Preference Optimization for Language Model
Alignment

Yue Wu∗† Zhiqing Sun∗‡ Huizhuo Yuan∗§ Kaixuan Ji¶ Yiming Yang‖ Quanquan Gu∗∗

Abstract

Standard reinforcement learning from human feedback (RLHF) approaches relying on paramet-
ric models like the Bradley-Terry model fall short in capturing the intransitivity and irrationality in
human preferences. Recent advancements suggest that directly working with preference probabili-
ties can yield a more accurate reflection of human preferences, enabling more flexible and accurate
language model alignment. In this paper, we propose a self-play-based method for language model
alignment, which treats the problem as a constant-sum two-player game aimed at identifying
the Nash equilibrium policy. Our approach, dubbed Self-Play Preference Optimization (SPPO),
utilizes iterative policy updates to provably approximate the Nash equilibrium. Additionally,
we propose a new SPPO objective which is both strongly motivated by theory and is simple
and effective in practice. In our experiments, using only 60k prompts (without responses) from
the UltraFeedback dataset and without any prompt augmentation, by leveraging a pre-trained
preference model PairRM with only 0.4B parameters, SPPO can obtain a model from fine-tuning
Mistral-7B-Instruct-v0.2 that achieves the state-of-the-art length-controlled win-rate of 28.53%
against GPT-4-Turbo on AlpacaEval 2.0. It also outperforms the (iterative) DPO and IPO on
MT-Bench, Arena-Hard, and the Open LLM Leaderboard. Starting from a stronger base model
Llama-3-8B-Instruct, we are able to achieve a length-controlled win rate of 38.77%. Notably, the
strong performance of SPPO is achieved without additional external supervision (e.g., responses,
preferences, etc.)
from GPT-4 or other stronger language models. Codes are available at
https://github.com/uclaml/SPPO.

1

Introduction

Large Language Models (LLMs) (e.g., Ouyang et al., 2022; OpenAI et al., 2023), have shown
remarkable capabilities in producing human-like text, fielding questions, and coding. Despite

∗Equal contribution
†Department of Computer Science, University of California, Los Angeles, Los Angeles, CA 90095; e-mail:

ywu@cs.ucla.edu

‡Language Technologies Institute, Carnegie Mellon University, Pittsburgh, PA 15213; e-mail: zhiqings@cs.cmu.edu
§Department of Computer Science, University of California, Los Angeles, Los Angeles, CA 90095; e-mail:

hzyuan@cs.ucla.edu

¶Department of Computer Science, University of California, Los Angeles, Los Angeles, CA 90095; e-mail:

kauxuanji@cs.ucla.edu

‖Language Technologies Institute & Machine Learning Department, Carnegie Mellon University, Pittsburgh, PA

15213; e-mail: yiming@cs.cmu.edu

∗∗Department of Computer Science, University of California, Los Angeles, Los Angeles, CA 90095; e-mail:

qgu@cs.ucla.edu

1

 
 
 
 
 
 
their advancements, these models encounter challenges in tasks requiring high levels of reliability,
safety, and ethical alignment. To address these challenges, Reinforcement Learning from Human
Feedback (RLHF), also known as Preference-based Reinforcement Learning (PbRL), presents a
promising solution. This framework for policy optimization, highlighted in works by Christiano et al.
(2017) and recently in Ouyang et al. (2022), has led to significant empirical success in fine-tuning
instruction-following LLMs, making them more aligned with human preferences and thus more
helpful.

Most existing approaches to RLHF rely on either explicit or implicit reward models. Taking
InstructGPT (Ouyang et al., 2022) as an example, a reference policy πref is first established,
typically from supervised pre-training or instruction-based (supervised) fine-tuning. An explicit
reward function is obtained by training a reward model based on human preference feedback data,
employing the Bradley-Terry (BT) model (Bradley and Terry, 1952). Subsequently, reinforcement
learning algorithms such as Proximal Policy Optimization (Schulman et al., 2017, PPO) are used to
fine-tune the reference LLM πref by maximizing the expected reward function. The reward model
provides a “reward score” r(y; x) for the given response y and prompt x, approximately reflecting how
humans value these responses. More recently, methods like Direct Preference Optimization (Rafailov
et al., 2024b, DPO) have been introduced. These methods forgo the training of a separate reward
model but still fundamentally adhere to the reward maximization objective and are determined by
parametric models such as the BT model.

These models presuppose a monotonous and transitive relationship among preferences for
different choices. However, empirical evidence suggests otherwise. For instance, Tversky (1969)
observed human decisions can be influenced by different factors and exhibit inconsistency. Such
observations indicate that human preferences do not always adhere to a single, value-based hierarchy
and can even appear irrational, such as exhibiting loops in preference relations. For LLMs, another
motivating evidence is that Munos et al. (2023) has empirically shown that directly predicting the
pairwise preference can achieve higher accuracy than predicting the preference via a BT-based
reward model.

To address the inconsistency in human preference, researchers have proposed to work directly
with the preference probability and design algorithms that can more flexibly represent human
preferences (Lou et al., 2022; Wu et al., 2023) in the ranking or bandit setting. Recently, an
emerging line of work (Wang et al., 2024; Munos et al., 2023; Swamy et al., 2024) also proposed to
study RLHF for LLMs under such general preference P(y ≻ y′|x), where y and y′ are two different
responses and x is prompt. The goal is to identify the Nash equilibrium or von Neumann winner of
the two-player constant-sum game

(π∗, π∗) = arg max

π

(cid:104)

Ex∼X

min
π′

Ey∼π(·|x),y′∼π′(·|x)

(cid:2)P(y ≻ y′|x)(cid:3)(cid:105)

,

where each player is an LLM that outputs responses and aims to maximize its probability of being
preferred over its opponent.

Independent from our work, Swamy et al. (2024) proposed Self-play Preference Optimization
(SPO)1 for the same (unregularized) two-player constant-sum game. They provide a general reduction
of preference optimization to no-regret online learning for the multi-step Markov Decision Process.
When constrained to the bandit setting for LLMs, their proposed algorithmic framework reduces to

1The SPO framework does not pertain to the efficient fine-tuning of LLMs. Our Self-Play Preference Optimization
(SPPO) focuses on LLM alignment and was developed independently. To distinguish it from the SPO framework, we
use the abbreviation SPPO.

2

the famous Hedge algorithm (Freund and Schapire, 1997), which admits the exponential update
rule as described in (4.1). To approximately solve the exponential update, Swamy et al. (2024) then
proposed to employ typical policy optimization algorithms such as Proximal Policy Optimization
(PPO) (Schulman et al., 2017) or Soft Actor-Critic (SAC) (Haarnoja et al., 2018) to maximize the
win rate against the reference policy and evaluated the performance of their self-play algorithms
in robotic and game tasks. However, it typically requires more effort to apply PPO or SAC to
large-scale fine-tuning of LLM and make them work stably. Therefore, it remains unclear how their
self-play framework can be applied to large-scale language model alignment efficiently.

In this paper, motivated by these developments mentioned above, we propose a new self-play
algorithm that (1) enjoys provable guarantees to solve the two-player constant-sum game; and (2)
can scale up to large-scale efficient fine-tuning of large language models. In detail, we formulate
the RLHF problem as a constant-sum two-player game. Our objective is to identify the Nash
equilibrium policy, which consistently provides preferred responses over any other policy on average.
To identify the Nash equilibrium policy approximately, we adopt the classic online adaptive algorithm
with multiplicative weights (Freund and Schapire, 1999) as a high-level framework that solves the
two-player game. Further, each step of the high-level framework can be approximated by a self-play
mechanism, where in each round the policy is playing against itself in the previous round by
fine-tuning it on synthetic data that are generated by the policy and annotated by the preference
model.

Our contributions are highlighted as follows:

• Starting from the exponential weight update algorithm which provably converges to the Nash equi-
librium of the two-player constant-sum game, we propose the Self-Play Preference Optimization
(SPPO) algorithm for large language model alignment. The algorithm converges to an approximate
Nash equilibrium provably and admits a simple form of loss function for easy optimization.

• Unlike the symmetric pairwise loss such as DPO and Identity Preference Optimization (IPO) (Azar
et al., 2023), we propose a new optimization objective that does not rely on pairwise comparisons.
The new loss objective (4.4), initially driven by game-theoretical concepts, turns out strongly
motivated by the policy gradient theory and implicitly encourages the LLM to learn a token-level
optimal value function.

• Empirically, SPPO significantly enhances the well-aligned Mistral-7B-Instruct-v0.2 and Llama-3-
8B-Instruct model, achieving an increase of over 11% on the length-controlled win rate against
GPT-4-Turbo on the AlpacaEval 2.0 (Dubois et al., 2024a) test set. Additionally, SPPO exhibits
strong generalist abilities across different tasks, including MT-Bench, the Open LLM Leaderboard,
and the more recent, more challenging benchmark, Arena-Hard. Unlike iterative DPO/IPO, which
tends to show performance decay on other benchmarks when optimized towards the PairRM
score, SPPO’s performance gain is consistent. Notably, all the strong performances are achieved
without external supervision (e.g., responses, preferences, etc.) from GPT-4 or other stronger
language models.

Concurrent to our work, several studies, including Direct Nash Optimization (Rosset et al.,
2024) and REBEL (Gao et al., 2024) have also explored using either cross-entropy loss or square
loss minimization to approximate the exponential update. Specifically, they used the same trick
proposed in DPO (Rafailov et al., 2024b) to cancel out the log-partition factor and directly regress
on the win-rate difference. However, it is shown theoretically and empirically by Pal et al. (2024)

3

that the pairwise loss may only drive the relative likelihood gap to be large, but may not necessarily
drive up the likelihood of the preferred responses. Our method instead has a deeper connection to
the policy gradient theory and can effectively match the likelihood of the response to its win rate.

2 Related Work

RLHF with Explicit/Implicit Reward Model Originally, reinforcement learning from human
feedback (RLHF) was proposed by Christiano et al. (2017) as a methodology that first learns a reward
model reflecting human preferences and then uses reinforcement learning algorithms to maximize
the reward. This methodology is applied by Ouyang et al. (2022) to fine-tune instruction-following
large language models and leads to the popular ChatGPT.

The reward model in the works mentioned above assumes a parametric model such as the
Bradley-Terry model (Bradley and Terry, 1952), which assigns a “score” representing how preferred
a given response is. More recently, Rafailov et al. (2024b) proposed to instead directly solve
the closed-form solution of such a score implied by the Bradley-Terry model. The Direct Policy
Optimization (DPO) method is claimed to be more efficient and stable, yet, still implicitly assumes
such a reward model that specifies the “score”. In a similar spirit, Zhao et al. (2023) proposed to
calibrate the score so that the score of the winner in comparison has a margin over the score of
the loser, and induces a different SLic loss. Similarly, Ethayarajh et al. (2024) derived a different
loss function (called KTO) from the Kahneman-Tversky human utility function, which implicitly
denotes a score of the given response. Liu et al. (2023) proposed Rejection Sampling Optimization
(RSO) which utilizes a preference model to generate preference pairs with candidates sampled from
the optimal policy; then preference optimization is applied on the sampled preference pairs. Hong
et al. (2024) proposed Odds Ratio Preference Optimization (ORPO) algorithm that can perform
supervised fine-tuning and preference alignment in one training session without maintaining an
intermediate reference policy.

RLHF with General Preference Model Often, the human preference is not strictly transitive,
and cannot be sufficiently represented by a single numerical score. Azar et al. (2023) proposed a
general preference optimization objective based on the preference probability between a pair of
responses instead of a score of a single response. They further propose a learning objective based on
identity mapping of the preference probability called IPO (Preference Optimization with Identity
mapping), which aims to maximize the current policy’s expected winning probability over a given
reference policy. Munos et al. (2023) formulated the RLHF problem with general preference as a
two-player, constant-sum game, where each player is one policy that aims to maximize the probability
of its response being preferred against its opponent. They aim to identify the Nash equilibrium policy
of this game and propose a mirror-descent algorithm that guarantees the last-iterate convergence of
a policy with tabular representations2. Wang et al. (2024) proposed to identify the Nash equilibrium
policy for multi-step MDPs when a general preference model is present and shows that the problem
can be reduced to a two-player zero-sum Markov game.

Theory of RLHF There is also a line of research to analyze RLHF and provide its theoretical
guarantees. Zhu et al. (2023) studied the standard RLHF with separate reward-learning and

2Due to the tabular representation, computing the normalizing factor is prohibitive and the algorithm is approxi-

mately executed by sampling one token instead of a full response.

4

model-tuning and proposed a pessimistic reward-learning process that provably learns a linear
reward model. Wang et al. (2024) proposed a framework to reduce any RLHF problem with a
reward model to a reward-based standard RL problem. Additionally, they proposed to identify the
Nash equilibrium policy when a general preference model is present and show that the problem
can be reduced to a two-player zero-sum Markov game. Xiong et al. (2023) studied the reverse-KL
regularized contextual bandit for RLHF in different settings and proposed efficient algorithms with
finite-sample theoretical guarantees. Ye et al. (2024) studied the theoretical learnability of the
KL-regularized Nash-Learning from Human Feedback (NLHF) by considering both offline and online
settings and proposed provably efficient algorithms. Ji et al. (2024) proposed an active-query-based
proximal policy optimization algorithm with regret bounds and query complexity based on the
problem dimension and the sub-optimality gap.

Self-Play Fine-Tuning Most works mentioned above (Rafailov et al., 2024b; Zhao et al., 2023;
Azar et al., 2023; Ethayarajh et al., 2024) consider one single optimization procedure starting from
some reference policy. The same procedure may be applied repeatedly for multiple rounds in a
self-play manner. In each round, new data are generated by the policy obtained in the last round;
these new data are then used for training a new policy that can outperform the old policy.

The self-play fine-tuning can be applied to both scenarios with or without human preference
data. For example, Singh et al. (2023) proposed an Expectation-Maximization (EM) framework
where in each round, new data are generated and annotated with a reward score; the new policy is
obtained by fine-tuning the policy on the data with a high reward. Chen et al. (2024) proposed
a self-play framework to fine-tune the model in a supervised way. In each round, new preference
pairs are synthesized by labeling the policy-generated responses as losers and the human-generated
responses as winners. Then DPO is applied in each round to fine-tune another policy based on
these synthesized preference data. Yuan et al. (2024) proposed Self-Rewarding Language Models,
where the language model itself is used to annotate preference on its own responses. Iterative
DPO is applied to fine-tune language models on these annotated data. These works show iterative
fine-tuning can significantly improve the performance.

Swamy et al. (2024) considered a more general multi-step Markov Decision Process (MDP)
setting and proposed Self-play Preference Optimization (SPO), an RLHF framework that can
utilize any no-regret online learning algorithm for preference-based policy optimization. They then
instantiated their framework with Soft Policy Iteration as an idealized variant of their algorithm,
which reduces to the exponential weight update rule (4.1) when constrained to the bandit setting.
The main difference is that they focus on the multi-round Markov decision process (MDP) in robotic
and game tasks rather than on fine-tuning large language models and approximating the update
using policy optimization methods such as PPO.

Concurrent to our work, Rosset et al. (2024) proposed the Direct Nash Optimization (DNO)
algorithm based on the cross-entropy between the true and predicted win rate gaps, and provided
theoretical guarantees on the error of finite-sample approximation. However, their practical version
still utilizes the iterative-DPO framework as in Xu et al. (2023) with the DPO loss instead of their
derived DNO loss. Notably, in their experiments, they added the GPT-4 generated responses as
their “gold sample” into their fine-tuning data, and used GPT-4 as a judge to assign a numerical
score to each response for preference pair construction. In sharp contrast, our work does not require
the use of any strong external supervision besides a small-sized reward model. Another concurrent
work (Gao et al., 2024) proposed REBEL, an iterative fine-tuning framework via regressing the

5

relative reward. When applied to the preference setting, it results in a similar algorithm to our
algorithm SPPO, except that SPPO approximates the log-partition factor log Zπt(x) with a constant
η/2 while REBEL regresses on the win rate difference (so that log Zπt(x) is canceled). Additionally,
Calandriello et al. (2024) pointed out that optimizing the IPO loss (Azar et al., 2023) iteratively
with self-play generated data is equivalent to finding the Nash equilibrium of the two-player game,
and they proposed the IPO-MD algorithm based on this observation, which generates data with a
mixture policy similar to the Nash-MD algorithm.

3 Preliminaries

We consider the preference learning scenario as follows. Given a text sequence (commonly referred
to as prompt) x = [x1, x2, . . . ], two text sequences y = [y1, y2, . . . ] and y′ are generated as responses
to the prompt x. An autoregressive language model π given the prompt x can generate responses y
following the probability decomposition

π(y|x) =

N
(cid:89)

i=1

π(yi|x, y<i).

Given the prompt x and two responses y and y′, a preference oracle (either a human annotator or
a language model) will provide preference feedback o(y ≻ y′|x) ∈ {0, 1} indicating whether y is
preferred over y′. We denote P(y ≻ y′|x) = E[o(y ≻ y′|x)] as the probability of y “winning the
duel” over y′. The KL divergence of two probability distributions of density p and q is defined as
KL(p∥q) = Ey∼p(y)

(cid:105)

(cid:104)

.

log p(y)
q(y)

3.1 RLHF with Reward Models

Christiano et al. (2017) first learn a reward function r(y; x) following the Bradley-Terry model (Bradley
and Terry, 1952). For a prompt-response-response triplet (x, y, y′), the Bradley-Terry model specifies
the probability of y being chosen over y as

P(y ≻ y′|x) =

exp(r(y; x))
exp(r(y; x)) + exp(r(y′; x))

= σ(cid:0)r(y; x) − r(y′; x)(cid:1),

(3.1)

where σ(x) = ex/(ex + 1) is the logistic function. The reward function associated with the Bradley-
Terry model can be estimated by maximizing the log-likelihood log P(y ≻ y′|x). Suppose the
true reward function r(y; x)) is available, Christiano et al. (2017) proposed to solve the following
optimization problem with policy optimization algorithms in RL such as PPO (Schulman et al.,
2017):

max
θ

Ex∼X ,y∼πθ(·|x)[r(y; x)] − η−1Ex∼X [KL(πθ(·|x)∥πref(·|x))],

(3.2)

where X is the prompt distribution.

Rafailov et al. (2024b) identified that the optimization problem above has a closed-form solution

such that for any y,

π∗(y|x) ∝ πref(y|x) exp(ηr(y; x)),

6

which can be further converted to the DPO loss for any triplet (x, yw, yl) where the winner yw is
chosen over the loser yl:

ℓDPO(x, yw, yl; θ; πref) := − log σ

(cid:32)

(cid:20)

η−1

log

(cid:19)

(cid:18) πθ(yw|x)
πref(yw|x)

− log

(cid:18) πθ(yl|x)
πref(yl|x)

(cid:19)(cid:21)(cid:33)
.

3.2 RLHF with General Preference

Following Wang et al. (2024); Munos et al. (2023), we aim to establish RLHF methods without a
reward model, as the human preference can be non-transitive (Tversky, 1969). Under a general
preference oracle P(y ≻ y′|x), we follow Dud´ık et al. (2015) and aim to identify the von Neumann
winner. More specifically, the von Neumann winner π∗ is the (symmetric) Nash equilibrium of the
following two-player constant-sum game:

(π∗, π∗) = arg max

π

(cid:104)

Ex∼X

min
π′

Ey∼π(·|x),y′∼π′(·|x)

(cid:2)P(y ≻ y′|x)(cid:3)(cid:105)

.

(3.3)

In addition, we define the winning probability of one response y against a distribution of

responses π as

P(y ≻ π|x) = Ey′∼π(·|x)[P(y ≻ y′|x)],

and the winning probability of one policy π against another policy π′ as

P(π ≻ π′|x) = Ey∼π(·|x)Ey′∼π′(·|x)[P(y ≻ y′|x)].

Furthermore, we define P(π ≻ π′) = Ex∼X [P(π ≻ π′|x)], where x is a prompt drawn from the
prompt distribution X . The two-player constant-sum game (3.3) can be simplified as

(π∗, π∗) = arg max

π

P(π ≻ π′).

min
π′

4 Self-Play Preference Optimization(SPPO)

In this section, we introduce the Self-Play Preference Optimization(SPPO) algorithm, derived from
the following theoretical framework.

4.1 Theoretical Framework

There are well-known algorithms to approximately solve the Nash equilibrium in a constant-sum
two-player game. In this work, we follow Freund and Schapire (1999) to establish an iterative
framework that can asymptotically converge to the optimal policy on average. We start with a
theoretical framework that conceptually solves the two-player game as follows:

πt+1(y|x) ∝ πt(y|x) exp(ηP(y ≻ πt|x)), for t = 1, 2, . . . .

(4.1)

(4.1) is an iterative framework that relies on the multiplicative weight update in each round t and
enjoys a clear structure. Initially, we have a base policy π1 usually from some supervised fine-tuned
model. In each round, the updated policy πt+1 is obtained from the reference policy πt following

7

the multiplicative weight update. More specifically, a response y should have a higher probability
weight if it has a higher average advantage over the current policy πt.

Equivalently, (4.1) can be written as

πt+1(y|x) =

πt(y|x) exp (cid:0)ηP(y ≻ πt|x)(cid:1)
Zπt(x)

,

(4.2)

y πt(y|x) exp (cid:0)ηP(y ≻ πt|x)(cid:1) is the normalizing factor (a.k.a., the partition
where Zπt(x) = (cid:80)
function). For any fixed x and y, the ideal update policy πt+1 should satisfy the following equation:

log

(cid:19)

(cid:18) πt+1(y|x)
πt(y|x)

= η · P(y ≻ πt|x) − log Zπt(x).

(4.3)

Unlike the pair-wise design in DPO or IPO that cancels the log normalizing factor log Zπt(x) by
differentiating (4.3) between y and y′, we choose to approximate (4.3) directly in terms of L2
distance:

πt+1 = argmin

π

Ex∼X ,y∼πt(·|x)

(cid:18)

log

(cid:19)

(cid:18) π(y|x)
πt(y|x)

(cid:16)

−

ηP(y ≻ πt|x) − log Zπt(x)

(cid:17)(cid:19)2

.

(4.4)

Estimation of the Probability The optimization objective (4.4) can be approximated with finite
samples. We choose to sample K responses y1, y2, . . . , yK ∼ πt(·|x) for each prompt x, and denote
the empirical distribution by (cid:98)πK
t . The finite-sample optimization problem can be approximated as

πt+1 = argmin

π

Ex∼X ,y∼πt(·|x)

(cid:18)

log

(cid:19)

(cid:18) π(y|x)
πt(y|x)

(cid:16)

−

ηP(y ≻ (cid:98)πK

t |x) − log Z

(cid:98)πK

t

(cid:17)(cid:19)2

(x)

.

(4.5)

t

t |x) = (cid:80)K

Specifically, P(y ≻ (cid:98)πK
t |x))].
Z
(x), treated as an expectation, can be further estimated by B new samples with in total O(KB)
(cid:98)πK
queries of the preference oracle P. (4.5) is an efficiently tractable optimization problem. Informally
speaking, when K → ∞, (4.5) will recover (4.4). We have the following guarantee on the convergence
of (4.4):

(x) = Ey∼πt(·|x)[exp(ηP(y ≻ (cid:98)πK

P(y ≻ yk|x)/K and Z

(cid:98)πK

k=1

t

Theorem 4.1. Assume the optimization problem (4.4) is realizable. Denote πt as the policy
obtained via (4.4) and the mixture policy ¯πT = 1
T ), we have that
T
(cid:2)P(π ≻ ¯πT )(cid:3) − min

t=1 πt. By setting η = Θ(1/

(cid:2)P(π ≺ ¯πT )(cid:3) = O(1/

(cid:80)T

T ).

√

√

max
π

π

Theorem 4.1 characterizes the convergence rate of the average policy across the time horizon T
towards the Nash equilibrium, in terms of the duality gap. The proof is based on Theorem 1 in
Freund and Schapire (1999) with slight modification. For completeness, we include the proof in
Appendix B.

Alternatively, we can avoid estimating log Z

(x) by replacing it with a constant based on the
(cid:98)πK
t
human preference model. The choice of the constant is discussed in detail in Appendix A. Here, we
replace log Z

(x) with η/23 in (4.5) to obtain a more clear objective:

(cid:98)πK

t

πt+1 = argmin

π

Ex∼X ,y∼πt(·|x)

(cid:18)

log

(cid:19)

(cid:18) π(y|x)
πt(y|x)

(cid:18)
P(y ≻ (cid:98)πK

t |x) −

− η

(cid:19)(cid:19)2

.

1
2

(4.6)

3Assuming the winning probability between any given pair is either 1 or 0 with equal chance, when K → ∞, we

can show that indeed Z

(cid:98)πK
t

(x) → eη/2. Also see Appendix A for a complete derivation.

8

Algorithm 1 Self-Play Preference Optimization(SPPO)
1: input: base policy πθ1, preference oracle P, learning rate η, number of generated samples K.
2: for t = 1, 2, . . . do
3: Generate synthetic responses by sampling x ∼ X and y1:K ∼ πt(·|x).
4:

Annotate the win-rate P(yk ≻ yk′|x), ∀k, k′ ∈ [K].
Select responses from y1:K to form dataset Dt = {(xi, yi, (cid:98)P (yi ≻ πt|xi))}i∈[N ].

5:
6: Optimize πθt+1 according to (4.6):

θt+1 ← argmin

θ

E

(x,y, (cid:98)P (y≻πt|x))∼Dt

(cid:18)

log

(cid:19)

(cid:18) πθ(y|x)
πt(y|x)

(cid:18)

− η

(cid:98)P (y ≻ πt|x) −

(cid:19)(cid:19)2

1
2

.

(4.7)

7: end for

Intuitively, if a tie occurs (i.e., P(y ≻ (cid:98)πK
at y. If y wins over (cid:98)πK
density at y to employ the advantage of y over (cid:98)πK
objective (4.6).

t on average (i.e., P(y ≻ (cid:98)πK

t |x) = 1/2), we prefer the model does not update weight
t |x) > 1/2), then we increase the probability
t . In our experiments, we choose to minimize the

4.2 The SPPO Algorithm

Based on the aformentioned theoretical framework, we propose the Self-Play Preference Optimization
algorithm in Algorithm 1. In each round t, Algorithm 1 will first generate K responses y1, y2, . . . , yK
according to πt(·|x) for each prompt x (Line 3). Then, the preference oracle P will be queried
to calculate the win rate among the K responses (Line 4). At Line 5, certain criteria can be
applied to determine which response should be kept in the constructed dataset Dt and construct
the prompt-response-probability triplet (x, y, (cid:98)P (y ≻ πt|x)). We will discuss the design choices later
in Section 5. One straightforward design choice is to include all K responses into Dt and each
(cid:98)P (yi ≻ πt|x) is estimated by comparing yi to all K responses. In total, O(K2) queries will be made.
Then the algorithm will optimize (4.6) on the dataset Dt (Line 6).

4.3 Connection to Policy Gradient

While SPPO is derived from the iterative framework (Freund and Schapire, 1999) for two-player
games, the square loss in the SPPO objective (4.4) provides an alternative interpretation for SPPO
as a semi-online variant of policy gradient method due to its special loss form. The difference from
standard policy gradient is that it collects samples from πθt at the start of iteration t, rather than
perform on-policy sampling at each gradient step.

Consider a general reward function r(y; x), the RLHF problem (3.2) can be written as:

max
θ

J(θ) := Ex∼X ,y∼πθ(·|x)

(cid:20)

r(y; x) − η−1 log

πθ(y|x)
πref(y|x)

(cid:21)

.

The policy gradient of the objective J(θ) is:

∇J(θ) = Ex∼X ,y∼πθ(·|x)

(cid:20)(cid:18)

r(y; x) − η−1 log

πθ(y|x)
πref(y|x)

9

− b(x)

(cid:19)

(cid:21)
∇ log πθ(y|x)

(4.8)

(4.9)

= ηEx∼X ,y∼πθ(·|x)

(cid:20)

− ∇

(cid:18)

r(y; x) − η−1 log

πθ(y|x)
πref(y|x)

(cid:19)2(cid:21)

− b(x)

,

(4.10)

where the first line follows the policy gradient theorem (Sutton et al., 1999) and the baseline b(x)
is an arbitrary constant relying only on x used for variance reduction. Comparing the square loss
(4.10) with the SPPO objective (4.4) (rewritten below):

θt+1 = argmin

θ

Ex∼X ,y∼πθt (·|x)

(cid:20)(cid:18)

P(y ≻ πθt|x) − η−1 log

(cid:19)

(cid:18) πθ(y|x)
πθt(y|x)

− η−1 log Zπθt

(cid:19)2(cid:21)

(x)

,

one can see that the win rate P(y ≻ πθt|x) is exactly the reward SPPO aims to maximize, and
η−1 log Zπθt
(x) is in fact the best possible baseline–the (soft) value function. When the value
function is not available in practice, it can be replaced by any constant baseline to reduce the
variance of the policy gradient. We choose 1/2 as a good approximation to η−1 log Zπθt
(x) but the
constant can vary depending on the human preference model (see Appendix A). Equation (4.9) is
also discussed in Munos et al. (2023).

Comparing with the general framework proposed by Swamy et al. (2024), SPPO can be seen as a
new, straightforward variant of policy gradient method without the need of extra modifications such
as gradient clipping in PPO, Hessian calculation in TRPO, or maintaining multiple components
(Q-critic, V-critic, actor, etc.) in many policy optimization algorithms.

4.4 Token-Level Q∗ Learning

Rafailov et al. (2024a) showed that under the Max-Entropy RL formulation, the token-level log-ratio
log πθ(y|x)
πref(y|x) can be seen as an implicit token-level reward or advantage function (invariant under
reward shaping). Below we show the square loss in SPPO can also lead to the optimal Max-Entropy
policy π∗, with token-level optimal value/advantage function.

We first briefly restate the setting and results in Rafailov et al. (2024b). The token-level MDP
defines the state sh = (x, y1, y2, . . . , yh−1) as the prefix tokens, and the action ah = yh as the next
token. An auto-regressive language model π(y|x) can be viewed as a token-level policy π(ah|sh)
and the transition kernel is known and deterministic because it only concatenates the next token to
the prefix to form a new token sequence sh+1 = (x, y1, y2, . . . , yh).

The Max-Entropy RL setting again considers the reverse-KL regularized reward maximization

problem (3.2):

max
θ

Ex∼X ,y∼πθ(·|x)[r(y; x)] − η−1Ex∼X [KL(πθ(·|x)∥πref(·|x))]

=Ex∼X ,y∼πθ(·|x)[r(y; x) + η−1 log πref(y|x)] + η−1Ex∼X [H(πθ(·|x))].
We denote the optimal solution for the problem above as π∗. Rafailov et al. (2024a) showed that
the Bradley-Terry preference model (4.13) can be rewritten as:

P(yw ≻ yl|x) = σ

(cid:18)

η−1

|yw|
(cid:88)

h=1

log

π∗(aw
πref(aw

h |sw
h )
h |sw
h )

− η−1

|yl|
(cid:88)

h=1

log

π∗(al
πref(al

h|sl
h)
h|sl
h)

(cid:19)
,

where the state and action is defined as in the token-level MDP introduced above, with superscription
(·)w and (·)l denoting if it is for the winner yw or the loser yl. And maximizing the log likelihood
with π∗ replaced by πθ gives the DPO loss.

10

From now on we assume the horizon is fixed at H for simplicity. The derivation of the Max-

Entropy RL formulation relies on the (soft) optimal value function Q∗ and V ∗ as4:

V ∗(sH+1) = r(sH+1) := r(y; x), (reward at EOS)
Q∗(sh, ah) = η−1 log πref(ah|sh) + V ∗(sh+1),
(cid:88)

V ∗(sh) = η−1 log

exp (cid:0)ηQ∗(sh, a)(cid:1), when h ≤ H.

Rafailov et al. (2024a) showed that the optimal policy π∗ satisfies:

a

η−1 log π∗(ah|sh) = Q∗(sh, ah) − V ∗(sh)

= η−1 log πref(ah|sh) + V ∗(sh+1) − V ∗(sh).

It can be verified that for s1 = (x), we have ηV ∗(s1) = log (cid:80)

y πref(y|x) exp (cid:0)ηr(y; x)(cid:1):

exp (cid:0)ηV ∗(s1)(cid:1) =

=

=

· · ·

=

=

(cid:88)

a1
(cid:88)

exp (cid:0)ηQ∗(s1, a1)(cid:1)

πref(a1|s1) exp (cid:0)ηV ∗(s2)(cid:1)

a1
(cid:88)

a1,a2

πref(a1|s1)πref(a2|s2) exp (cid:0)ηV ∗(s3)(cid:1)

(cid:88)

H
(cid:89)

h=1

πref(ah|sh) exp (cid:0)ηr(sH+1)(cid:1)

(a1,a2,...,aH )
(cid:88)

πref(y|x) exp (cid:0)ηr(y; x)(cid:1).

y

Going back to the SPPO objective (4.4) at t-th iteration, if we set πref = πt and r(y; x) = P(y ≻

πt|x), we have V ∗(s1) = η−1 log Zπt(x), and the learning objective at t-th iteration becomes:

πt+1 = argmin

π

Ex∼X ,y∼πt(·|x)

(cid:18)

log

(cid:19)

(cid:18) π(y|x)
πt(y|x)

(cid:16)

−

ηP(y ≻ πt|x) − log Zπt(x)

(cid:17)(cid:19)2

= argmin

π

Es1∼X ,ah∼πt(·|sh)

(cid:18) H
(cid:88)

h=1

log

π(ah|sh)
π∗(ah|sh)

(cid:19)2

.

(4.11)

Similar to DPO, SPPO “secretly” encourages the policy πθ to converge to the optimal policy π∗
at token level via the square loss form (4.11). Additionally, one may realize that minimizing the
square-loss form is related to minimizing the KL divergence KL(πθ∥π∗) via policy gradient:

∇θKL(πθ∥π∗) = Es1∼X ,ah∼πθ(·|sh)

(cid:20)(cid:18) H
(cid:88)

h=1

log

πθ(ah|sh)
π∗(ah|sh)

(cid:19) H
(cid:88)

(cid:21)
∇θ log πθ(ah|sh)

h=1

4Here we restated with the sequence-level reward r(y; x). Rafailov et al. (2024a) started their derivation from
a ground-truth token-level reward r(sh, ah), which is under-specified due to the reward reshaping issue (Ng et al.,
1999): reshaping the reward will not affect the Bradley-Terry preference probability so it is impossible to recover the
ground-truth reward from the preference signal (Rafailov et al., 2024a, Section 4.2).

11

= Es1∼X ,ah∼πθ(·|sh)

(cid:18) H
(cid:88)

(cid:20)

∇θ

log

h=1

πθ(ah|sh)
π∗(ah|sh)

(cid:19)2(cid:21)

.

4.5 Comparison with DPO, IPO, and KTO

In practice, we utilize mini-batches of more than 2 responses to estimate the win rate of a given
response, while the DPO and IPO loss focus on a single pair of responses. When only a pair of
responses yw and yl is available, we have the pair-wise symmetric loss based on the preference
triplet (x, yw, yl) defined as:

ℓSPPO(x, yw, yl; θ; πref) :=

log

(cid:18)

(cid:19)

(cid:16)

− η

P(yw ≻ yl|x) −

(cid:17)(cid:19)2

1
2

(cid:18) πθ(yw|x)
πref(yw|x)
(cid:18)

+

log

(cid:18) πθ(yl|x)
πref(yl|x)

(cid:19)

(cid:16)

− η

P(yw ≺ yl|x) −

(cid:17)(cid:19)2

,

1
2

(4.12)

where P(yw ≻ yl|x) can be either a soft probability within [0, 1] or a hard label 1 indicating yw ≻ yl.
We now compare the SPPO loss to other baselines assuming a hard label yw ≻ yl is given. For

the ease of comparison, let (β = η−1):
(cid:18) πθ(yw|x)
πref(yw|x)

a = β log

(cid:19)

, b = β log

(cid:18) πθ(yl|x)
πref(yl|x)

(cid:19)

, c = βKL(πθ∥πref),

then we have

ℓDPO(yw, yl, x) = − log σ(a − b),
ℓIPO(yw, yl, x) = [(a − b) − 1]2,
ℓKTO(yw, yl, x) = σ(−a + c) + σ(b − c) (simplified),

(4.13)

(4.14)

(4.15)

where σ(x) = ex/(ex + 1) and the SPPO loss can be written as

ℓSPPO(yw, yl, x) = (a − 1/2)2 + (b + 1/2)2.

It can be seen that SPPO not only pushes the gap between a and b to be 1, but also attempts
to push the value of a to be close to 1/2 and the value of b to be close to −1/2 so that πθ(yw|x) >
πref(yw|x) and πθ(yl|x) < πref(yl|x). We believe this to be particularly important: when the
preference pairs are scarce (e.g., one pair for each prompt), there is no guarantee that the winner
log-ratio a will increase and the loser log-ratio b will decrease. Instead, only the gap between the
winner and the loser (i.e., a − b) will increase. This phenomenon is observed by Pal et al. (2024)
that DPO only lowers the loser’s likelihood, but barely change the winner’s likelihood.

As discussed above, fitting β log

directly to P(y ≻ πt|x) − 1/2 under a square loss is
closely related to the policy gradient. This explains why SPPO is more effective than IPO which
to P(yw ≻ πt|x) − P(yl ≻ πt|x). In addition,
attempts to fit β log
SPPO shares a similar spirit as KTO. The KTO loss pushes a to be large by minimizing σ(−a + c)
and pushes b to be small by minimizing σ(b − c). In contrast, SPPO pushes a to be as large as 1/2
and b to be as small as −1/2.

(cid:16) πt+1(yw|x)
πt(yw|x)

(cid:16) πt+1(yl|x)
πt(yl|x)

− β log

(cid:17)

(cid:17)

(cid:16) πt+1(y|x)
πt(y|x)

(cid:17)

On the other hand, we would like to comment that although DPO and KTO can be extended
to their iterative variants, they are not by nature iterative algorithms and do not have provable

12

guarantees that they can reach the Nash equilibrium. In contrast, SPPO and IPO are by design
capable to solve the Nash equilibrium iteratively. SPPO is superior to IPO because its design
explicitly alleviates the data sparsity issue, as discussed above and detailed in Pal et al. (2024).

5 Experiments

5.1 Experiment Setup

Base Model and Datasets We follow the experimental setup of Snorkel5, a model that utilizes
iterative DPO to achieve state-of-the-art performance on AlpacaEval benchmarks. Specifically,
we use Mistral-7B-Instruct-v0.2 as our base model6. Mistral-7B-Instruct-v0.2 is an instruction
fine-tuned version of Mistral-7B-v0.2 model (Jiang et al., 2023a). We also adopt Ultrafeedback (Cui
et al., 2023) as our source of prompts which includes around 60k prompts from diverse resources.
During generation, we follow the standard chat template of Mistral-7B. To avoid overfitting during
the fine-tuning, we split the dataset into three portions and use only one portion per iteration.
These settings were also adopted by training the model Snorkel-Mistral-PairRM-DPO7 (Snorkel).
We follow the splitting in Snorkel for a fair comparison. Additionally, we use Llama-3-8B-Instruct8
as a stronger base model along with the same preference dataset and data splitting.

Preference Model We employ PairRM (Jiang et al., 2023b), an efficient pair-wise preference
model of size 0.4B. PairRM is based on DeBERTA-V3 (He et al., 2021) and trained on high-quality
human-preference datasets. Results on benchmarks like Auto-J Pairwise dataset (Li et al., 2023a)
show that it outperforms most of the language-model-based reward models and performs comparably
with larger reward models like UltraRM-13B (Cui et al., 2023). We refer the readers to the homepage
on Huggingface9 for detailed benchmark results. We therefore keep PairRM as our ranking model
following Snorkel for a balance between accuracy and efficiency.

Specifically, PairRM will output a “relative reward” s(y, y′; x) that reflects the strength difference

between y and y′, i.e.,

P(y ≻ y′|x) =

exp(s(y, y′; x))
1 + exp(s(y, y′; x))

.

Unlike the Bradley-Terry-based reward model, PairRM only assigns the relative reward which is not
guaranteed to be transitive (i.e., s(y1, y2; x) + s(y2, y3; x) ̸= s(y1, y3; x)). So it indeed models the
general preference.

Response Generation and Selection During the generation phase in each iteration, we use top
p = 1.0 and temperature 1.0 to sample from the current policy. We sample with different random
seeds to get K = 5 different responses for each prompt. Previous works utilizing Iterative DPO
choose 2 responses to form a pair for each prompt. For a fair comparison, we do not include all
K = 5 responses in the preference data but choose two responses among them. Following Snorkel,

5https://huggingface.co/snorkelai/Snorkel-Mistral-PairRM-DPO
6https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
7https://huggingface.co/snorkelai/Snorkel-Mistral-PairRM-DPO
8https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
9https://huggingface.co/llm-blender/PairRM

13

we choose the winner yw and loser yl to be the response with the highest and lowest PairRM score,
which is defined for each response yi as:

sPairRM(yi; x) :=

1
K

K
(cid:88)

k=1

s(yi, yk; x).

Probability Estimation We then estimate the win rate over the distribution by the average win
rate over all the sampled responses as explained in (4.5):

(cid:98)P (yi ≻ πt|x) =

1
K

K
(cid:88)

k=1

P(yi ≻ yk|x), ∀i ∈ [K].

Hyperparameter Tuning The experiments are conducted on 8 × Nvidia A100 GPUs. For
SPPO, we trained three iterations in total. In each iteration, we selected the model trained on
the first epoch of the 20k prompts from UltraFeedback to proceed to the next iteration. For both
Mistral-7B-Instruct-v0.2 and Llama-3-8B-Instruct, the global training batch size is set to 64, and η
is set to 1e3. The learning rate schedule is determined by the following hyperparameters: learning
rate=5.0e-7, number of total training epochs=18, warmup ratio=0.1, linear schedule. The best
hyper-parameters for each model are selected by the average win rate (judged by PairRM-0.4B)
on a hold-out subset of Ultrafeedback as the metric. For more details on the win-rate comparison
using PairRM as a judge, please refer to Section 5.2 and Figure 3.

Baselines We evaluate the following base models as well as baseline methods for fine-tuning
LLMs:

• Mistral-7B-Instruct-v0.2: Mistral-7B-Instruct-v0.2 is an instruction fine-tuned version of Mistral-

7B-v0.2 model (Jiang et al., 2023a). It is the starting point of our algorithm.

• Snorkel (Mistral-PairRM-DPO): We directly evaluate the uploaded checkpoint on HuggingFace10.

This model is obtained by three rounds of iterative DPO from Mistral-7B-Instruct-v0.2.

• (Iterative) DPO: We also implement the iterative DPO algorithm by ourselves. The experimental
settings and model selection schemes align with those used for SPPO, except for the adoption
of the DPO loss function as defined in (4.13). Hyperparameters are optimized to maximize the
average win-rate assessed by PairRM at each iteration. Note that the practical algorithm in
Rosset et al. (2024) is essentially the same as iterative DPO.

• (Iterative) IPO: We implement the iterative IPO algorithm by ourselves. The experimental setting
and the model selection scheme is the same as iterative DPO, except that the loss function is
the IPO loss (4.14). For fair comparison, hyperparameters for IPO is also selected by evaluation
using the average PairRM win-rate on the hold-out subset of Ultrafeedback.

• Self-rewarding LM: Yuan et al. (2024) proposed to prompt the LLM itself as a preference judge
to construct new preference pairs and iteratively fine-tune the LLM with the DPO algorithm.
We use the AlpacaEval 2.0 win rate reported by Yuan et al. (2024) for comparison. Note that
Self-rewarding LM is a trained from Llama 2 70B.

10https://huggingface.co/snorkelai/Snorkel-Mistral-PairRM-DPO

14

• Llama-3-8B-Instruct: Llama-3-8B-Instruct is an instruction-tuned model optimized for dialogue
use cases and outperforms many of the available open-source chat models on common industry
benchmarks.

Benchmarks Following previous works, we use AlpacaEval 2.0 (Dubois et al., 2024a), Arena-
Hard(Li et al., 2024), MT-Bench (Zheng et al., 2024), and Open LLM Leaderboard (Beeching et al.,
2023a) as our evaluation benchmarks.

• AlpacaEval 2.0 is an LLM-based automatic evaluation benchmark. It employs AlpacaFarm
(Dubois et al., 2024b) as its prompts set composed of general human instructions. The model
responses and the reference response generated by GPT-4-Turbo are fed into a GPT-4-Turbo-based
annotator to be judged. We follow the standard approach and report the win rate over the
reference responses.

• Arena-Hard (Li et al., 2024) is a high-quality benchmark that claims to be harder and has
the highest correlation and separability to Chatbot Arena among popular open-ended LLM
benchmarks including AlpacaEval 2.0. We evaluate our models Mistral-PairRM-SPPO and the
baseline models.

• MT-Bench (Zheng et al., 2024) is a collection of 80 high-quality multi-turn open-ended questions.
The questions cover topics like writing, role-playing, math, coding, etc.. The generated answer is
judged by GPT-4 and given a score directly without pairwise comparison.

• Open LLM Leaderboard (Beeching et al., 2023a) consists of six datasets, each of which
focuses on a facet of language model evaluation. In detail, the evaluation rubric includes math
problem-solving, language understanding, human falsehood mimicking, and reasoning. We follow
the standard evaluation process and use in-context learning to prompt the language model and
compute the average score over six datasets to measure the performance.

5.2 Experimental Results

Evaluation using GPT-4 as a judge Human evaluation remains the benchmark for quality
and accuracy (Askell et al., 2021; Ouyang et al., 2022). However, due to its limitations in scalability
and reproducibility, we explore the alternative approach of using the advanced capabilities of
GPT-4 (OpenAI et al., 2023) as an automatic evaluation tool. We conduct GPT-4-based automatic
evaluation on AlpacaEval 2.0 (Li et al., 2023b), MT-Bench (Zheng et al., 2023), and Arena-Hard (Li
et al., 2024) to measure the chatbot capability of our model. The results can be found in Table 1
for AlpacaEval 2.0, Figure 2 (left) for MT-Bench, and Figure 2 (right) for Arena-Hard. We found
that the performance of SPPO models consistently improves throughout all iterations.

Table 1 (AlpacaEval 2.0) shows the win rate over the GPT-4-Turbo baseline of different models
on 805 prompts. We also include one column indicating the length-controlled win rate, and one
column on the average length of each model, to account for the tendency of the LLM-based judge
to favor longer sequence outputs — an issue colloquially termed the ”reward hacking” phenomenon.

15

Table 1: AlpacaEval 2.0 evaluation of various models (detailed in Baselines) in terms of both normal
and length-controlled (LC) win rates in percentage (%). Mistral-7B-SPPO Iter3 model achieves
the highest LC win rate of 28.53% and a normal win rate of 31.02%. SPPO demonstrates steady
performance gains across iterations and outperforms other baselines which show a tendency to
produce longer responses. Additionally, re-ranking with the PairRM reward model (best-of-16) at
test time consistently enhances the performance across all models and SPPO (best-of-16) achieves
high win rate without strong external supervision like GPT-4. We additionally include the results
obtained from fine-tuning Llama-3-8B-Instruct, which also show steady performance improvement.

Model

Mistral-7B-Instruct-v0.2
Mistral-7B-Instruct-v0.2 (best-of-16)

Snorkel (Mistral-PairRM-DPO)
Snorkel (Mistral-PairRM-DPO best-of-16)

Self-Rewarding 70B Iter1
Self-Rewarding 70B Iter2
Self-Rewarding 70B Iter3

Mistral-7B-DPO Iter1
Mistral-7B-DPO Iter2
Mistral-7B-DPO Iter3

Mistral-7B-IPO Iter1
Mistral-7B-IPO Iter2
Mistral-7B-IPO Iter3

Mistral-7B-SPPO Iter1
Mistral-7B-SPPO Iter2
Mistral-7B-SPPO Iter3

Mistral-7B-SPPO Iter1 (best-of-16)
Mistral-7B-SPPO Iter2 (best-of-16)
Mistral-7B-SPPO Iter3 (best-of-16)

Llama-3-8B-Instruct

Llama-3-8B-SPPO Iter1
Llama-3-8B-SPPO Iter2
Llama-3-8B-SPPO Iter3

AlpacaEval 2.0
LC Win Rate Win Rate

Avg. Len

17.11
22.45

26.39
29.97

-
-
-

23.81
24.23
22.30

23.78
21.08
20.06

14.72
17.94

30.22
34.86

9.94
15.38
20.44

20.44
24.46
23.39

20.77
23.38
22.47

24.79(+7.69)
26.89(+2.10)
28.53(+1.64)

28.71(+6.26)
31.23(+2.52)
32.13(+0.9)

23.51(+8.79)
27.62(+4.11)
31.02(+3.40)

27.77(+9.83)
32.12(+4.35)
34.94(+2.82)

22.92

22.57

31.73(+8.81)
35.15(+3.42)
38.77(+3.62)

31.74(+9.17)
35.98(+4.24)
39.85(+3.87)

1676
1529

2736
2616

1092
1552
2552

1723
2028
2189

1693
2660
2760

1855
2019
2163

1901
2035
2174

1899

1962
2021
2066

According to the table, Mistral-7B-SPPO Iter3 has the highest win rate, 28.52% for the length-
controlled version, and 31.02% for the overall win rate.

The performance gains over previous iterations are 7.69% (Mistral-7B-Instruct → Iter1), 2.10%
(Iter1 → Iter2), and 1.64% (Iter2 → Iter3), respectively, indicating steady improvements across
iterations, as illustrated in Figure 1. We also apply SPPO to a stronger baseline model, i.e.,
Llama-3-8B-Instruct, and the fine-tuned model Llama-3-8B-SPPO has a higher length-controlled

16

Table 2: AlpacaEval 2.0 leaderboard results of both normal and length-controlled (LC) win rates in
percentage (%). Mistral-7B-SPPO can outperform larger models and Mistral-7B-SPPO (best-of-16)
can outperform proprietary models such as GPT-4(6/13). Llama-3-8B-SPPO exhibits even better
performance.

Model

GPT-4 Turbo
Claude 3 Opus
Llama-3-8B-SPPO Iter3
GPT-4 0314
Llama 3 70B Instruct
Mistral-7B-SPPO Iter3 (best-of-16)
GPT-4 0613
Snorkel (Mistral-PairRM-DPO best-of-16)
Mistral Medium
Mistral-7B-SPPO Iter3
Claude 2
Snorkel (Mistral-PairRM-DPO)
Gemini Pro
Mistral 8×7B v0.1
Llama 3 8B Instruct

AlpacaEval 2.0
LC. Win Rate Win Rate

50.0
40.5
38.8
35.3
34.4
32.1
30.2
30.0
28.6
28.5
28.2
26.4
24.4
23.7
22.9

50.0
29.1
39.9
22.1
33.2
34.9
15.8
34.9
21.9
31.0
17.2
30.2
18.2
18.1
22.6

(a)

(b)

Figure 1: Win Rate against GPT-4-Turbo with (a) and without (b) Length Controlling (LC) on
AlpacaEval 2.0. SPPO demonstrates steady improvements on both LC and raw win rates.

win rate 38.77% and overall win rate 39.85%. The performance gains are more significant: 8.81%
(Llama-3-8B-Instruct → Iter1), 3.42% (Iter1 → Iter2), and 3.62% (Iter2 → Iter3), summing up to a
total gain of 15.85%.

Additionally, the result indicates that SPPO achieves superior performance compared to the
iterative variants of DPO and IPO. The length-controlled win rate for SPPO reaches 28.53%,

17

Iter1Iter2Iter320253035LC. Win Rate (%)GPT-4 0314Snorkel (Mistral-PairRM-DPO)Mistral-7B-Instruct-v0.2Mistral-7B-DPOMistral-7B-IPOMistral-7B-SPPOIter1Iter2Iter31015202530Win Rate (%)GPT-4 0314Snorkel (Mistral-PairRM-DPO)Mistral-7B-Instruct-v0.2Self-Rewarding 70BMistral-7B-DPOMistral-7B-IPOmistral-7B-SPPOModel

Mistral-7B-Instruct-v0.2
Snorkel (Mistral-PairRM-DPO)

Mistral-7B-DPO Iter1
Mistral-7B-DPO Iter2
Mistral-7B-DPO Iter3

Mistral-7B-SPPO Iter1
Mistral-7B-SPPO Iter2
Mistral-7B-SPPO Iter3

MT-Bench
1st Turn 2nd Turn Average

Model

Arena-Hard-Auto-v0.1

7.78
7.83

7.45
7.57
7.49

7.63
7.90
7.84

7.25
7.33

6.58
6.56
6.69

6.79
7.08
7.34

7.51
7.58

7.02
7.06
7.09

7.21
7.49
7.59

Mistral-7B-Instruct

Snorkel (Mistral-PairRM-DPO)

Mistral-7B-SPPO Iter1
Mistral-7B-SPPO Iter2
Mistral-7B-SPPO Iter3

12.6

20.7

18.7
20.4
23.3

Figure 2: MT-Bench & Arena-Hard Evaluation. Left: Mistral-7B-SPPO Iter3 outperforms
all baseline models by achieving an average score of 7.59 in MT-Bench. Despite initial drops in
performance in the first two iterations, SPPO Iter3 improves upon the base model by the final
iteration. Right: Mistral-7B-SPPO Iter3 outperforms the baseline model Snorkel(Mistral-PairRM-
DPO) in Arena-Hard. The improvement across different iterations is consistent.

outperforming the DPO’s best rate of 26.39% (by Snorkel) and IPO’s rate of 25.45%. Notably, while
DPO and IPO training tend to significantly increase the average output length—2736 and 2654,
respectively—SPPO shows a more moderate length increase, moving from 1676 in the base model
to 2163 at the third iteration. Finally, we present the best-of-16 results for each model, selected
using the PairRM reward model. We find that re-ranking with the preference model at test time
can consistently improve the performance of base model (Mistral-7B-Instruct-v0.2), DPO (Snorkel),
and SPPO (Iter3) by 5.34%, 3.57%, and 3.6%, respectively. Notably, this shows that while SPPO
significantly enhances model alignment using PairRM-0.4B as the sole external supervision, it has
not resulted in over-optimization against the preference model (Gao et al., 2023).

In Table 2, we compare SPPO on the AlpacaEval 2.0 leaderboard with other state-of-the-art AI
chatbots. We found our SPPO model outperforms many competing models trained on proprietary
alignment data (e.g., Claude 2, Gemini Pro, & Llama 3 8B Instruct). When applied to Llama 3
8B Instruct, our Llama-3-8B-SPPO exhibits an even higher win rate. With test-time reranking,
Mistral-7B-SPPO Iter3 (best-of-16) is even competitive to GPT-4 0613 and Llama 3 70B Instruct.
In Figure 2 (left), we evaluate the performance of SPPO on MT-Bench. We can see that
Mistral-7B-SPPO Iter3 outperforms all baseline models, achieving an average score of 7.59. While
we are not certain why the MT-Bench performance drops at the first two iterations, the performance
of SPPO at the final iteration still improves over the base model.

Arena-Hard (Li et al., 2024) contains 500 challenging user queries and follow the same evaluation
method as AlpacaEval 2.0. In Figure 2 (right), we evaluate the performance of SPPO on Arena-Hard.
We can see that Mistral-7B-SPPO exhibits a steady performance gain across iterations.Mistral-7B-
SPPO Iter 3 outperforms the baseline models, achieving an average score of 23.3.

Open LLM Leaderboard We further evaluate the capabilities of SPPO models using Huggingface
Open LLM Leaderboard (Beeching et al., 2023b). This leaderboard encompasses 6 different datasets,
each focusing on a specific capability of LLMs: Arc (Clark et al., 2018), HellaSwag (Zellers et al.,
2019), Winogrande (Sakaguchi et al., 2021), MMLU (Hendrycks et al., 2020), TruthfulQA (Lin et al.,
2021), and GSM8k (Cobbe et al., 2021). The models are prompted with zero or few-shot exemplars.

18

Table 3: Open LLM Leaderboard Evaluation. SPPO fine-tuning improves the base model’s
performance on different tasks, reaching a state-of-the-art average score of 66.75 for Mistral-7B
and 70.29 for Llama-3-8B. For Mistral-7B, subsequent iterations of DPO, IPO, and SPPO see
a decline in performance. It is possible that aligning with human preferences (simulated by the
PairRM preference model in our study) may not always enhance, and can even detract from, overall
performance.

Models

Arc TruthfulQA WinoGrande GSM8k HellaSwag MMLU Average

Mistral-7B-Instruct-v0.2

63.65

Snorkel

Mistral-7B-DPO Iter1
Mistral-7B-DPO Iter2
Mistral-7B-DPO Iter3

Mistral-7B-IPO Iter1
Mistral-7B-IPO Iter2
Mistral-7B-IPO Iter3

Mistral-7B-SPPO Iter1
Mistral-7B-SPPO Iter2
Mistral-7B-SPPO Iter3

Llama-3-8B-Instruct

Llama-3-8B-SPPO Iter1
Llama-3-8B-SPPO Iter2
Llama-3-8B-SPPO Iter3

66.04

63.14
64.16
65.19

64.68
62.12
62.97

65.02
65.53
65.36

62.29

63.82
64.93
65.19

66.85

70.86

68.39
67.84
67.89

68.60
66.30
67.12

69.40
69.55
69.97

51.65

54.96
56.48
58.04

77.98

77.74

77.19
76.09
77.27

77.98
77.51
77.51

77.82
77.03
76.80

76.09

76.40
76.87
77.11

41.93

36.77

40.33
39.95
32.30

43.75
39.20
37.45

43.82
44.35
42.68

75.89

75.44
75.13
74.91

84.89

85.64

85.25
85.23
85.49

85.08
83.15
83.69

85.11
85.29
85.16

78.73

79.80
80.39
80.86

59.15

60.83

59.41
59.03
59.00

59.04
59.70
59.57

58.84
58.72
58.45

65.59

65.65
65.67
65.60

65.74

66.31

65.62
65.38
64.52

66.52
64.66
64.72

66.67
66.75
66.40

68.37

69.35
69.91
70.29

The results, presented in Table 3, demonstrate that SPPO can enhance the performance of the
base model on Arc, TruthfulQA, and GSM8k, and achieve the state-of-the-art performance with an
averagte score of 66.75. However, these improvements do not hold in subsequent alignment iterations:
DPO, IPO, and SPPO’s performance declines after the first or second iterations. This limitation
may be attributed to the “alignment tax” phenomenon (Askell et al., 2021), which suggests that
aligning with human preferences (simulated by PairRM preference in our study) might not improve
or even hurt the general performance. Improving language model capabilities through alignment
iterations remains a topic for future research, and we posit that incorporating high-quality SFT
annotations (Chen et al., 2024) could play a significant role in this endeavor.

Evaluation using PairRM as a judge As SPPO identifies the von Neumann winner (see (3.3))
in a two-player constant-sum game, we examine the pairwise preferences among SPPO models and
other baselines. The pairwise win rates, measured by PairRM, are depicted in Figure 3. We observe
that in all algorithms—namely DPO, IPO, and SPPO—the newer model iterations surpass the
previous ones. For example, SPPO Iteration 3 outperforms SPPO Iteration 2. Both SPPO and IPO
consistently outperform DPO across all iterations. While SPPO is superior to IPO in the first two
iterations, IPO exceeds SPPO in performance during the final iteration. Considering the superior
performance of SPPO in standard benchmarks evaluated by GPT-4 or against ground-truth answers
(e.g., AlpacaEval 2.0, MT-Bench, and Open LLM Leaderboard), along with IPO’s tendency to

19

Figure 3: Pairwise win rates among base model (Mistral-7B-Instruct-v0.2), DPO models, IPO
models, and SPPO models using PairRM-0.4B as a judge, which may favor models with longer
outputs. On benchmarks with more powerful judge models (e.g., GPT-4), such as AlpacaEval 2.0
and MT-Bench, SPPO outperforms other baseline algorithms by a large margin.

produce longer sequence outputs (see Avg. Len in Table 1), we believe this is due to IPO exploiting
the length bias in PairRM that favors longer sequences. Conversely, SPPO models benefit from a
more robust regularization within a multiplicative weight update framework.

5.3 Ablation Study

We study the effect of mini-batch size when estimating the win rate P(y ≻ πt|x). Specifically, for
each prompt, we still generate 5 responses and choose the winner yw and loser yl according to the
PairRM score. When estimating the probability, we varies the batch size to be K = 2, 3, 5. For
K = 2, we estimate P(y ≻ πt|x) with only 2 samples yw and yl:

(cid:98)P (yw ≻ πt|x) =

P(yw ≻ yw|x) + P(yw ≻ yl|x)
2

=

1/2 + P(yw ≻ yl|x)
2

,

and (cid:98)P (yl ≻ πt|x) similarly. K = 5 indicates the original setting we use.

We compare the results on AlpacaEval 2.0, as shown in Figure 4. We find that the performance
of SPPO is robust to the noise in estimating P(y ≻ πt|x). While K = 5 initially outperforms

20

0.5000.5520.5720.5770.6310.6560.6640.7190.7130.7410.8090.4480.5000.5200.5230.5770.6010.6270.6760.6820.7060.7810.4280.4800.5000.4960.5680.5800.6090.6590.6520.6730.7580.4230.4770.5040.5000.5520.5770.6080.6360.6310.6670.7500.3690.4230.4320.4480.5000.5170.5510.5950.5980.6310.7220.3440.3990.4200.4230.4830.5000.5300.5740.5770.6130.7070.3360.3730.3910.3920.4490.4700.5000.5500.5620.5870.6700.2810.3240.3410.3640.4050.4260.4500.5000.5140.5400.6510.2870.3180.3480.3690.4020.4230.4380.4860.5000.5260.6330.2590.2940.3270.3330.3690.3870.4130.4600.4740.5000.6130.1910.2190.2420.2500.2780.2930.3300.3490.3670.3870.500IPO Iter3SPPO Iter3Snorkel (Mistral-PairRM-DPO)DPO Iter3SPPO Iter2IPO Iter2DPO Iter2SPPO Iter1DPO Iter1IPO Iter1Mistral-7B-Instruct-v0.2IPO Iter3SPPO Iter3Snorkel (Mistral-PairRM-DPO)DPO Iter3SPPO Iter2IPO Iter2DPO Iter2SPPO Iter1DPO Iter1IPO Iter1Mistral-7B-Instruct-v0.20.20.30.40.50.60.70.8Mini-Batch
Size

Iteration

AlpacaEval 2.0

Win Rate
LC.

Raw

Avg. Len
(chars)

K = 2

K = 5

Iter1
Iter2
Iter3

Iter1
Iter2
Iter3

23.85
26.91
28.26

23.53
27.24
28.22

23.51
24.79
26.89
27.62
28.53 31.02

1948
1999
1961

1855
2019
2163

Figure 4: AlpacaEval 2.0 evaluation on SPPO of different mini-batch size in terms of both normal
and length-controlled (LC) win rates in percentage (%). K = 2, 5 denote different mini-batch sizes
when estimating the win rate P(y ≻ πt|x).

K = 2 in the first iteration, the difference in their performance diminishes in subsequent iterations.
Additionally, we observe that K = 2 exhibits a reduced tendency to increase output length.

6 Conclusions

This paper introduced Self-Play Preference Optimization(SPPO), an approach to fine-tuning Large
Language Models (LLMs) from Human/AI Feedback. SPPO has demonstrated significant im-
provements over existing methods such as DPO and IPO across multiple benchmarks, including
AlpacaEval 2.0, MT-Bench, Arena-Hard, and the Open LLM Leaderboard. By integrating a prefer-
ence model and employing a new optimization objective, SPPO can align LLMs more closely with
human preferences.
Limitations Theoretically, approximating the optimal policy update via regression relies on the
assumption that the model class is expressive enough and the generated data well cover the input
space. Approximating the log-partition factor with a constant can help reduce variance only when it
is close to the soft value function. The experiments are run on one dataset UltraFeedback and the
models are tested on a few benchmarks due to limited computational resources, but the proposed
methods can be further validated on more models, datasets, and benchmarks to have a holistic
evaluation given more resources.

Acknowledgement

We would like to thank Alekh Agarwal for his insightful comments that helped clarify the technical
contributions of our work and its connection and distinction from SPO (Swamy et al., 2024). We
would also like to thank Wen Sun for pointing out the concurrent work (Gao et al., 2024) and for
the valuable discussion on the design choice of objective functions.

21

Iter1Iter2Iter3182022242628LC. Win Rate (%)Snorkel (Mistral-PairRM-DPO)Mistral-7B-Instruct-v0.2SPPO (K=2)SPPO (K=5)A Approximating the Normalizing Factor

As discussed before, we replace the log-partition factor with a constant to avoid either estimating
or predicting the log-partition factor. In hindsight, the approximation of the normalizing factor
serves as a baseline for variance reduction, and does not need to be exact. Here we discuss the
implicit assumptions and how we obtained an approximation based on different assumptions on
human preference behaviour.

We first consider the case where we have K responses and then calculate the limit of Z

when K → ∞. We have two extreme cases:

(x)

(cid:98)πK

t

1. The most “disordered” case: any preference is a fair coin flip

2. The most “ordered” case: there is a strict ordering among all responses.

The most “disordered” case Specifically, we have K different responses y1, y2, . . . , yK for the
given prompt x. Since we consider the general preference setting, we assume that the preference
probability between yi and yj (i < j) we observe is a fair coin toss:

P(yi ≻ yj|x) =

(cid:40)

1, w.p. 1/2,
0, w.p. 1/2.

Note that for simplicity, we assumed that the preference probability follows the Bernoulli distribution,
not the preference feedback. The preference feedback is deterministic since the preference probability
is either 0 or 1. Assuming P(yi ≻ yj|x) follows any other 1/2-mean distribution will yield the same
constant.

We define the random variable pi,j := 2P(yi ≻ yj|x) − 1 for convenience. In total, we have
K(K − 1)/2 independent Rademacher random variables for all i < j, and then we have pj,i = −pi,j
for all i > j. For i = j, pi,j = 0. We also define Xi = (cid:80)K
Given the setting and notations above, we have

j=1 pi,j/K.

P(yi ≻ (cid:98)πK

t |x) =

K
(cid:88)

j=1

P(yi ≻ yj|x)/K = 1/2 + Xi.

Furthermore,

Z

(cid:98)πK

t

(x) =

K
(cid:88)

i=1

exp(ηP(yi ≻ (cid:98)πK

t |x))/K = eη/2 ·

K
(cid:88)

i=1

eηXi/K.

For any fixed i, we have the expectation as follows:

E[eηXi] = E

(cid:34) K
(cid:89)

(cid:35)

eηpi,j /K

=

j=1

K
(cid:89)

j=1

(cid:104)
eηpi,j /K(cid:105)
E

=

(cid:18) eη/K + e−η/K
2

(cid:19)K−1

,

where the last equation comes from the definition of pi,j (note that pi,i = 0). The variance is:

Var[eηXi] = E[e2ηXi] − E[eηXi]2 =

(cid:18) e2η/K + e−2η/K
2

(cid:19)K−1

−

(cid:18) eη/K + e−η/K
2

(cid:19)2K−2

.

22

Additionally, the covariance between eηXi and eηXj (i ̸= j) is:

Cov(eηXi, eηXj ) = E[eηXi+ηXj ] − E[eηXi]E[eηXj ]
K
(cid:88)

K
(cid:88)

(cid:34)
= E

(cid:32)
η

exp

pi,k/K + η

(cid:33)(cid:35)

pj,l/K

− E[eηXi]E[eηXj ]

k=1
(cid:18) eη/K + e−η/K
2
(cid:18) eη/K + e−η/K
2

=

=

l=1

(cid:19)2K−4

− E[eηXi]E[eηXj ]

(cid:19)2K−4

−

(cid:18) eη/K + e−η/K
2

(cid:19)2K−2

,

where the third line holds because pi,i = pj,j = 0, pi,j + pj,i = 0, and the rest terms are i.i.d..

One can check that when K → ∞, we have E[eηXi] → 1, Var[eηXi] → 0, and Cov(eηXi, eηXj ) → 0.

By Chebyshev’s inequality, (cid:80)K

i=1 eηXi/K will converge to 1 in probability. So we have

(x) = eη/2 ·

Z

(cid:98)πK

t

K
(cid:88)

i=1

eηXi/K → eη/2,

and we can approximate log Z

(x) with η/2.

(cid:98)πK

t

The most “ordered” case We assume there is an ordering σ(·) among the K different responses
y1, y2, . . . , yK for the given prompt x. The preference probability between yi and yj (i < j) is:

P(yi ≻ yj|x) =

(cid:40)

1, if σ(i) < σ(j),
0, if σ(i) > σ(j).

Again, the preference feedback is deterministic: as long as yi is ranked higher than yj, yi will always
be preferred over yj. The same responses still tie: P(yi ≻ yi|x) = 1/2.

Without loss of generality, we can assume y1 ≺ y2 ≺ y3 ≺ · · · ≺ yK. Given the setting and

notations above, we have

P(yi ≻ (cid:98)πK

t |x) =

K
(cid:88)

j=1

P(yi ≻ yj|x)/K =

i − 1 + 1/2
K

=

i − 1/2
K

,

because for yi, there are i − 1 responses that are strictly worse, and yi ties with itself.

For the normalizing factor, we have

log Z

(cid:98)πK

t

(x) = log

= log

(cid:18) K
(cid:88)

i=1
(cid:18) K
(cid:88)

exp(ηP(y ≻ (cid:98)πK

t |x))/K

(cid:19)

(cid:18)
η

exp

i − 1/2
K

(cid:19)

(cid:19)

/K

(cid:19)

exp(ηx)dx

→ log

= log

i=1
(cid:18) (cid:90) 1

0
eη − 1
η

.

23

where the third line (limiting) can be obtained by the squeeze theorem.

For η = 1, log eη−1

η ≈ 0.54η. For large η ≈ 1e3 as we used in the experiments, we have

log eη−1

η ≈ η.

Choice of η Depending on how “disordered” the preference is, η can vary between η/2 and η. As
this paper is partially motivated by human intransitive and irrational preference behavior,
we chose to use η/2 to approximate log Z
(x). Fine-tuning the coefficient of this constant as a
(cid:98)πK
t
hyperparameter is also an option and can help improve performance on given dataset.

B Proof of Theorem 4.1

Proof of Theorem 4.1. Suppose the optimization problem is realizable, we have exactly that

πt+1(y|x) ∝ πt(y|x) exp(ηP(y ≻ πt|x)), for t = 1, 2, . . . .

(B.1)

To prove that the exponential weight update can induce the optimal policy, we directly invoke a
restated version of Theorem 1 in Freund and Schapire (1999):

Lemma B.1 (Theorem 1 in Freund and Schapire (1999), restated). For any oracle P and for any
sequence of mixed policies µ1, µ2, . . . , µT , the sequence of policies π1, π2, . . . , πT produced by (B.1)
satisfies:

T
(cid:88)

t=1

P(πt ≺ µt) ≤ min
π

(cid:20)

η
1 − e−η

T
(cid:88)

t=1

P(π ≺ µt) +

KL(π∥π0)
1 − e−η

(cid:21)
.

By setting µt = πt, we have that

T
2

≤ min

π

(cid:20)

ηT
1 − e−η

P(π ≺ ¯πT ) +

KL(π∥π0)
1 − e−η

(cid:21)

,

where the LHS comes from that P(πt ≺ πt) = 1/2 and the RHS comes from that 1
T
πt) = P(π ≺ ¯πt). Now rearranging terms gives

(cid:80)T

t=1

P(π ≺

1 − e−η
2η

≤ min

π

(cid:20)
P(π ≺ ¯πT ) +

KL(π∥π0)
ηT

(cid:21)

.

Note that π0 is an autoregressive model that is fully supported on a finite vocabulary (π0(yk+1|x, y1:k)
has non-zero probability for every token). Because its support is a large but finite set, | log π0(·)| is
bounded from above. So we can naively bound the KL-divergence KL(π∥π0) ≤ ∥ log π0(·)∥∞, which
can be seen as a (large) constant.
By choosing η = ∥ log π0(·)∥∞

, we have

√

T

1
2

−

∥ log π0(·)∥∞
√
4

T

+ O(T −1) ≤ min

π

(cid:2)P(π ≺ ¯πT )(cid:3) +

(cid:114)

∥ log π0(·)∥∞
T

,

where the LHS comes from Taylor’s expansion 1−e−η
4 + O(η2). Notice that 1/2 at the
LHS is already the value of the symmetric two-player constant-sum game. This shows that for

2η = 1

2 − η

24

appropriately chosen η and T , the mixture policy ¯πT is close to the minimax optimal policy (Nash
equilibrium).

The optimality gap is thus bounded by

max
π

(cid:2)P(π ≻ ¯πT )(cid:3) − min

(cid:2)P(π ≺ ¯πT )(cid:3)

π

(cid:2)1 − P(π ≺ ¯πT )(cid:3) − min

(cid:2)P(π ≺ ¯πT )(cid:3)

π
(cid:19)
(cid:2)P(π ≺ ¯πT )(cid:3)

= 2

= max
π
(cid:18) 1
2
(cid:18) 1
√
T

= O

− min
π
(cid:19)

.

C Response Examples in Different Iterations

25

Table 4: Generation example of our fine-tuned model by SPPO at different iterations.

Prompt

You will be given a definition of a task first, then some input of the task. You are
given a dialog between 2 or more individuals. The dialog will include information
as to the relationship between 2 of the participants. At the end, you will be asked
to provide an answer as to the relationship of those 2 participants. In this scenario,
’0’ will represent a spousal relationship, ’1’ will represent siblings, and ’2’ will
represent a child/parent relationship. Speaker 1: I’ll be right with you. Okay?
Thanks, Wendy. Speaker 2: Oh my God! How cute is the on-call doctor? Speaker
3: Ooh, so cute, that I’m thinking about jamming this pen in my eye. Speaker 4:
Dr. Burke will see you know. Speaker 2: Oh no-no-no, not Dr. Burke. Dr. Burke
is out of town. The-the on-call doctor will see me now. Speaker 4: Dr. Richard
Burke is out of town. Dr. Timothy Burke, his son, will see you now. Speaker 5:
Ready? What is the relationship between Speaker 5 and Richard Burke? Output:

Mistral-7B The relationship between Speaker 5 and Richard Burke is that of a parent and
child, as Richard Burke is identified as the son of another character in the dialog.

Iteration 1

Iteration 2

Iteration 3

The relationship between Speaker 5 and Richard Burke is that of a father and
child. Based on the dialogue, it was revealed that Dr. Timothy Burke is the on-call
doctor and the son of Dr. Richard Burke.

The relationship between Speaker 5 and Richard Burke is that of a father and
child. The conversation reveals that Dr. Timothy Burke is the on-call doctor and
he is identified as the son of Dr. Richard Burke.

Speaker 5 is the individual asking about the relationship, so they are not directly
related to Dr. Richard Burke. The information provided in the dialog reveals
that Dr. Timothy Burke is the on-call doctor and the son of Dr. Richard Burke.
Therefore, the answer is: The relationship between Speaker 5 and Dr. Richard
Burke is that of a child (Dr. Timothy Burke) and parent. Speaker 5 is inquiring
about the relationship, but they are not themselves a part of it. The specified
relationship codes (0 for spousal, 1 for sibling, 2 for child/parent) align with
this description: Speaker 5 (0, 1, or 2) and Dr. Richard Burke (2) connect as a
parent-child duo. Output: 2 (child/parent relationship) In this scenario, Speaker
5’s role is to determine the relationship between herself and Dr. Richard Burke,
but since they are not related (she is an outside inquirer), the correct answer
pertains only to the relationship identified within the dialog: Dr. Richard Burke
(father) and Dr. Timothy Burke (son).

26

Table 5: Another generation example of our fine-tuned model by SPPO at different iterations.

Prompt

Q: Context: Roman Brady is a fictional character from the long running NBC soap
opera, Days of Our Lives. The role was originated in 1981 by Wayne Northrop
who played the role until 1984, was filled by Drake Hogestyn from 1986 – 1991,
again by Northrop from 1991 – 1994, and is currently being portrayed by series
veteran Josh Taylor, who stepped into the role in 1997. Question: who is roman
brady on days of our lives? Let’s give stream of consciousness below

Mistral-7B Roman Brady is a fictional character on the NBC soap opera ”Days of Our Lives.”
He was first portrayed by Wayne Northrop from 1981 to 1984. Later, Drake
Hogestyn took over the role from 1986 to 1991. Northrop returned to the role from
1991 to 1994. Since 1997, the character is being portrayed by Josh Taylor. Roman
Brady is a significant character in the show’s long-running storylines.

Iteration 1

Iteration 2

Iteration 3

Roman Brady is a fictional character on the NBC soap opera ”Days of Our Lives.”
He was first portrayed by Wayne Northrop from 1981 to 1984. Drake Hogestyn
took over the role from 1986 to 1991. Northrop returned to play Roman again
from 1991 to 1994. Since 1997, the character has been portrayed by Josh Taylor.
So, in summary, Roman Brady is a long-standing character on ”Days of Our Lives”
who has been played by three different actors over the years: Wayne Northrop
(1981-1984, 1991-1994), Drake Hogestyn (1986-1991), and currently by Josh Taylor
(1997-present).

Roman Brady is a fictional character on the NBC soap opera ”Days of Our Lives.”
He was first portrayed by Wayne Northrop from 1981 to 1984. Drake Hogestyn
took over the role from 1986 to 1991. Northrop returned to play Roman again
from 1991 to 1994. Since 1997, the character has been portrayed by Josh Taylor.
Overall, Roman Brady is a long-standing and complex figure in the show’s history,
with multiple actors bringing their unique interpretations to the role.

Roman Brady is a fictional character on the NBC soap opera ”Days of Our Lives.”
He was originally played by Wayne Northrop from 1981 to 1984. Drake Hogestyn
took over the role from 1986 to 1991. Northrop returned to portray Roman
once more from 1991 to 1994. Since 1997, the character has been portrayed by
Josh Taylor.Roman Brady is a complex and prominent figure in the long-running
daytime drama. Initially played by Northrop in the early 1980s, his tenure included
involvement in various storylines, such as romantic relationships and business
dealings. Hogestyn’s portrayal during the late 1980s and early 1990s introduced
new dimensions to the character. Northrop’s second stint, from 1991 to 1994,
further developed Roman’s narrative. Since 1997, Josh Taylor has assumed the
role, continuing to contribute to the evolving story of Roman Brady on ”Days of
Our Lives.” Throughout these different portrayals, Roman has remained a central
character, known for his intricate relationships, business ventures, and personal
struggles.

27

References

Askell, A., Bai, Y., Chen, A., Drain, D., Ganguli, D., Henighan, T., Jones, A., Joseph,
N., Mann, B., DasSarma, N. et al. (2021). A general language assistant as a laboratory for
alignment. arXiv preprint arXiv:2112.00861 .

Azar, M. G., Rowland, M., Piot, B., Guo, D., Calandriello, D., Valko, M. and Munos,
R. (2023). A general theoretical paradigm to understand learning from human preferences. arXiv
preprint arXiv:2310.12036 .

Beeching, E., Fourrier, C., Habib, N., Han, S., Lambert, N., Rajani, N., Sanseviero,
O., Tunstall, L. and Wolf, T. (2023a). Open llm leaderboard. https://huggingface.co/
spaces/HuggingFaceH4/open_llm_leaderboard.

Beeching, E., Fourrier, C., Habib, N., Han, S., Lambert, N., Rajani, N., Sanseviero, O.,

Tunstall, L. and Wolf, T. (2023b). Open llm leaderboard. Hugging Face .

Bradley, R. A. and Terry, M. E. (1952). Rank Analysis of Incomplete Block Designs: I. The

Method of Paired Comparisons. Biometrika 39 324–345.

Calandriello, D., Guo, D., Munos, R., Rowland, M., Tang, Y., Pires, B. A., Richemond,
P. H., Lan, C. L., Valko, M., Liu, T. et al. (2024). Human alignment of large language
models through online preference optimisation. arXiv preprint arXiv:2403.08635 .

Chen, Z., Deng, Y., Yuan, H., Ji, K. and Gu, Q. (2024). Self-play fine-tuning converts weak

language models to strong language models. arXiv preprint arXiv:2401.01335 .

Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S. and Amodei, D. (2017).
Deep reinforcement learning from human preferences. Advances in neural information processing
systems 30.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C. and Tafjord,
O. (2018). Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv
preprint arXiv:1803.05457 .

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M.,
Tworek, J., Hilton, J., Nakano, R. et al. (2021). Training verifiers to solve math word
problems. arXiv preprint arXiv:2110.14168 .

Cui, G., Yuan, L., Ding, N., Yao, G., Zhu, W., Ni, Y., Xie, G., Liu, Z. and Sun, M.
(2023). Ultrafeedback: Boosting language models with high-quality feedback. arXiv preprint
arXiv:2310.01377 .

Dubois, Y., Galambosi, B., Liang, P. and Hashimoto, T. B. (2024a). Length-controlled
alpacaeval: A simple way to debias automatic evaluators. arXiv preprint arXiv:2404.04475 .

Dubois, Y., Li, C. X., Taori, R., Zhang, T., Gulrajani, I., Ba, J., Guestrin, C., Liang,
P. S. and Hashimoto, T. B. (2024b). Alpacafarm: A simulation framework for methods that
learn from human feedback. Advances in Neural Information Processing Systems 36.

28

Dud´ık, M., Hofmann, K., Schapire, R. E., Slivkins, A. and Zoghi, M. (2015). Contextual

dueling bandits. In Conference on Learning Theory. PMLR.

Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D. and Kiela, D. (2024). Kto: Model

alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306 .

Freund, Y. and Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning

and an application to boosting. Journal of computer and system sciences 55 119–139.

Freund, Y. and Schapire, R. E. (1999). Adaptive game playing using multiplicative weights.

Games and Economic Behavior 29 79–103.

Gao, L., Schulman, J. and Hilton, J. (2023). Scaling laws for reward model overoptimization.

In International Conference on Machine Learning. PMLR.

Gao, Z., Chang, J. D., Zhan, W., Oertell, O., Swamy, G., Brantley, K., Joachims, T.,
Bagnell, J. A., Lee, J. D. and Sun, W. (2024). Rebel: Reinforcement learning via regressing
relative rewards. arXiv preprint arXiv:2404.16767 .

Haarnoja, T., Zhou, A., Abbeel, P. and Levine, S. (2018). Soft actor-critic: Off-policy
maximum entropy deep reinforcement learning with a stochastic actor. In International conference
on machine learning. PMLR.

He, P., Gao, J. and Chen, W. (2021). Debertav3:

Improving deberta using electra-style

pre-training with gradient-disentangled embedding sharing.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D. and Steinhardt, J.
(2020). Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300 .

Hong, J., Lee, N. and Thorne, J. (2024). Reference-free monolithic preference optimization with

odds ratio. arXiv preprint arXiv:2403.07691 .

Ji, K., He, J. and Gu, Q. (2024). Reinforcement learning from human feedback with active queries.

arXiv preprint arXiv:2402.09401 .

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D.
d. l., Bressand, F., Lengyel, G., Lample, G., Saulnier, L. et al. (2023a). Mistral 7b.
arXiv preprint arXiv:2310.06825 .

Jiang, D., Ren, X. and Lin, B. Y. (2023b). Llm-blender: Ensembling large language models with

pairwise ranking and generative fusion. arXiv preprint arXiv:2306.02561 .

Li, J., Sun, S., Yuan, W., Fan, R.-Z., Zhao, H. and Liu, P. (2023a). Generative judge for

evaluating alignment. arXiv preprint arXiv:2310.05470 .

Li, T., Chiang, W.-L., Frick, E., Dunlap, L., Wu, T., Zhu, B., Gonzalez, J. E. and Stoica,
I. (2024). From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder
pipeline. arXiv preprint arXiv:2406.11939 .

Li, X., Zhang, T., Dubois, Y., Taori, R., Gulrajani, I., Guestrin, C., Liang, P. and
Hashimoto, T. B. (2023b). Alpacaeval: An automatic evaluator of instruction-following models.
https://github.com/tatsu-lab/alpaca_eval.

29

Lin, S., Hilton, J. and Evans, O. (2021). Truthfulqa: Measuring how models mimic human

falsehoods. arXiv preprint arXiv:2109.07958 .

Liu, T., Zhao, Y., Joshi, R., Khalman, M., Saleh, M., Liu, P. J. and Liu, J. (2023).
Statistical rejection sampling improves preference optimization. arXiv preprint arXiv:2309.06657
.

Lou, H., Jin, T., Wu, Y., Xu, P., Gu, Q. and Farnoud, F. (2022). Active ranking without

strong stochastic transitivity. Advances in neural information processing systems .

Munos, R., Valko, M., Calandriello, D., Azar, M. G., Rowland, M., Guo, Z. D., Tang,
Y., Geist, M., Mesnard, T., Michi, A. et al. (2023). Nash learning from human feedback.
arXiv preprint arXiv:2312.00886 .

Ng, A. Y., Harada, D. and Russell, S. (1999). Policy invariance under reward transformations:

Theory and application to reward shaping. In Icml, vol. 99.

OpenAI, J., Achiam, Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L.,
Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S. et al. (2023). Gpt-4 technical
report. arXiv preprint arXiv:2303.08774 .

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C.,
Agarwal, S., Slama, K., Ray, A. et al. (2022). Training language models to follow instructions
with human feedback. Advances in Neural Information Processing Systems 35 27730–27744.

Pal, A., Karkhanis, D., Dooley, S., Roberts, M., Naidu, S. and White, C. (2024). Smaug:
Fixing failure modes of preference optimisation with dpo-positive. arXiv preprint arXiv:2402.13228
.

Rafailov, R., Hejna, J., Park, R. and Finn, C. (2024a). From r to q*: Your language model is

secretly a q-function. arXiv preprint arXiv:2404.12358 .

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S. and Finn, C. (2024b).
Direct preference optimization: Your language model is secretly a reward model. Advances in
Neural Information Processing Systems 36.

Rosset, C., Cheng, C.-A., Mitra, A., Santacroce, M., Awadallah, A. and Xie, T. (2024).
Direct nash optimization: Teaching language models to self-improve with general preferences.
arXiv preprint arXiv:2404.03715 .

Sakaguchi, K., Bras, R. L., Bhagavatula, C. and Choi, Y. (2021). Winogrande: An

adversarial winograd schema challenge at scale. Communications of the ACM 64 99–106.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. (2017). Proximal

policy optimization algorithms. arXiv preprint arXiv:1707.06347 .

Singh, A., Co-Reyes, J. D., Agarwal, R., Anand, A., Patil, P., Liu, P. J., Harrison,
J., Lee, J., Xu, K., Parisi, A. et al. (2023). Beyond human data: Scaling self-training for
problem-solving with language models. arXiv preprint arXiv:2312.06585 .

30

Sutton, R. S., McAllester, D., Singh, S. and Mansour, Y. (1999). Policy gradient methods
for reinforcement learning with function approximation. Advances in neural information processing
systems 12.

Swamy, G., Dann, C., Kidambi, R., Wu, Z. S. and Agarwal, A. (2024). A minimaximalist
approach to reinforcement learning from human feedback. arXiv preprint arXiv:2401.04056 .

Tversky, A. (1969). Intransitivity of preferences. Psychological review 76 31.

Wang, Y., Liu, Q. and Jin, C. (2024). Is rlhf more difficult than standard rl? a theoretical

perspective. Advances in Neural Information Processing Systems 36.

Wu, Y., Jin, T., Di, Q., Lou, H., Farnoud, F. and Gu, Q. (2023). Borda regret minimization for
generalized linear dueling bandits. In ICML 2023 Workshop The Many Facets of Preference-Based
Learning.

Xiong, W., Dong, H., Ye, C., Zhong, H., Jiang, N. and Zhang, T. (2023). Gibbs sampling from
human feedback: A provable kl-constrained framework for rlhf. arXiv preprint arXiv:2312.11456 .

Xu, J., Lee, A., Sukhbaatar, S. and Weston, J. (2023). Some things are more cringe than
others: Preference optimization with the pairwise cringe loss. arXiv preprint arXiv:2312.16682 .

Ye, C., Xiong, W., Zhang, Y., Jiang, N. and Zhang, T. (2024). A theoretical analysis
of nash learning from human feedback under general kl-regularized preference. arXiv preprint
arXiv:2402.07314 .

Yuan, W., Pang, R. Y., Cho, K., Sukhbaatar, S., Xu, J. and Weston, J. (2024). Self-

rewarding language models. arXiv preprint arXiv:2401.10020 .

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A. and Choi, Y. (2019). Hellaswag: Can a

machine really finish your sentence? arXiv preprint arXiv:1905.07830 .

Zhao, Y., Joshi, R., Liu, T., Khalman, M., Saleh, M. and Liu, P. J. (2023). Slic-hf: Sequence

likelihood calibration with human feedback. arXiv preprint arXiv:2305.10425 .

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li,
D., Xing, E. et al. (2023). Judging llm-as-a-judge with mt-bench and chatbot arena. Advances
in Neural Information Processing Systems 36.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li,
D., Xing, E. et al. (2024). Judging llm-as-a-judge with mt-bench and chatbot arena. Advances
in Neural Information Processing Systems 36.

Zhu, B., Jiao, J. and Jordan, M. I. (2023). Principled reinforcement learning with human

feedback from pairwise or k-wise comparisons. arXiv preprint arXiv:2301.11270 .

31

