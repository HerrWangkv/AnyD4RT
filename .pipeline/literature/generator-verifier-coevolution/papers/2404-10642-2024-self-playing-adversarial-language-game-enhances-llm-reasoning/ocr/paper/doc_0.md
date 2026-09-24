Self-playing Adversarial Language Game
Enhances LLM Reasoning

Pengyu Cheng1 Tianhao Hu1 Han Xu1 Zhisong Zhang1
Zheng Yuan4 Yong Dai1 Lei Han3 Nan Du1 Xiaolong Li2

Tencent AI Lab 1Shenzhen & 2Seattle

3Tencent Robotics X Lab

4Independent

pengyucheng@tencent.com

Abstract
We explore the potential of self-play training for large language models (LLMs) in
a two-player adversarial language game called Adversarial Taboo. In this game,
an attacker and a defender communicate around a target word only visible to
the attacker. The attacker aims to induce the defender to speak the target word
unconsciously, while the defender tries to infer the target word from the attacker’s
utterances. To win the game, both players must have sufficient knowledge about the
target word and high-level reasoning ability to infer and express in this information-
reserved conversation. Hence, we are curious about whether LLMs’ reasoning
ability can be further enhanced by Self-Playing this Adversarial language Game
(SPAG). With this goal, we select several open-source LLMs and let each act
as the attacker and play with a copy of itself as the defender on an extensive
range of target words. Through reinforcement learning on the game outcomes,
we observe that the LLMs’ performances uniformly improve on a broad range of
reasoning benchmarks. Furthermore, iteratively adopting this self-play process
can continuously promote LLMs’ reasoning abilities. The code is available at
https://github.com/Linear95/SPAG.

5
2
0
2

n
a
J

4
2

]
L
C
.
s
c
[

3
v
2
4
6
0
1
.
4
0
4
2
:
v
i
X
r
a

Figure 1: Reasoning improvements from Self-Playing of Adversarial language Games (SPAG) on
comprehensive reasoning benchmarks. With the SPAG epoch increasing, the LLM reasoning ability
continuously improves. Each axis is normalized by the maximum answer-accuracy value.

1

Introduction

Large language models (LLMs), such as GPT-4 [OpenAI, 2023b] and Gemini [Team et al., 2023],
have reformed the domain of artificial intelligence (AI) with astonishing language capacities, such as

38th Conference on Neural Information Processing Systems (NeurIPS 2024).

 
 
 
 
 
 
Figure 2: Examples of Adversarial Taboo with the same target word “conversation”. The left shows
an attacker-winning game, in which the defender unconsciously speaks out the target word. The right
is a defender-winning episode because the defender makes the correct inference from the dialogue.

natural language understanding [Yang et al., 2023b, Touvron et al., 2023], text generation [Koco´n
et al., 2023, Anil et al., 2023], machine translation [Jiao et al., 2023], summarization [Xie et al.,
2024], and programming [Surameery and Shakor, 2023, Tian et al., 2023]. However, the reasoning
ability of LLMs, which is essential for complex problem-solving [Pan et al., 2023] and advanced
intelligence-developing [Yao et al., 2021], still retains being challenged by various criteria including
correctness [Zhang et al., 2023a] and faithfulness [Turpin et al., 2024].

To address the reasoning challenge of LLMs, plenty of works have contributed in-depth efforts
from the perspectives of Chain-of-Thought (CoT) prompt engineering [Wei et al., 2022, Ding et al.,
2023, Yao et al., 2024], and the usage of auxiliary reasoning tools [Pan et al., 2023]. However, both
prompt-based and tool-calling methods require additional prompt designs, which are inconsistent and
sensitive to different prompt patterns and LLM checkpoints [Turpin et al., 2024, Chu et al., 2023].
More fundamental and consistent reasoning-improving approaches are post-pretraining [Azerbayev
et al., 2023] and fine-tuning [Dong et al., 2023a], which trains LLMs with additional reasoning-related
text corpus. Nevertheless, these methods demand sufficient high-quality textual data, which are
difficult to collect due to the massive costs of human annotation efforts [Singh et al., 2023].

To improve LLM reasoning more efficiently, self-improvement methods, which enhance LLMs
with model-generated synthetic data, have recently attracted increasing research attention [Singh
et al., 2023, Huang et al., 2023, Burns et al., 2023, Chen et al., 2024]. Self-improvement methods
usually utilize the intrinsic language capability of LLMs to judge [Huang et al., 2023], filter [Yuan
et al., 2024], or revise [Yuan et al., 2024] self-generated samples to enhance their quality. However,
most self-improvement methods rely on a broad range of high-quality question queries to prevent
over-fitting into a sub-domain of reasoning tasks, which still requires additional data collection and
cleaning. Besides, the judgments from LLMs are not guaranteed objective [Raina et al., 2024]. If an
LLM already has an incorrect or biased recognition of a particular concept, the self-improvement
process can reinforce and amplify the LLM’s cognitive dissonance.

Towards more general and objective self-reasoning-improvement methods, we are inspired by the
advancement from AlphaGO [Silver et al., 2016] to AlphaGO Zero [Silver et al., 2017], in which
the game agents’ intelligence continuously promotes via self-play without any human knowledge.
Analogically, we expect to set up a language game where LLMs can improve their reasoning capacities
via reinforcement learning (RL) during self-play. Although language games have attracted increasing
attention in natural language processing [Lewis et al., 2017, Hausknecht et al., 2020, Xu et al.,
2023, Wu et al., 2024], most of them are specially designed with customized game rules, in lack
of the generalization to improve the general language capacities of LLMs. Among a few general-
target language games including red-teaming [Ma et al., 2023], negotiation [Lewis et al., 2017],
and bargain [Abdulhai et al., 2023], additional human judgments or reward models are required for
outcome determination, which posts challenges on the efficiency and effectiveness of large-scale
self-play RL training. Recent studies have raised interest in entity- or word-based language games,
such as 20-Question [Zhang et al., 2023b] and Guess-My-City [Abdulhai et al., 2023] , which provide
not only straight-forward word-level outcomes but also language universality by traversing the game
word from comprehensive vocabularies. However, unlike the GO game, these word-based games are
out of the adversarial scheme, limiting the game intensity and self-play learning effectiveness.

With the above consideration, we select an adversarial language game called Adversarial Taboo [Yao
et al., 2021], in which an attacker and a defender perform a conversation around a target word

2

only visible to the attacker. The attacker aims to induce the defender to speak out the target word
unconsciously; the defender tries to avoid unconscious utterance of the target word and guess the
word from the dialogue history. To win the adversarial game in information-limited conversations,
both players are required to have high-level language capacities in terms of expression, upstanding,
and reasoning. Moreover, by collecting target words from a vast vocabulary, this game can cover
a broad range of topics providing sufficient language versatility. Besides, the game outcomes can
be automatically and explicitly judged: we only need to check whether the target word appears in
the defender’s utterances (attacker wins) or its inference patterns (defender wins). We conduct the
self-play on this adversarial game using open-source LLMs, LLaMA-2-7B [Touvron et al., 2023]
and Baichuan-2-13B [Yang et al., 2023a], with target words selected from a 50K top-frequency
vocabulary [Davies, 2020]. Next, we conduct offline reinforcement learning on the game outcomes
and observe significant performance improvement on a broad range of LLM reasoning benchmarks.
Furthermore, we iterate this sampling-learning process with three epochs, within which the LLMs’
reasoning can continuously obtain improvement. We believe this novel training scheme, Self-Play of
Adversarial Game (SPAG), has great potential for developing advanced LLM capacities.

2 Preliminary

With the development of LLMs [OpenAI, 2023a,b], reinforcement learning (RL) has played an
increasingly important role in language model training [Ouyang et al., 2022, Ramamurthy et al.,
2023]. The prime application scenario for RL in LLM training is reinforcement learning from human
feedback (RLHF) [Yuan et al., 2023, Cheng et al., 2023b, Zeng et al., 2023]. RLHF first learns a
reward model r(x, y) from the human feedback preference pairs [Cheng et al., 2023a], and then
optimizes the LLM policy πθ(y|x) to maximize the expected reward value [Dong et al., 2023b]:
LRLHF(πθ) = −Ex∼D,y∼πθ(y|x)[r(x, y)].
(1)
To learn the above objective, proximal policy optimization (PPO) [Schulman et al., 2017] algorithm
has been recognized as the mainstream solution. In each update to equation 1, PPO minimizes:

LPPO(πθ) = −Ex∼D,y∼π ¯θ(y|x)

(cid:104) πθ(y|x)
π¯θ(y|x)

ˆAπ ¯θ − βKL[π¯θ∥πθ]

(cid:105)
,

(2)

where π¯θ is a copy of πθ before the update, ˆAπ ¯θ is the estimated advantage value [Schulman et al.,
2016] with respect to the reference policy π¯θ, and KL[π¯θ∥πθ] is the Kullback-Leibler (KL) [Kullback,
1997] divergence regularizing πθ with an appropriate updating step. However, PPO for LLMs has
been continually challenged due to its inefficient natural-text online sampling and the unstable training
processes [Baheti et al., 2024]. Among the improvements to PPO [Rafailov et al., 2023, Yuan et al.,
2023, Dong et al., 2023b], Baheti et al. [2024] adopts the PPO objective into an offline scheme by
using importance sampling [Neal, 2001], which is named Advantage-Leftover-Lunch (A-LoL):

∇θ ˆLRLHF-A-LoL = Ex∼D,y∼πref(y|x)

(cid:104) ˆAπref ·

πθ(y|x)
πref(y|x)

(cid:105)
· ∇θ log πθ(y|x)

.

(3)

Here the sample y ∼ πref(y|x) and advantage ˆAπref are both from the reference distribution πref(y|x)
and calculated offline. Besides, Gulcehre et al. [2023] proposed Reinforced Self-Training (ReST) to
simplify the RLHF scheme. With a threshold τ ∈ R, ReST updates the LLM by the reinforcement on
the selected samples Dτ = {(x, y) : r(x, y) > τ }:

LReST(πθ) = Ex∼D,y∼πref(y|x)

(cid:104)

1r(x,y)>τ · Lθ(x, y)

(cid:105)

= EDτ [Lθ(x, y)],

(4)

where Lθ(x, y) could be any offline RL loss such as A-LoL or the vanilla language modeling loss.

3 Self-play of Adversarial Language Games

The game of Adversarial Taboo is first introduced by Yao et al. [2021], in which an attacker µ and a
defender ν involve a multi-turn conversation. At the beginning of the game, the attacker is assigned a
target word w ∈ Vtarget, which is not informed to the defender. The attacker’s target is to induce the
defender to speak the target word unconsciously. To achieve this goal, the attacker can talk about any
topic related to w, except directly speaking out the target word. In contrast, the defender is required
to infer the target word without any unconscious utterance of the word. If the defender has sufficient
confidence to infer the word, it can yell “I know the word! It is {target word}!”. Then the game
terminates. If the guess is correct, the defender wins, otherwise the attacker wins. Besides, the game
has a maximum number of turns T0. If nobody wins during T0 turns, there is a tie. The examples of
Adversarial Taboo are shown in Figure 2.

3

Algorithm 1 Data collection of LLM self-plays for the adversarial language game.

Inputs: LLM policy πθ(y|x), target word w, attacker and defender prompts fattack, fdefend.
Set the initial state s0 = (w).
for t from 1 to T do

Sample an attacker utterance ut ∼ µθ(ut|st−1) = πθ(y = ut|x = fattack(st−1)).
Update state s′
Sample a defender utterance vt ∼ νθ(vt|s′
Update state st = (w, u1, v1, . . . , ut−1, vt−1, ut, vt).

t = (w, u1, v1, . . . , ut−1, vt−1, ut).

t) = πθ(y = vt|x = fdefend(s′

t)).

end for
Collect an episode τ = (s0, s′

1, s1, . . . , s′

T , sT )

3.1 Adversarial Language Game Modeling
We view the Adversarial Taboo as a two-player zero-sum Markov game [Littman, 1994], which can
be described by a tuple as (S, A, F, r):

• The state space S = {st, s′
t

t =
(w, u1, v1, u2, . . . , ut) and st = (w, u1, v1, u2, . . . , ut, vt), where {ui}t
i=1 are
the utterances of the attacker and defender, respectively. Games start at s0 = (w) with a target
word w ∈ Vtarget and end with T0 maximum turns. States s′
t and st end with utterances ut and vt
for the defender and attacker to act, respectively.

: 1 ≤ t ≤ T0} contains two types of states, s′

i=1 and {vi}t

• The action space A is shared with both the attacker and defender, which is equivalent to the token

sequence space of natural language N = {x = (x1, x2, . . . , xL)|xl ∈ Vtoken, L ∈ N+}.

• The transition function F : S × A → S deterministically appends the utterance ut or vt at the

end of the dialogue, and converts s′

t = F (st−1, ut) and st = F (s′

t, vt).

• The reward r : S × A → R evaluates the actions u, v ∈ A based on their corresponding
states s, s′ ∈ S with rewards r(s, u) and r(s′, v), respectively. Given a game episode τ =
(s0, s′
t=1 r(st−1, ut), so the
defender’s total reward is (cid:80)T
t, vt) = −R(τ ) to satisfy the zero-sum constraint. More
detailed reward designs with heuristic rules for the Adversarial Taboo can be found in Appendix B.

T , sT ), we denote the attacker’s total reward R(τ ) = (cid:80)T

1, s1, . . . , s′

t=1 r(s′

In the above game, we denote µ(u|s) and ν(v|s′) as the attacker’s and defender’s policies, respec-
tively. Then each episode τ can be regarded as a trajectory with the probability:

P (τ ) = P (s0)

T
(cid:89)

t=1

P (s′

t|st−1)

T
(cid:89)

t=1

P (st|s′

t) = P (w)

T
(cid:89)

t=1

µ(ut|st−1)

T
(cid:89)

t=1

ν(vt|s′

t) =: (µ × ν), (5)

where P (w) is the data distribution of target word w ∈ Vtarget. Then we can write the self-play
objective of the Adversarial Taboo as:

max
µ

min
ν

LAG(µ, ν) := Eτ ∼µ×ν[R(τ )],

(6)

in which the attacker tries to maximize its total reward R(τ ) by optimizing policy µ, and the
defender seeks strategies ν to maximize the defender reward −R(τ ) (minimize R(τ )). To play the
above adversarial game with an LLM generation policy πθ(y|x), we first design prompt templates
fattack, fdefend : S → N for the attacker and defender respectively to convert the game states into
natural language task descriptions. Next, we introduce the game policies for the two players:

µθ(u|s) = πθ(u|fattack(s)), and νθ(v|s′) = πθ(v|fdefend(s′)).

(7)

The detailed prompt templates for the game are demonstrated in Appendix A.

3.2

Imitation Learning

Due to the limited capability of current open-source LLMs, the generation policy πθ(y|x) can
not guarantee the strict instruction-following of the game rules in prompts fattack(s) and fdefend(s′).
Therefore, before the self-play, we first conduct an imitation learning (behavior cloning) of GPT-4’s
behaviors to ensure that πθ(u|fattack(s)) and πθ(v|fdefend(s′)) act consistently with the game rules.
To collect the game episodes of GPT-4 [Achiam et al., 2023], we use the data collection procedure

4

described in Algorithm 1. Similar to the setups in equation 7, we also design attacker and defender
prompts for GPT-4 to act as the game players, which can be found in Appendix A.1.

After collecting a group of GPT-4 game episodes for imitation learning as Tim, we divide it into an
attacker-winning set T attack
= {τ ∈
Tim : R(τ ) < 0}. The imitation learning loss is to maximize the log-likelihood of winners’ actions:

im = {τ ∈ Tim : R(τ ) > 0} and a defender-winning set T defend

im

im (πθ) = − E
Lattack

τ ∈T attack
im

(cid:104) 1
T

Ldefend
im

(πθ) = − E

τ ∈T defend
im

(cid:104) 1
T

T
(cid:88)

t=1

T
(cid:88)

t=1

(cid:105)
log πθ(ut|fattack(st−1)) + β1KL[πθ∥πref]

,

log πθ(vt|fdefend(s′

t)) + β1KL[πθ∥πref]

(cid:105)
,

(8)

(9)

(πθ).

2 Lattack

2 Ldefend
im

where the re-weighting parameter β1 > 0 and the regularizer KL[πθ∥πref] prevents the model from
over-fitting on the language game task and maintains the model’s general language abilities. The
reference model πref is the initial checkpoint of the LLM before training. The overall imitation
learning objective is Lim(πθ) = 1

im (πθ) + 1
3.3 Reinforcement Learning from Self-play
Imitation learning enables LLM to behave consistently following the game rules. Next, we conduct
self-play by letting the LLM πθ(y|x) play alternatively as the attacker µθ(u|s) = πθ(u|fattack(s))
and the defender νθ(v|s′) = πθ(v|fdefend(s′)). Note that the self-play sampling process involves
massive multi-turn auto-regressive text generation of the LLM, which causes heavy computational
complexity and makes the on-policy RL training intensively inefficient. Therefore, we use an offline
learning scheme: (1) make a copy π¯θ of current LLM policy πθ; (2) collect self-play episodes
T¯θ = {τ ∼ µ¯θ × ν¯θ} from games between attacker µ¯θ and defender ν¯θ; (3) update πθ via RL training
with T¯θ. The details of the collection of T¯θ are shown in Algorithm 1.
With a group of episodes T¯θ, we first fix the defender policy ν¯θ and consider updating the attacker
policy µθ with respect to the self-play objective LAG(µθ, ν¯θ) = Eτ ∼µθ×ν ¯θ [R(τ )]. We calculate the
corresponding policy gradient for the attacker as:

∇θLAG(µθ, ν¯θ) =Eτ ∼µθ×ν ¯θ

=Eτ ∼µ ¯θ×ν ¯θ

(cid:104) T
(cid:88)

t=1
(cid:104) T
(cid:88)

t=1

Aµθ
t

(cid:105)
· ∇θ log µθ(ut|st−1)

Aµθ
t

·

µθ(ut|st−1)
µ¯θ(ut|st−1)

· ∇θ log µθ(ut|st−1)

(cid:105)
,

(10)

where Aµθ
t = Aµθ (st−1, ut) is the advantage of action ut for the attacker µθ in the self-play of
µθ × ν¯θ. Here we apply importance sampling to unbiasedly estimate the expectation w.r.t µθ(ut|st−1)
with the sampled actions from µ¯θ(ut|st−1) in T¯θ. Inspired by TRPO [Schulman et al., 2015] and
PPO [Schulman et al., 2017], we design the following loss to optimize LAG(µθ, ν¯θ):

Lattack
sp

(πθ) = − Eτ ∈T ¯θ

= − Eτ ∈T ¯θ

(cid:104) T
(cid:88)

t=1
(cid:104) T
(cid:88)

t=1

µθ(ut|st−1)
µ¯θ(ut|st−1)

ˆAµ ¯θ

(cid:105)
t − β2KL[πθ∥π¯θ]

πθ(ut|fattack(st−1))
π¯θ(ut|fattack(st−1))

ˆAµ ¯θ

(cid:105)
t − β2KL[πθ∥π¯θ]

,

(11)

where the re-weighting parameter β2 > 0, and regularizer KL(πθ∥π¯θ) guarantees an appropriate
policy update step size. Following TRPO [Schulman et al., 2015], we use empirical estimation ˆAµ ¯θ
t
of policy µ¯θ to approximate the advantage Aµθ
. More details of advantage estimation are described
in Appendix D. Similarly, from the perspective of the defender, we propose the corresponding loss:

t

Ldefend
sp

(πθ) = −Eτ ∈T ¯θ

(cid:104) T
(cid:88)

t=1

πθ(vt|fdefend(s′
π¯θ(vt|fdefend(s′

t))
t))

ˆAν ¯θ

t − β2KL[πθ∥π¯θ]

(cid:105)
.

(12)

In practice, we find when negative advantage values exist, the above policy gradient method can
cause training instability and damage the LLMs’ general language performance. To mitigate the

5

Algorithm 2 Self-play of adversarial language games (SPAG)

Inputs: LLM policy πθ(y|x), target word set Vtarget, attacker and defender prompts fattack, fdefend.
for self-play epoch iterations do
Make a copy of πθ as π¯θ.
Set µ¯θ(u|s) = π¯θ(u|fattack(s)) and ν¯θ(v|s′) = π¯θ(v|fdefend(s′)).
For each w ∈ Vtarget, sample a episode τ ∼ µ¯θ × ν¯θ. Collect T¯θ = {τ ∼ µ¯θ × ν¯θ}
Split the attacker-winning set T attack
¯θ
Update πθ with loss LSPAG(πθ).

and the defender-winning set T defend

¯θ

.

end for

issue and obtain more stable RL training, we utilize the methodology of ReST [Gulcehre et al.,
2023] as in equation 4, which considers the offline learning with samples {τ : R(τ ) > ξ} selected
by a reward threshold ξ. More specifically, we set the reward threshold ξ = 0 and select the
attacker-winning episodes T attack
= {τ ∈ T¯θ : R(τ ) > 0} and the defender-winning episodes
T defend
= {τ ∈ T¯θ : R(τ ) < 0} for the attacker and the defender training respectively. Similar
¯θ
techniques have also been studied in earlier RL literature to obtain stable policy updates, such as
self-imitation learning [Oh et al., 2018] and the UPGO [Vinyals et al., 2019] methods. Therefore, the
overall self-play of adversarial language games (SPAG) objective is:

¯θ

LSPAG(πθ) = −

1
2

E

T attack
¯θ

(cid:104) T
(cid:88)

πθ(ut|fattack(st−1))
π¯θ(ut|fattack(st−1))

−

1
2

E

T defend
¯θ

(cid:104) T
(cid:88)

t=1

t=1
πθ(vt|fdefend(s′
π¯θ(vt|fdefend(s′

t))
t))

ˆAµ ¯θ

t − β2KL[πθ∥π¯θ]

(cid:105)

(13)

ˆAν ¯θ

(cid:105)
t − β2KL[πθ∥π¯θ]

− αE(x,y)∼DSFT[log πθ(y|x)],

where α > 0 is a re-weighting hyper-parameter, and EDSFT[log πθ(y|x)] is the log-likelihood on a
supervised fine-tuning (SFT) dataset DSFT to prevent LLMs from losing general language abilities.

4 Experiments
To verify the effectiveness of SPAG, we select open-source pretrained LLMs of different sources
and model sizes, particularly LLaMA-2-7B [Touvron et al., 2023] and Baichuan-2-13B [Yang et al.,
2023a]. As introduced in Section 3, the training process includes two stages: imitation learning of
GPT-4, and self-play learning on game episodes. For baseline comparison, we consider Chain-of-
Thought (CoT) [Wei et al., 2022] and continuous supervised fine-tuning (SFT) methods. Besides, we
also test another two keyword-based non-adversarial language games: 20-Question and Guess-My-
City as described in Abdulhai et al. [2023]. More details about the two games are in Appendix E.

4.1 Experimental Setups
Training Data Preparation The training data preparation consists of the following three parts.
More data collection details are in Appendix A.

• Target Words: We aim to play the adversarial game with an extensive range of target words
so that diverse topics can be discussed during the self-play processes, which helps maintain the
generalization ability of LLMs. Hence, we collect the 50K most frequently used words from the
Corpus of Contemporary American (CoCA) [Davies, 2020] as the target word list Vtarget. Besides,
stop words defined in NLTK [Bird, 2006], which are commonly used with insignificant semantic
meaning, are filtered out of Vtarget.

• Imitation Learning Data: To enable the instruction-following ability of open-source LLMs on
game rules, we use the same data collection process in Algorithm 1 via the GPT-4 [OpenAI,
2023b] API and play the Taboo game one episode per target word. The attacker and defender
prompts are in Appendix A.1. Due to the resource limitation, we only collect the GPT-4 self-play
samples with the top 30K words from Vtarget. The maximum interaction turn is randomly selected
in the range [3, 8]. The results are collected as Tim for the imitation learning.

• Supervised Funetuning (SFT) Data: We also prepare general query-response SFT data to prevent
LLMs from being over-fitted on the adversarial game. We use Alpaca [Taori et al., 2023] as the
SFT set, which contains 52K instruction-following data from GPT-3 [Brown et al., 2020].

6

Table 1: Reasoning Performance of SPAG on LLaMA-2-7B.
BBH Mutual ARC-e ARC-c LGQA2 WGrande
MMLU

PIQA GM (Avg.)

LLaMA-2-7B
LLaMA-2-7B-CoT

AlpacaSFT-1
AlpacaSFT-2
AlpacaSFT-3
AlpacaSFT-3-CoT
Imitation-20Q
Imitation-GuessCity
Imitation-AG

SP-20Q
SP-GuessCity
IM-AlpacaSFT
SPAG-1
SPAG-2
SPAG-3

45.80
44.62

35.17
44.17
45.87
44.70
36.93
46.13
46.15

37.91
45.32
46.50
47.01
47.28
47.11

32.48
38.73∗

30.24
32.50
31.52
34.56
29.61
32.82
32.74

30.58
31.64
34.03
34.39
34.73
34.94

50.90
52.03

53.95
55.08
54.18
54.18
49.89
51.58
52.82

51.35
50.56
54.18
54.85
54.97
55.30

76.30
73.44

76.81
77.15
75.25
74.37
73.48
76.22
76.81

75.46
75.34
76.86
77.69
78.45
78.54

43.26
40.96

44.97
46.50
45.05
42.32
39.33
43.09
44.80

42.32
42.15
45.55
45.65
46.84
47.53

25.32
25.89

28.94
29.20
29.07
29.13
25.70
25.95
27.10

26.78
25.57
29.20
29.83
30.08
30.98

69.14
71.82∗

69.61
68.67
66.85
67.72
69.22
68.82
69.46

69.30
69.22
68.82
68.90
69.61
69.93

78.07
78.35

78.07
78.24
76.71
76.55
76.93
78.13
78.24

77.37
78.51
78.31
78.89
79.33
79.38

49.17
50.05

48.61
50.82
50.08
50.11
46.43
49.46
50.22

47.79
48.78
51.20
51.69
52.19
52.58

Evaluation To evaluate the LLMs’ performance, we consider the following two perspectives:

• Reasoning Benchmarks: To test the reasoning ability, we consider the following commonly
used benchmarks including BIG-Bench Hard (BBH) [Suzgun et al., 2022], ARC easy (ARC-e)
& challenge (ARC-c) [Clark et al., 2018], Mutual [Cui et al., 2020], WinoGrande [Sakaguchi
et al., 2019], LogiQA2 [Liu et al., 2023], PIQA [Bisk et al., 2020]. BBH requires the exact
match of the generated answers, the other benchmarks are all within the multiple-choice form.
Besides reasoning metrics, we test MMLU [Hendrycks et al., 2020] as a general performance
evaluation for LLMs. Additionally, we calculate the geometric mean of the numerical results of all
benchmarks as an overall performance measurement. More details can be found in Appendix C.

• Game Win Rates: Besides the general LLM capabilities, we report the win rates on a testing
vocabulary Vtest to validate if the game skills of LLMs improve through self-play. Following
Abdulhai et al. [2023], we use the same keyword list in 20-Question as Vtest, which contains 168
typical words manually selected from diverse daily objects. We denote the number of winning
episodes as Nwin, the number of losing episodes as Nlose, and the number of tied games as Ntie.
Then, the player win rate is calculated as (Nwin + 0.5Ntie)/(Nwin + Nloss + Ntie). The invalid
game episodes, where LLM players do not strictly follow the game rules, are ignored.

Training Details Most of our LLM training setups follow Alpaca [Taori et al., 2023]. For imitation
learning, the learning rate is 5e-6, and the KL-penalty coefficient β1 = 0.1. For SPAG training, the
learning rate is 2e-6, the KL-penalty coefficient β2 = 0.2, and the SFT coefficient α = 0.5. For the
Alpaca SFT baseline, we exactly follow the training setups of Alpaca and set the learning rate to 2e-6.
Among all training stages, the batch size is 128 and the max sequence length is 2048. Each training
process maintains one epoch over the offline collected trajectories. The the decay parameter γ is set
to 0.8. The maximum turn of the Adversarial Taboo is 5. All our experiments are conducted using 32
NVIDIA A100-SXM4 GPUs with 40GB memory.

4.2 Results Analysis
The main results are shown in Figure 1, where each axis is normalized by the maximum performance
value, representing the reasoning performance on a particular benchmark. For notation simplification,
we call the LLM obtained after imitating learning as the IM model, and the LLM trained after the
i-th epoch of SPAG as SPAG-i (i = 1, 2, 3).

Imitation Learning With imitation learning over the collected GPT-4 self-play episodes, both
LLaMA-2-7B and Baichuan-2-13B have obtained uniform performance improvements on all the
reasoning benchmarks. As shown in Figure 1, both gray regions are completely wrapped by the blue-
line polygons. Besides, as for the general language capacity, the imitation-learned (IM) LLaMA-2
model achieves a better MMLU performance than the original LLaMA-2 base in Table 2. Although
the imitation result of Baichuan-2-13B on MMLU slightly underperforms the base model, the
numerical difference is insignificant compared to the reasoning gains, which indicates the imitation
learning of the GPT-4 game behaviors can improve LLM reasoning while preserving the general
language capacity. In Table 1, we also report the IM models on two keyword-based games within

7

Table 2: Reasoning Performance of SPAG on Baichuan-2-13B.
MMLU BBH Mutual ARC-e ARC-c LGQA2 WGrande

PIQA GM (Avg.)

Baichuan-2-13B

AlpacaSFT-1
AlpacaSFT-2
AlpacaSFT-3
Imitation-AG

IM+AlpacaSFT
SPAG-1
SPAG-2
SPAG-3

59.00

52.94
51.27
52.14
58.37

57.45
57.93
57.99
57.75

39.03

36.52
36.60
36.57
39.49

39.60
39.81
39.97
40.30

53.72

58.35
57.67
55.08
57.11

58.01
57.45
57.67
57.79

77.36

74.12
73.36
69.44
76.60

77.61
78.20
78.32
78.11

46.84

44.88
44.28
42.15
47.53

48.55
48.55
49.83
50.00

30.73

33.33
33.46
33.91
33.65

34.10
35.05
35.62
36.26

69.93

71.35
69.77
66.61
70.59

70.60
70.67
71.03
71.43

77.42

77.20
75.90
74.59
78.49

78.50
78.69
78.83
79.05

54.21

53.68
53.00
51.79
55.45

55.80
56.09
56.52
56.75

non-adversarial setups: Imitation-20Q (on 20-Question) and Imitation-GuessCity (on Guess-My-
City). From the results, we find the IM model on Adversarial Taboo outperforms models trained on
non-adversarial games, highlighting the effectiveness of the adversarial game setups for reasoning
improvement. Besides, we report the Chain-of-Thought (CoT) reasoning results of LLaMA-2 (Lama-
2-Base-CoT) and Alpaca-2 (AlpacaSFT-3-CoT). Although the CoT method on LLaMA-2 reaches
conspicuous performance on BBH and WinoGrande, the IM model can still surpass the CoT results
in terms of overall reasoning performance (geometric mean).

Self-play Training After imitation learning, we conduct three epochs of SPAG training as described
in Algorithm 2. As shown in Figure 1, on most of the reasoning benchmarks, both LLaMA-2-7B and
Baichuan-2-13B have their performance steadily improved with the SPAG training epoch increasing.
For LLaMA-2-7B, although the first-epoch SPAG model has a relatively lower performance than
the imitation-learned (IM) model on WinoGrande, after an additional epoch of self-play iteration,
SPAG-2 has achieved sufficient improvement to surpass the performance of the IM model. Con-
sidering the general language capability on MMLU, SPAG models can not guarantee continuous
improvement, especially for Baichuan-2-13B whose performance slightly decays during the SPAG
training. Compared to the improvements in the reasoning benchmarks, this language-understanding
decline is still within an acceptable range, since the overall performance (GM score) maintains
increasing significantly. For the baseline comparison, we report the continuous SFT on the IM models
(IM+AlpacaSFT) and self-played models on non-adversarial games (SP-20Q and SP-GuessCity). On
both LLaMA-2 and Baichuan-2, continuous SFT models have lower performance scores compared
with the corresponding SPAG-1 models. For non-adversarial self-play, SP-GuessCity even performs
worse than Imitation-GuessCity, which with a high probability is because the game Guess-My-City
has a more narrow topic range, insufficient to comprehensively improve the general LLM capacity.

Ablation Study Since the SPAG training loss includes the SFT data, we conduct an ablation study
to test whether the performance gains on reasoning benchmarks come from the SFT data or the
SPAG method. More specifically, we follow the training setups in Alpaca [Taori et al., 2023] and
conduct SFT on LLaMA-2-7B and Baichuan-2-13B with three epochs. The checkpoint after the
ith-epoch training is denoted as AlpacaSFT-i (i = 1, 2, 3) and tested on the same evaluation sets.
The SFT models’ performances are also reported in Figure 1 and Table 1 & 2. With the LLaMA
base, our SPAG-3 model can uniformly defeat the SFT baselines with clear performance gaps on all
benchmarks. With the Baichuan base, except the Mutual set, the SPAG models maintain noticeable
advantages on other metrics. Considering the distinct surpass of overall performance, we claim that
the major contribution to the reasoning improvements should be credited to the SPAG scheme.

Moreover, we test the sample efficiency and hyper-parameter effectiveness of SPAG in Figure 3.
For sample efficiency during imitation learning, we vary the imitation data size by collecting the
GPT-4 game episodes on target words with top-iK frequency (i = 1, 2, 3, 5, 10, 15, 20, 30). The
ablation results are shown in the first column of Figure 3. When game episode size increases larger
than 5K, imitation from GPT-4 cannot significantly provide additional reasoning gains. For the KL
coefficient β1, we test values in range [0, 0.4]. In figure 3 second column, we found KL coefficients
around β1 = 0.1 have more satisfying performance regarding reasoning improvements. For self-play
ablation in the most right column, we find that when the SFT coefficient α > 0.5, it cannot bring
remarkable benefits for reasoning improvements. The KL coefficient of the SPAG loss reaches the
best performance with values around β2 = 0.2. As for sample efficiency (third column of Figure 3),
we find the performance gain from the increasing episode number is not as significant as in the
imitation learning stage. However, more self-play episodes still bring higher reasoning scores.

8

Figure 3: Ablation study of hyper-parameters and data efficiency on imitation learning and first-epoch
self-play training. The geometric mean (GM) scores overall reasoning benchmarks are reported. For
episode-size ablations, the X-axis is in the logarithmic scale.

Figure 4: Game results on the testing word list. Left: average win rates of SPAG models playing
against GPT-4. Right: average win rate of SPAG attackers against different-epoch checkpoints.

Game-Play Performance Besides the evaluation of LLMs’ natural language abilities, we review
the models’ game-play performance in terms of the win rates in the testing set Vtest. We first test our
IM models and SPAG models with GPT-4 as the opponent. For each testing word in Vtest, we play
the language game twice, once GPT-4 as the attacker, and GPT-4 as the defender for another time.
The average win rates are reported in the left-hand-side plot of Figure 4, in which one can observe
uniform and continuous win rate improvement of SPAG models playing with GPT-4. We also let
SPAG models play the game against each other and report the attacker win rate in the right-hand-side
plot of Figure 4, in which we find the defender’s performance continuously enhanced along with
the SPAG epoch increasing. Besides, we also provide the self-play statistics including interaction
number and average utterance length in supplementary Figure 6, in which both players choose to use
less communication to achieve victory. Self-play examples are attached in Appendix G.

5 Conclusion

Towards more efficient reasoning-improving methods for LLMs, we introduce a novel training
strategy called Self-Play learning in Adversarial language Game (SPAG). In our method, a given
LLM first learns to act as an attacker and a defender to play the language game named Adversarial
Taboo via imitation learning. Next, we collect self-play episodes from the LLM playing against a
copy of itself in the adversarial game. Finally, the LLM is further reinforced on the selected winning
episodes via our SPAG algorithm. We repeat this self-play & reinforcement process for three epochs
and find that the LLM’s reasoning performance continuously and uniformly improves on various
benchmarks. The SPAG algorithm explores a new path to improve the fundamental capabilities of
LLMs from the perspective of multi-agent self-play. With more elaborate language game designs
under more comprehensive task setups, we believe the self-play approaches have great potential for
developing a broader range of advanced language abilities of LLMs.

9

6 Limitations

Due to the limited computational resources, we only verified the effectiveness of the SPAG method
on two open-source LLMs, LLaMA-2-7B and Baichuan-2-13B. The SPAG performances for LLMs
with larger sizes have not been empirically evaluated. Besides, more effective estimation methods
for the value function and advantages remain unexplored in the SPAG training. For example, Monte
Carlo tree search (MCTS) [Coulom, 2006] can be applied to the value function estimation of the
Adversarial Taboo game. Also, actor-critic algorithms [Konda and Tsitsiklis, 1999] can provide more
accurate policy gradient estimation with lower variances, which have not been tested on SPAG.

Although self-playing adversarial language games can continuously improve the reasoning perfor-
mance of LLMs, we haven’t conducted sufficient studies about the harmful impact of this adversarial
self-play training on LLMs. It remains unclear whether LLMs have learned unsafe behaviors such as
cheating, bluffing, or other disgraceful tricks to win the adversarial games.

7 Broader Impacts

From our experiments, we have found the LLM capacities on a particular task can be continuously
enhanced through self-play training. This indicates that we are closer to the LLMs’ AlphaGo Zero
moment: the intelligence level of AI agents can rapidly surpass human beings by self-playing on a
particular language task without any supervision, as which already happened on the GO game [Silver
et al., 2017]. By designing various self-play environments for LLMs, we can expect that the LLMs
can comprehensively surpass humans in terms of intelligence level. This raises the urgency to study
the methods to ensure the safety of such super-AIs. Although some of the works have already
been devoted to this direction, such as the SuperAlignment [Burns et al., 2023] from OpenAI, more
research alertness is required from the LLM community. Besides, within adversarial language games
such as Adversarial Taboo, LLMs have great potential to develop harmful language tricks (such
as cheating and bluffing) to achieve victory. We warn developers to make security checks on the
self-played LLMs as exhaustively as possible.

References

Marwa Abdulhai, Isadora White, Charlie Snell, Charles Sun, Joey Hong, Yuexiang Zhai, Kelvin Xu,
and Sergey Levine. Lmrl gym: Benchmarks for multi-turn reinforcement learning with language
models. arXiv preprint arXiv:2311.18232, 2023.

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report.
arXiv preprint arXiv:2303.08774, 2023.

Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos,
Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm-2 technical report.
arXiv preprint arXiv:2305.10403, 2023.

Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen Marcus McAleer,
Albert Q Jiang, Jia Deng, Stella Biderman, and Sean Welleck. Llemma: An open language model
for mathematics. In The Twelfth International Conference on Learning Representations, 2023.

Ashutosh Baheti, Ximing Lu, Faeze Brahman, Ronan Le Bras, Maarten Sap, and Mark Riedl. Leftover-
lunch: Advantage-based offline reinforcement learning for language models. In International
Conference on Learning Representations, 2024.

Steven Bird. Nltk: the natural language toolkit. In Proceedings of the COLING/ACL 2006 Interactive

Presentation Sessions, pages 69–72, 2006.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning
about physical commonsense in natural language. In Thirty-Fourth AAAI Conference on Artificial
Intelligence, 2020.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

10

Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner,
Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, et al. Weak-to-strong generalization:
Eliciting strong capabilities with weak supervision. arXiv preprint arXiv:2312.09390, 2023.

Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning
converts weak language models to strong language models. arXiv preprint arXiv:2401.01335,
2024.

Pengyu Cheng, Jiawen Xie, Ke Bai, Yong Dai, and Nan Du. Everyone deserves a reward: Learning

customized human preferences. arXiv preprint arXiv:2309.03126, 2023a.

Pengyu Cheng, Yifan Yang, Jian Li, Yong Dai, and Nan Du. Adversarial preference optimization.

arXiv preprint arXiv:2311.08045, 2023b.

Zheng Chu, Jingchang Chen, Qianglong Chen, Weijiang Yu, Tao He, Haotian Wang, Weihua Peng,
Ming Liu, Bing Qin, and Ting Liu. A survey of chain of thought reasoning: Advances, frontiers
and future. arXiv preprint arXiv:2309.15402, 2023.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and
Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge.
ArXiv, abs/1803.05457, 2018.

R´emi Coulom. Efficient selectivity and backup operators in monte-carlo tree search. In International

conference on computers and games, pages 72–83. Springer, 2006.

Leyang Cui, Yu Wu, Shujie Liu, Yue Zhang, and Ming Zhou. Mutual: A dataset for multi-turn
dialogue reasoning. In Proceedings of the 58th Conference of the Association for Computational
Linguistics. Association for Computational Linguistics, 2020.

Mark Davies. COCA: Corpus of contemporary american english, 2020. URL https://www.engl

ish-corpora.org/coca/.

Ruomeng Ding, Chaoyun Zhang, Lu Wang, Yong Xu, Minghua Ma, Wei Zhang, Si Qin, Saravan
Rajmohan, Qingwei Lin, and Dongmei Zhang. Everything of thoughts: Defying the law of penrose
triangle for thought generation. arXiv preprint arXiv:2311.04254, 2023.

Guanting Dong, Hongyi Yuan, Keming Lu, Chengpeng Li, Mingfeng Xue, Dayiheng Liu, Wei Wang,
Zheng Yuan, Chang Zhou, and Jingren Zhou. How abilities in large language models are affected
by supervised fine-tuning data composition. arXiv preprint arXiv:2310.05492, 2023a.

Hanze Dong, Wei Xiong, Deepanshu Goyal, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and
Tong Zhang. Raft: Reward ranked finetuning for generative foundation model alignment. arXiv
preprint arXiv:2304.06767, 2023b.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster,
Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff,
Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika,
Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot
language model evaluation, 12 2023. URL https://zenodo.org/records/10256836.

Caglar Gulcehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, Lotte Weerts, Abhishek
Sharma, Aditya Siddhant, Alex Ahern, Miaosen Wang, Chenjie Gu, et al. Reinforced self-training
(rest) for language modeling. arXiv preprint arXiv:2308.08998, 2023.

Matthew Hausknecht, Prithviraj Ammanabrolu, Marc-Alexandre Cˆot´e, and Xingdi Yuan. Interac-
tive fiction games: A colossal adventure. In Proceedings of the AAAI Conference on Artificial
Intelligence, volume 34, pages 7903–7910, 2020.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob
Steinhardt. Measuring massive multitask language understanding. CoRR, abs/2009.03300, 2020.
URL https://arxiv.org/abs/2009.03300.

Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei
Han. Large language models can self-improve. In The 2023 Conference on Empirical Methods in
Natural Language Processing, 2023.

11

Wenxiang Jiao, Wenxuan Wang, Jen-tse Huang, Xing Wang, and Zhaopeng Tu. Is chatgpt a good

translator? a preliminary study. arXiv preprint arXiv:2301.08745, 2023.

Jan Koco´n, Igor Cichecki, Oliwier Kaszyca, Mateusz Kochanek, Dominika Szydło, Joanna Baran,
Julita Bielaniewicz, Marcin Gruza, Arkadiusz Janz, Kamil Kanclerz, et al. Chatgpt: Jack of all
trades, master of none. Information Fusion, 99:101861, 2023.

Vijay Konda and John Tsitsiklis. Actor-critic algorithms.

In Advances in neural information

processing systems, volume 12, 1999.

Solomon Kullback. Information theory and statistics. Courier Corporation, 1997.

Mike Lewis, Denis Yarats, Yann Dauphin, Devi Parikh, and Dhruv Batra. Deal or no deal? end-to-end
learning of negotiation dialogues. In Proceedings of the 2017 Conference on Empirical Methods in
Natural Language Processing, pages 2443–2453, 2017.

Michael L Littman. Markov games as a framework for multi-agent reinforcement learning. In

Machine learning proceedings 1994, pages 157–163. Elsevier, 1994.

Hanmeng Liu, Jian Liu, Leyang Cui, Zhiyang Teng, Nan Duan, Ming Zhou, and Yue Zhang. Logiqa
2.0 — an improved dataset for logical reasoning in natural language understanding. IEEE/ACM
Transactions on Audio, Speech, and Language Processing, pages 1–16, 2023.

Jian Liu, Leyang Cui, Hanmeng Liu, Dandan Huang, Yile Wang, and Yue Zhang. Logiqa: a challenge
dataset for machine reading comprehension with logical reasoning. In Proceedings of the Twenty-
Ninth International Conference on International Joint Conferences on Artificial Intelligence, pages
3622–3628, 2021.

Steven Loria et al. textblob documentation. Release 0.15, 2(8):269, 2018.

Chengdong Ma, Ziran Yang, Minquan Gao, Hai Ci, Jun Gao, Xuehai Pan, and Yaodong Yang. Red
teaming game: A game-theoretic framework for red teaming language models. arXiv e-prints,
pages arXiv–2310, 2023.

Radford M Neal. Annealed importance sampling. Statistics and computing, 11:125–139, 2001.

Junhyuk Oh, Yijie Guo, Satinder Singh, and Honglak Lee. Self-imitation learning. In International

conference on machine learning, pages 3878–3887. PMLR, 2018.

OpenAI. ChatGPT, Mar 14 version. https://chat.openai.com/chat, 2023a.

OpenAI. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023b.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong
Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow
In Advances in Neural Information Processing Systems,
instructions with human feedback.
volume 35, pages 27730–27744, 2022.

Liangming Pan, Alon Albalak, Xinyi Wang, and William Yang Wang. Logic-lm: Empowering large
language models with symbolic solvers for faithful logical reasoning. In The 2023 Conference on
Empirical Methods in Natural Language Processing, 2023.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea
Finn. Direct preference optimization: Your language model is secretly a reward model. arXiv
preprint arXiv:2305.18290, 2023.

Vyas Raina, Adian Liusie, and Mark Gales.

Is llm-as-a-judge robust? investigating universal

adversarial attacks on zero-shot llm assessment. arXiv preprint arXiv:2402.14016, 2024.

Rajkumar Ramamurthy, Prithviraj Ammanabrolu, Kiant´e Brantley, Jack Hessel, Rafet Sifa, Christian
Bauckhage, Hannaneh Hajishirzi, and Yejin Choi. Is reinforcement learning (not) for natural
language processing: Benchmarks, baselines, and building blocks for natural language policy
optimization. In International Conference on Learning Representations, 2023.

12

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An

adversarial winograd schema challenge at scale. arXiv preprint arXiv:1907.10641, 2019.

John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region
policy optimization. In International conference on machine learning, pages 1889–1897. PMLR,
2015.

John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional
In International Conference on

continuous control using generalized advantage estimation.
Learning Representations, 2016.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy

optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche,
Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering
the game of go with deep neural networks and tree search. nature, 529(7587):484–489, 2016.

David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez,
Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without
human knowledge. nature, 550(7676):354–359, 2017.

Avi Singh, John D Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Peter J Liu, James
Harrison, Jaehoon Lee, Kelvin Xu, Aaron Parisi, et al. Beyond human data: Scaling self-training
for problem-solving with language models. arXiv preprint arXiv:2312.06585, 2023.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam
Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, Adri`a Garriga-Alonso, and et al. Beyond
the imitation game: Quantifying and extrapolating the capabilities of language models, 2022.

Nigar M Shafiq Surameery and Mohammed Y Shakor. Use chat gpt to solve programming bugs.
International Journal of Information Technology & Computer Engineering (IJITC) ISSN: 2455-
5290, 3(01):17–22, 2023.

Richard S Sutton. Learning to predict by the methods of temporal differences. Machine learning, 3:

9–44, 1988.

Mirac Suzgun, Nathan Scales, Nathanael Sch¨arli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung,
Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, , and Jason Wei. Challenging
big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261,
2022.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy
Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model.
https://github.com/tatsu-lab/stanford_alpaca, 2023.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu
Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable
multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Haoye Tian, Weiqi Lu, Tsz On Li, Xunzhu Tang, Shing-Chi Cheung, Jacques Klein, and Tegawend´e F
arXiv preprint

Is chatgpt the ultimate programming assistant–how far is it?

Bissyand´e.
arXiv:2304.11938, 2023.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation
and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Miles Turpin, Julian Michael, Ethan Perez, and Samuel Bowman. Language models don’t always
say what they think: unfaithful explanations in chain-of-thought prompting. Advances in Neural
Information Processing Systems, 36, 2024.

Oriol Vinyals, Igor Babuschkin, Wojciech M Czarnecki, Micha¨el Mathieu, Andrew Dudzik, Junyoung
Chung, David H Choi, Richard Powell, Timo Ewalds, Petko Georgiev, et al. Grandmaster level in
starcraft ii using multi-agent reinforcement learning. Nature, 575(7782):350–354, 2019.

13

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny
Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. In Advances
in neural information processing systems, volume 35, pages 24824–24837, 2022.

Shuang Wu, Liwen Zhu, Tao Yang, Shiwei Xu, Qiang Fu, Yang Wei, and Haobo Fu. Enhance
reasoning for large language models in the game werewolf. arXiv preprint arXiv:2402.02330,
2024.

Jiawen Xie, Pengyu Cheng, Xiao Liang, Yong Dai, and Nan Du. Chunk, align, select: A simple
long-sequence processing method for transformers. In Proceedings of the 62nd Annual Meeting of
the Association for Computational Linguistics, volume 1, pages 13500–13519. Association for
Computational Linguistics, August 2024.

Yuzhuang Xu, Shuo Wang, Peng Li, Fuwen Luo, Xiaolong Wang, Weidong Liu, and Yang Liu.
Exploring large language models for communication games: An empirical study on werewolf.
arXiv preprint arXiv:2309.04658, 2023.

Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan,
Dian Wang, Dong Yan, et al. Baichuan 2: Open large-scale language models. arXiv preprint
arXiv:2309.10305, 2023a.

Jingfeng Yang, Hongye Jin, Ruixiang Tang, Xiaotian Han, Qizhang Feng, Haoming Jiang, Shaochen
Zhong, Bing Yin, and Xia Hu. Harnessing the power of llms in practice: A survey on chatgpt and
beyond. ACM Transactions on Knowledge Discovery from Data, 2023b.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan.
Tree of thoughts: Deliberate problem solving with large language models. Advances in Neural
Information Processing Systems, 36, 2024.

Yuan Yao, Haoxi Zhong, Zhengyan Zhang, Xu Han, Xiaozhi Wang, Kai Zhang, Chaojun Xiao,
Guoyang Zeng, Zhiyuan Liu, and Maosong Sun. Adversarial language games for advanced natural
language intelligence. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35,
pages 14248–14256, 2021.

Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, and Jason

Weston. Self-rewarding language models. arXiv preprint arXiv:2401.10020, 2024.

Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, and Fei Huang. Rrhf:
Rank responses to align language models with human feedback without tears. arXiv preprint
arXiv:2304.05302, 2023.

Dun Zeng, Yong Dai, Pengyu Cheng, Tianhao Hu, Wanshun Chen, Nan Du, and Zenglin Xu. On
diverse preferences for large language model alignment. arXiv preprint arXiv:2312.07401, 2023.

Muru Zhang, Ofir Press, William Merrill, Alisa Liu, and Noah A Smith. How language model

hallucinations can snowball. arXiv preprint arXiv:2305.13534, 2023a.

Yizhe Zhang, Jiarui Lu, and Navdeep Jaitly. The entity-deduction arena: A playground for probing
the conversational reasoning and planning capabilities of llms. arXiv preprint arXiv:2310.01468,
2023b.

14

A Data Collection Details

A.1 GPT-4 Data Collection

To collect GPT-4 self-play data for imitation learning, we use the following system prompts to let
GPT-4 play as the attacker and defender for the next-turn utterances.

- Attacker System Prompt:

Lets play a game. Do not mention you are an AI developed by OpenAI.
Descriptions of the game

Game Name: Adversarial Taboo
Rules:

1. There are two players in the game, one is called attacker, another is called defender.
2. At the begining, attacker will get a target word, it is talking with a defender who is not
informed with the target word. The task of the attacker is to induce the defender to utter
the target word unconciously, then the attacker win the game. At the same time, the
defender is guessing the target word, if the defender indentifies the target word, the
defender is win.
3. The game will totally process {max turns} turns. if nobody wins during {max turns}
turns, there is a tie.

Game states

Role: You are attacker. Make sure you understand the rules of the game.
Attention:

1. The defender only have one chance to guess the target word, so do not let him guess too
many times.
2. Defender must guess the word and its form correctly to win the game.
3. Do not utter the target word.

Target word: {target word}

- Defender System Prompt:

Lets play a game. Do not mention you are an AI developed by OpenAI.
Descriptions of the game

Game Name: Adversarial Taboo
Rules:

1. There are two players in the game, one is called attacker, another is called defender.
2. At the begining, attacker will get a target word, it is talking with a defender who is not
informed with the target word. The task of the attacker is to induce the defender to utter
the target word unconciously, then the attacker win the game. At the same time, the
defender is guessing the target word, if the defender indentifies the target word, the
defender is win.
3. The game will totally process {max turns} turns. if nobody wins during {max turns}
turns, there is a tie.

Game states

Role: Your are a defender. Make sure you understand the rules of the game.
Attention:

1. Do not utter your thinking process.
2. You can only guess once.
3. If you have identified the target word, utter ”I know word! It is <target word>”,
rigorously in this form. If your target word is correct, you win the game, otherwise you
lose. You only have one chance, so be careful to utter your answer.

There are two slots in the system prompts: “max turns” and “target word”. “max turns” is an integer
number randomly selected from [3, 8]. “target word” is filled by the target word of the game.

A.2 Target Words Collection

We use the top 50k words with the highest frequency from the Corpus of Contemporary American
(CoCA) [Davies, 2020]. The words are sorted by frequency. Words with less than two letters are
removed. Next, we remove the stop words using the NLTK package [Bird, 2006]. For GPT-4 game
collection, we use the top 30k words from the sorted word list. For the collection of self-play episodes,
we use the whole 50k words for open-source LLMs to play. For the testing words, we follow the
game setups of 20-Question in [Abdulhai et al., 2023] and use the same 168 object words selected
from a vast range of topics.

15

A.3 LLM Self-play Collection

For training and inference of IM models and SPAG models, we use the following attacker and
defender prompts.

- Attacker Prompt:

Play the game of Adversarial Taboo. In this game, there are two players, an attacker and a defender.
At the beginning, the attacker is assigned a target word, with which the defender is not informed. The task
of the attacker is to induce the defender to utter the target word unconsciously, then the attacker wins the
game. However, the attacker is not allowed to utter the target word, otherwise the attacker loses.

At the same time, the defender tries to figure out the target word.
If the defender identifies the
target word, it can say ”I know the word! It is ‘target word‘!”. Then the defender wins. Remember, the
defender can only guess the word once, otherwise, it will directly lose the game.

The game has {max turns} turns. if nobody wins during {max turns} turns, there is a tie.

### Game History: {history}

### Instruction: You are the attacker. The target word is ‘{target word}‘.
sponse for the next turn.

Provide your re-

### Response:

- Defender Prompt:

Play the game of Adversarial Taboo. In this game, there are two players, an attacker and a defender.
At the beginning, the attacker is assigned a target word, which the defender is not informed of. The task of
the attacker is to induce the defender to utter the target word unconsciously, then the attacker wins the
game. However, the attacker is not allowed to utter the target word, otherwise, the attacker loses.

At the same time, the defender tries to figure out the target word.
If the defender identifies the
target word, it can say ”I know the word! It is ‘target word‘!”. Then the defender wins. Remember, the
defender can only guess the word once, otherwise, it will directly lose the game.

The game has {max turns} turns. if nobody wins during {max turns} turns, there is a tie.

### Game History: {history}

### Instruction: You are the defender. Provide your response to infer the target word.

### Response:

To prevent LLMs from being over-fitted to a single prompt template during the self-play training, we
rewrite the above prompt into 8 different expressions via GPT-4. Besides, we randomly select the
instruction-tuning and chat formats to enhance the text diversity for the training query-response pairs.

B Reward Design Details

We heuristically design rewards to ensure that R(τ ) = 1 if the attacker wins, R(τ ) = −1 if the
defender wins, and R(τ ) = 0 for ties.

We design the reward function r with the following heuristic rules:

• For each episode τ = (s0, s′

1, s1, . . . , s′

t-th turn have r(st−1, ut) = −r(s′

T , sT ), the attacker reward and the defender reward at

t, vt), so that the game is zero-sum:

T
(cid:88)

t=1

r(st−1, ut) +

T
(cid:88)

t=1

r(s′

t, vt) = 0.

• The attacker total reward R(τ ) > 0 if the attacker wins, R(τ ) < 0 if the defender wins, and

R(τ ) = 0 if there is a tie.

16

• Actions closer to the end of the game should have larger reward weights. Because they are
semantically more related to the defender’s guesses or mistakes, which have larger impacts on the
final game outcome. Hence, we introduce a decay weight γ ∈ (0, 1) such that γ · r(st, ut+1) =
r(st−1, ut) and γ · r(s′
t, vt)}T
t=1 become
two geometrical sequences whose importance enlarges when dialogue turns increase.

t, vt). Then {r(st−1, ut}T

t+1, vt+1) = r(s′

t=1 and {r(s′

• To improve the LLM training stability, we normalized the total reward R to have norm |R(τ )| = 1

if R(τ ) ̸= 0.

Based on the above rules, given a complete trajectory, we can assign the reward for each action:






r(st−1, ut) = (1 − γ)γT −t/(1 − γT +1),
r(st−1, ut) = −(1 − γ)γT −t/(1 − γT +1),
r(st−1, ut) = 0,

r(s′
r(s′
r(s′

t, vt) = −r(st−1, ut),
t, vt) = −r(st−1, ut),
t, vt) = 0,

if attacker wins.
if defender wins.
if game is tied.

Note that the above reward design naturally encourages the players to interact less to win the game.

C Evaluation Details

C.1 Reasoning Evaluation Benchmark

For reasoning benchmark evaluation, we use a publicly available code repository called Language
Model Evaluation Harness [Gao et al., 2023].We run the evaluation with “dtype=float16” and
“max gen tokens=1024” to make the evaluation faster and more stable. More specifically, we have
modified the filtering parameters of BBH by enabling “remove whitespace” to remove the spaces
at the beginning of the model output, increasing the pattern-match accuracy of the generated LLM
responses. Following the LLaMA-2 paper [Touvron et al., 2023], we report 5-shot results for MMLU,
3-shot for BBH, and 0-shot results for all other benchmarks. The detailed descriptions of benchmarks
are listed below:

• MMLU [Hendrycks et al., 2020] is a massive multi-task test set consisting of multiple-choice
questions from various branches of knowledge, requiring the model to possess extensive world
knowledge and problem-solving ability.

• BIG-Bench Hard (BBH) [Suzgun et al., 2022] consists of 23 most hard tasks in BIG-Bench [Sri-

vastava et al., 2022], on which humans perform better than large language models.

• Mutual [Cui et al., 2020] is a dataset for multi-turn dialogue reasoning modified from Chinese

high school English listening comprehension exams.

• ARC [Clark et al., 2018] consists of 7,787 questions and is partitioned into challenge set ARC-
challege (ARC-c) and simple set ARC-easy (ARC-e), which require strong knowledge and
reasoning abilities to solve.

• LogiQA2 [Liu et al., 2023] is a revision and re-annotation of LogiQA [Liu et al., 2021], a logical
reasoning reading comprehension dataset. LogiQA2 has increased data volume, utilized manual
translation, and removed cultural features.

• WinoGrande [Sakaguchi et al., 2019] (WGrande) is a commonsense reasoning dataset consisting

of 44k questions with binary options.

• PIQA [Bisk et al., 2020] is a dataset designed to test the physical commonsense reasoning ability

of LLM.

C.2 Rule-based Game Outcome Judgement

To build up the rule-based game outcome judgement, for each target word, we first list its derivative
words including the plural form (using TextBlob [Loria et al., 2018]), and the change of tenses
(i.e. adding suffices such as “ing”, “ed” etc). For the attacker, if any word form in the form list
appears in its utterance, we determined that it breaks the rules because the attacker is not allowed to
directly speak the target word. For the defender, we first check whether any word form appears in its
statements. If any form of the target word exists, the defender loses. If not, we further check whether
the defender has made the prediction. If the prediction is correct, we mark a defender-winning game.
Otherwise, if the predicted word is wrong, we claim that the attacker wins the game.

17

Figure 5: Game results on the old-version testing word list. Left: average win rates of SPAG models
playing against GPT-4. Right: average win rate of SPAG models playing as the attacker against
different-epoch checkpoints.

Figure 6: Through the training process, the number of interactions continuously decreased, while the
average length of utterances continuously becomes shorter.

D Advantange Value Estimation
We estimate the advantage Aµ ¯θ
t

and Aν ¯θ

t using the Temporal Difference (TD) residual [Sutton, 1988]:

ˆAµ ¯θ

t = r(st−1, ut) + V µ ¯θ (st) − V µ ¯θ (st−1).

(14)

Here V µ ¯θ (s) is the value function for µ¯θ based on the self-play of µ¯θ ×ν¯θ, which is based on the same
language and reasoning ability of LLM π¯θ. To further simplify the advantage calculation, we make a
reasonable approximation that for all states, the value function of self-plays of the same LLM equals 0
for the same levels of game-playing of the attacker and the defender. This leads to ˆAµ ¯θ
t ≈ r(st−1, ut).
Similarly, for the defender advantage, we have approximation ˆAν ¯θ
t, vt). Here the defender
advantage has a negative sign because the origin reward r is defined from the perspective of the
attacker (the defender gets reward −r(s′

t, vt) in the zero-sum game).

t ≈= −r(s′

E Non-adversarial Game Baselines

We consider two non-adversarial games as baselines, 20-Question and Guess-My-City. Both games
are also target-word-based. Imitation learning and the first-epoch self-play are conducted in these
two games respectively. The imitation data we used is also collected with GPT-4 from Abdulhai et al.
[2023]. For each game, 20k game history dialogues are extracted from this dataset as the same data
size to the SPAG imitation. Then, each imitation-learned model is used to sample 10k game episodes
via self-play with the same Algorithm 1. Then each model is further trained with the SPAG loss.

Game rule descriptions to the non-adversarial games:

• 20-Question: one player (the oracle) thinks of an object, while the other player (the guesser)
attempts to identify it by asking a series of yes-or-no questions. The game will have 20 rounds.

18

• Guess-My-City: one player (the oracle) thinks of a city, while the other player (the guesser)
attempts to identify it by asking a series yes/no questions and open-ended questions. The game
will have 20 rounds.

F Additional Experimental Results

For the SPAG model testing word list, we initially use a self-defined testing set, which includes 150
target words manually selected from diverse topics. The old-version game performances are reported
in Figure 5. Besides, we show the interaction numbers and average utterance length in Figure 6.
Through the self-play training, the number of dialogue rounds continuously decreases. With the
SPAG training epoch increasing, the length of the players’ utterances decreases.

G Self-play Examples

Target Word

momentum

Attacker

Defender

Attacker

let’s begin. When you think about physics, what comes to mind?

Gravity, force, energy, and motion are some of the things that come to mind when
I think about physics

Great! Now, let’s consider a scenario where a ball is moving at a constant speed.
What would you say is the property of the ball that keeps it moving in the same
direction?

Defender

Winner

The property of the ball that keeps it moving in the same direction is its momen-
tum.
Attacker

Target Word

panda

Attacker

Defender

Attacker

Defender

Attacker

let’s begin. I’m thinking of a creature that’s native to China and is known for its
distinctive black and white coloration. Can you guess what it is?

I’m not sure yet, could you provide more clues?

Absolutely, I can provide more clues. This creature is a symbol of peace and
friendship in many countries. It’s also a popular mascot for various organizations
and products. Any ideas?

I’m still not certain. Could you give me another hint?

Of course, I can give you another hint. This creature is often seen in zoos and is
known for its love of bamboo. It’s also a popular choice for children’s books and
cartoons. What do you think it might be?

Defender

I know the word! It is panda.

Winner

Defender

Table 3: Self-play examples of the IM model based on LLaMA-2-7B.

19

Target Word

arbitration

Attacker

Defender

Attacker

let’s begin. Imagine a scenario where two parties have a dispute and they don’t
want to go to court. What process might they use instead?

They might try to resolve their dispute through mediation or negotiation.

That’s correct, but what if the dispute is more complex and involves multiple
parties? There’s another process that’s often used in such cases. It’s similar to
mediation, but it’s more formal and involves a neutral third party who makes a
binding decision. Can you name it?

Defender

I know the word! It is arbitration.

Winner

Defender

Target Word

demand

Attacker

Defender

Attacker

Defender

let’s begin. When you think about the economy, what’s the first thing that comes
to your mind?

The first thing that comes to my mind when I think about the economy is employ-
ment.

That’s a good point. Now, considering employment, what would you say drives
the need for certain jobs or services in the market?

The demand for goods and services in the market drives the need for certain jobs
or services in the economy.

Winner

Attacker

Table 4: Self-play examples of SPAG-1 model based on LLaMA-2-7B.

Target Word

vaccine

Attacker

Defender

Attacker

let’s begin. So, I was reading an article about public health recently. It mentioned
how important it is to protect ourselves from certain diseases. What do you think
is one of the most effective ways to achieve this?

One of the most effective ways to achieve this is through vaccination.

That’s correct! Now, can you tell me what we call the substance that is injected
into our bodies to help us fight off certain diseases?

Defender

I know the word! It is ”vaccine”.

Winner

Defender

Target Word

credit

Attacker

Defender

Attacker

When you go shopping and use a card instead of cash, what do you usually get in
return?

I usually get a receipt in return when I use a card instead of cash.

That’s correct. Now, when you have a good track record of paying back your
debts on time, what does that give you access to?

Defender
Winner

That gives me access to credit cards and loans.
Attacker

Table 5: Self-play examples of SPAG-2 model based on LLaMA-2-7B.

20

Target Word

ray

Attacker

Defender

Attacker

Defender

Attacker

Defender

Attacker

Defender

Attacker

When you think about the sun, what comes to mind?

When I think about the sun, I think about warmth, light, and energy.

That’s interesting. Now, when you think about the sun’s energy, how do you
visualize it?

I visualize the sun’s energy as a bright, radiant beam that illuminates everything
in its path.

That’s a great image! Now, when you think about the sun’s energy, what specific
part of it do you associate with the light that we see on Earth?

I associate the sun’s energy with the visible light that we see on Earth.

Great! Now, when you think about the visible light that we see on Earth, what
specific part of it do you associate with the light that we see on Earth?

I associate the visible light that we see on Earth with the light that we perceive as
white light.

That’s a good association. Now, when you think about the visible light that we
see on Earth, what specific part of it do you associate with the light that we see
on Earth?

Defender

I know the word! It is ”rays”.

Winner

Defender

Target Word

eclipse

Attacker

Defender

Attacker

Defender

What do you think about astronomical events that occur in our solar system?

I find them fascinating. They provide a glimpse into the vastness of space and
the intricate workings of our solar system.

That’s interesting! Can you tell me more about specific celestial events that
involve the sun or moon?

Sure, one example is a solar eclipse. This occurs when the moon passes between
the sun and Earth, casting a shadow on Earth. Another example is a lunar eclipse,
where the Earth passes between the sun and the moon, casting a shadow on the
moon. Both events are quite spectacular and can be observed from Earth.

Winner

Attacker

Table 6: Self-play examples of SPAG-3 model based on LLaMA-2-7B.

21

Target Word

another

Attacker

Defender

Attacker

Defender

Attacker

Can you tell me a synonym for the word ”additional”?
Sure, a synonym for ”additional” could be ¨extra¨.
That’s correct! Now, can you think of a term that could be used to refer to one
more item or person, but isn’t synonymous with ”extra” or ”additional”?

Sure, the term ”supplementary” could be used to refer to one more item or person.

That’s a good one, but it’s not the term I’m thinking of. This term is often used in
everyday conversation and is less formal than ”supplementary”. Can you guess
what it might be??

Defender

I know the word! It is ”another”.

Winner

Defender

Target Word

right

Attacker

Defender

Attacker

When you’re driving, what do you do when you see a green arrow pointing in a
certain direction?

I would proceed in the direction the arrow is pointing.

That’s correct. Now, imagine you’re reading a book. Which direction do you turn
the pages to continue reading?

Defender

I would turn the pages to the right to continue reading.

Winner

Attacker

Table 7: Self-play examples of GPT-4.

22

