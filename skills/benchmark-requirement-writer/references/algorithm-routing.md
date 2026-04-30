# Algorithm Routing

Use this reference when the hard part is deciding whether the chosen execution baseline should lead to `SFT`, `DPO`, or `GRPO`.

## Quick routing

- Choose `SFT` first when the baseline mainly fails on domain knowledge, terminology, legal citation habits, answer format, or task framing.
- Choose `DPO` first when you have or can build preference pairs such as better-vs-worse legal answers, safer-vs-riskier responses, or style / helpfulness tradeoffs that are hard to encode as a scalar reward.
- Choose `GRPO` first only when the baseline is already reasonably strong and you can define a stable, auditable reward or verifier.

## Strong signals for SFT

Prefer `SFT` when several of these are true:

- the baseline is small or mid-sized and clearly underfits the domain
- the model often misunderstands legal task format
- answers are incoherent because knowledge is missing rather than because search is weak
- the benchmark can be improved with better domain demonstrations
- you need the cheapest and lowest-risk first training path

## Strong signals for DPO

Prefer `DPO` when several of these are true:

- you have pairwise or ranked supervision instead of gold single answers
- legal answer quality depends on nuanced style, caution, coverage, or preference tradeoffs
- answer correctness is not fully machine-checkable but relative preference is easier to label
- you want a middle ground between plain SFT and verifier-heavy RL

## Strong signals for GRPO

Prefer `GRPO` when several of these are true:

- the baseline is already fairly capable
- tasks have verifiable outcomes, rule-based graders, or reward heuristics
- the expected gain is not mainly memorizing legal knowledge but improving search, reasoning, or tool use
- you can describe reward leakage risks and how to control them
- you can afford a more complex training and evaluation loop

## Benchmark-specific caution

Do not choose `GRPO` just because a paper used RL.

Check first:

- does the benchmark expose a stable reward or verifier?
- does the verifier align with the benchmark metric?
- can the reward be audited for leakage or overfitting?
- is the execution baseline already strong enough that RL has a realistic chance to help?

If the answer to any of these is unclear, default back toward `SFT` or `DPO`.
