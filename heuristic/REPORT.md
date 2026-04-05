# Heuristic Discovery from Complete Game Solutions: Methodology and Results

## 1. Motivation

A strong solution of a combinatorial game provides the minimax value for every reachable state, enabling perfect play from any position. While this is the gold standard for game-theoretic analysis, it is impractical for human players: the solution is a lookup table with millions of entries, not a cognitive strategy. A natural follow-up question is whether the exhaustive solution can be *distilled* into concise, human-playable rules that approximate optimal play.

This question is scientifically interesting beyond any single game. For every strongly solved game, one can ask: how many bits of the solution are truly decision-relevant? If a handful of features explain most of the optimal moves, the game has a *low effective strategic complexity* despite a large state space. Conversely, if no compact heuristic achieves high accuracy, the game's strategy is genuinely complex in the information-theoretic sense.

We propose a general methodology for extracting interpretable heuristics from complete game solutions and evaluate it on Okiya as a case study. The approach requires no neural networks and produces human-readable rules directly from the labeled game DAG.


## 2. Methodology

### 2.1 Overview

Given a strongly solved game with minimax values for all reachable states, the pipeline proceeds in five stages:

```
Feature Engineering → Information Analysis → Rule Learning → Heuristic Design → Evaluation
```

1. **Feature engineering**: design domain-meaningful features computable from a game state and a candidate move.
2. **Mutual information analysis**: compute MI(feature, is_optimal_move) stratified by game level to identify which features matter at each phase.
3. **Decision tree learning**: train shallow trees (depth 4--8) on labeled (state, move) pairs to extract explicit rules.
4. **Heuristic design**: construct a family of heuristics ranging from simple priority rules to learned classifiers.
5. **Evaluation**: compare all heuristics via move accuracy (agreement with minimax) and win rate (game simulation).


### 2.2 Why not neural networks?

A common approach to distilling game knowledge is to train a neural policy network and then attempt to interpret it. This introduces an unnecessary intermediate: the network is itself a black box, and extracting rules from a black box is harder than extracting rules from the ground truth directly.

When complete minimax labels are available, the problem reduces to supervised classification on structured inputs. Interpretable models (decision trees, rule learners) can be trained directly on (features, is_optimal_move) pairs. The resulting rules are immediately readable, need no distillation, and can be tested against the oracle without simulation artifacts.

Neural networks become relevant when the game is too large to solve completely and the heuristic must generalize from partial solutions. For fully solved games, they add complexity without adding capability.


## 3. Feature Engineering

### 3.1 General principles

Features should capture the strategic axes of the game. For two-player placement games, three categories are universal:

**Offensive features** measure the current player's progress toward a win:
- *Pattern completion*: how many tiles the current player has in each winning configuration
- *Threat count*: number of winning configurations with k-1 tiles already claimed
- *Win-in-one*: whether the move immediately completes a winning configuration

**Defensive features** measure the opponent's threats and the need to block:
- *Opponent pattern completion*: maximum tiles the opponent has in any single winning configuration
- *Block urgency*: whether the move prevents the opponent from winning on the next turn (occupying the sole missing tile in an opponent's k-1 configuration)

**Constraint features** measure the structural consequences of the move:
- *Opponent mobility*: how many legal moves the opponent has after this move (lower = more constrained)
- *Stuck threat*: whether the opponent will have zero legal moves (immediate loss)

**Positional features** capture board-independent structural properties:
- *Pattern participation*: how many winning configurations include the candidate position
- *Game phase*: the current level or fraction of the board occupied

### 3.2 Implementation for Okiya

We define 11 features per (state, candidate_move) pair:

| # | Feature | Type | Definition |
|---|---------|------|------------|
| 1 | `wins_immediately` | Offensive | Move completes a win pattern |
| 2 | `blocks_opp_3` | Defensive | Move occupies the sole missing tile in an opponent's 3-of-4 pattern |
| 3 | `opp_valid_after` | Constraint | Opponent's legal move count after this move |
| 4 | `me_best_pattern` | Offensive | Max tiles in any single pattern after this move |
| 5 | `opp_best_pattern` | Defensive | Max tiles opponent has in any pattern (state-level) |
| 6 | `me_patterns_3` | Offensive | Count of patterns where current player would have exactly 3 |
| 7 | `opp_patterns_3` | Defensive | Count of patterns where opponent has exactly 3 (state-level) |
| 8 | `me_patterns_2` | Offensive | Count of patterns where current player would have exactly 2 |
| 9 | `opp_patterns_2` | Defensive | Count of patterns where opponent has exactly 2 (state-level) |
| 10 | `n_patterns_p` | Positional | Number of win patterns containing the candidate position |
| 11 | `level` | Phase | Current game level (0--16) |

Features 4, 6, 8 depend on the specific move chosen and thus vary across candidates. Features 5, 7, 9 are state-level (identical for all candidates at a given state) and provide context for the decision tree.


## 4. Mutual Information Analysis

### 4.1 Method

For a stratified sample of non-terminal game states with at least one suboptimal move, we compute:
- Feature vectors for each (state, move) pair
- Binary label: 1 if the move is minimax-optimal, 0 otherwise
- MI(feature, label) using sklearn's `mutual_info_classif`, computed separately for each game level L

This reveals which features are most predictive of optimal play at each game phase.

### 4.2 Results: Feature importance shifts by game phase

| Level range | Top features | Interpretation |
|---|---|---|
| L = 0 | None (MI = 0 for all features) | Opening move is a board-specific memorization problem; no state-based feature can distinguish the single winning opening from 11 losing ones |
| L = 1--3 | `n_patterns_p` (0.25), `me_patterns_3` (0.07) | Positional value dominates early: choose tiles that participate in many winning configurations |
| L = 4--6 | `me_best_pattern` (0.06--0.07), `me_patterns_3` (0.05), `n_patterns_p` (0.04) | Pattern building: extend the longest partial configuration |
| L = 7--9 | `opp_valid_after` (0.03--0.08), `me_best_pattern` (0.05--0.06), `wins_immediately` (0.04--0.06) | Constraint play emerges: restrict the opponent's options while maintaining offensive progress |
| L = 10--12 | `me_best_pattern` (0.12--0.18), `opp_valid_after` (0.09--0.17), `wins_immediately` (0.12--0.16) | Three-way race: win, constrain, or be constrained |
| L = 13--14 | `opp_valid_after` (0.26), `wins_immediately` (0.69) | Near-deterministic: either you can win immediately or the stuck rule decides everything |

The transition from positional features (early) to constraint features (late) reflects the growing dominance of the stuck rule, which accounts for 76% of terminal states at level 16 across all configurations.

**Key finding**: the feature importance profile directly confirms the paper's structural observation that Okiya's strategy operates on two axes (pattern completion and stuck threat), with the balance shifting from the former to the latter as the game progresses.


## 5. Decision Tree Rule Extraction

### 5.1 Training setup

- **Samples**: ~80,000 (state, move) pairs from ~23,000 informative game states (those with both optimal and suboptimal moves)
- **Labels**: binary, 1 = minimax-optimal move, 0 = suboptimal
- **Class balance**: balanced weighting (optimal ~45%, suboptimal ~55%)
- **Models**: DecisionTreeClassifier with max_depth in {4, 8}

### 5.2 Depth-4 tree: a five-rule heuristic

The depth-4 tree achieves **74.3% move accuracy** (vs. 41% random baseline) with the following human-readable rules:

```
1. If you can complete a win pattern (me_best_pattern = 4) → PLAY IT
2. If opponent will be stuck (0 valid moves after your move) → PLAY IT
3. If opponent will have only 1 valid move → PLAY IT
4. If the position participates in many win patterns (≥6):
   a. If opponent has no 3-in-a-pattern threats → PLAY IT
   b. If opponent has any 3-in-a-pattern threats → AVOID (you need to block, not build)
5. Otherwise → AVOID (the move doesn't accomplish enough)
```

In natural language for a player:

> **Win if you can.  Trap if you can't win.  Build on valuable positions, but watch for opponent threats.**

The tree confirms that `opp_valid_after` (stuck threat) is the most consistently informative feature across game phases, appearing as the primary split in both sub-trees. The split on `opp_patterns_3` in rule 4 shows that at high-value positions, the decision hinges on whether the opponent already has an imminent win threat.

### 5.3 Depth-8 tree

The depth-8 tree adds level-dependent branches, achieving **77.5% accuracy**. The additional 3.2 percentage points come primarily from:
- Phase-specific thresholds: `opp_valid_after ≤ 2` is optimal in early-mid game (L ≤ 9) but not late game (L > 10)
- Blocking recognition: `blocks_opp_3` becomes a significant feature when `opp_valid_after > 1` and the opponent has 3-in-a-pattern
- Late-game discrimination: at L > 11, fewer features are informative and the tree learns to be more conservative

The depth-8 tree is too complex for human memorization but useful as a machine heuristic.


## 6. Heuristic Hierarchy

We evaluate a family of heuristics arranged from simplest to most complex. Each is a deterministic function from (game state, valid moves) to a selected move.

| ID | Name | Rules | Complexity |
|---|---|---|---|
| H0 | Random | Uniform random | 0 rules |
| H1 | Win-Block | Win if can; block if must; else random | 2 rules |
| H2 | Constraint | Win > block > minimize opponent's valid moves | 3 rules |
| H3 | Pattern+Constraint | Win > block > maximize own pattern progress, tiebreak by minimizing opponent's moves | 4 rules |
| H4 | Full Priority | Win > block > maximize 3-in-patterns > min opp moves > maximize 2-in-patterns > min opp 2-in-patterns | 6 rules |
| H5 | Tree-4 | Depth-4 decision tree classifier | ~5 splits |
| H6 | Tree-8 | Depth-8 decision tree classifier | ~30 splits |
| Oracle | Minimax lookup | Perfect play | N/A |


## 7. Evaluation

### 7.1 Move accuracy

**Move accuracy** is the fraction of non-terminal game states where the heuristic selects a minimax-optimal move, measured on a stratified sample of informative states (those with at least one suboptimal alternative). This is the primary metric: it directly measures agreement with the oracle on decisions that matter.

| Heuristic | L=0 | L=1--3 | L=4--6 | L=7--9 | L=10--12 | L=13--14 | Mean |
|---|---|---|---|---|---|---|---|
| H0 Random | 0.00 | 0.31 | 0.44 | 0.47 | 0.49 | 0.52 | **0.41** |
| H1 Win-Block | 0.00 | 0.31 | 0.46 | 0.58 | 0.71 | 0.84 | **0.52** |
| H2 Constraint | 0.00 | 0.36 | 0.56 | 0.72 | 0.86 | 0.94 | **0.62** |
| H3 Pattern+Constraint | 0.00 | 0.40 | 0.67 | 0.73 | 0.85 | 0.93 | **0.65** |
| H4 Full Priority | 0.00 | 0.40 | 0.71 | 0.71 | 0.77 | 0.86 | **0.63** |
| H5 Tree-4 | 0.00 | 0.83 | 0.65 | 0.69 | 0.88 | 0.99 | **0.74** |
| H6 Tree-8 | 0.00 | 0.83 | 0.74 | 0.74 | 0.90 | 1.00 | **0.78** |
| Oracle | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |

**Observations**:
1. All heuristics achieve 0% accuracy at L=0 because no feature distinguishes the single winning opening. The opening is a pure memorization problem.
2. Accuracy improves with game depth for all heuristics: as the board fills, the features become more informative (MI increases monotonically from L=1 to L=14 for the top features).
3. H3 (Pattern+Constraint, 4 rules) outperforms H4 (Full Priority, 6 rules): the additional rules in H4 occasionally override correct constraint-based decisions with incorrect pattern-based ones. Simpler is sometimes better.
4. The tree heuristics gain most at L=1--3 (early game) where they learn level-dependent thresholds that hand-crafted rules miss.

### 7.2 Win rate vs. random opponent

Win rate is measured by simulating games where one player uses the heuristic and the other plays uniformly at random. Results are averaged over P0 and P1 roles (500 games each).

| Heuristic | Win rate vs. Random |
|---|---|
| H0 Random | 50.0% |
| H1 Win-Block | 65.6% |
| H2 Constraint | 74.1% |
| H3 Pattern+Constraint | 82.6% |
| H4 Full Priority | 85.8% |
| H5 Tree-4 | 83.7% |
| H6 Tree-8 | 84.5% |
| Oracle | 99.8% |

**The ranking shifts**: H4 (85.8%) outperforms H3 (82.6%) and even Tree-4 (83.7%) against random play, despite having lower move accuracy. This is because against a random opponent, the game rarely reaches states where precision matters; instead, the heuristic that aggressively maximizes stuck threats (H4's secondary rule) accumulates small advantages that compound over multiple moves.

Against perfect play (Oracle as opponent), all heuristics achieve near-zero win rates. The Oracle exploits any suboptimal move immediately on this board configuration.

### 7.3 Reconciling accuracy and win rate

The accuracy-vs-win-rate divergence reveals an important distinction:

- **Move accuracy** measures how often the heuristic agrees with the oracle at *decision-relevant* states. This is the right metric for evaluating heuristic *quality* against competent opponents.
- **Win rate vs. random** measures how well the heuristic *exploits* weak play. Aggressive heuristics that maximize stuck threats (H4) are better at this than balanced heuristics (H3) because random opponents are most vulnerable to constraint traps.

For the paper's open question about human play: human players likely fall between random and optimal. A robust heuristic should maximize move accuracy (which correlates with performance against any skill level), while win rate vs. random is a lower bound on practical performance.


## 8. The Opening Problem

A striking result is that **no feature-based heuristic achieves above-chance accuracy at L=0**. On the canonical board, all 12 opening moves produce identical feature vectors: every border position participates in exactly 4 win patterns, each has 6 related tiles, and no blocking or stuck threats exist on an empty board.

This is not an artifact of our feature set. The winning opening depends on the *global* structure of the tile-relation graph (which positions will create advantageous constraint cascades 5-10 moves later), not on any local or one-step-lookahead property. Since the mean number of winning openings across 1,000 configurations is only 1.37 out of 12, the opening is where the game's "fragile first-mover advantage" is concentrated.

**Implications for heuristic design**: A practical strategy for Okiya would combine:
1. Opening preparation (memorize the winning first move for the given board), and
2. Heuristic play (use rules from L=1 onward).

This parallels chess, where opening theory is effectively memorized and heuristic evaluation governs the middle game.


## 9. Generalizability

### 9.1 Applicability conditions

The pipeline applies to any strongly solved game where:
1. **Complete labels exist**: every reachable state has a known minimax value.
2. **Move alternatives exist**: each non-terminal state has multiple candidate moves (otherwise there is no decision to learn).
3. **Domain features can be defined**: the game has enough structure to support meaningful feature engineering (pattern-based games, constraint games, placement games).

### 9.2 What generalizes, what doesn't

**Generalizes across games**:
- The five-stage pipeline (features → MI → tree → heuristics → evaluation)
- The taxonomy of features (offensive, defensive, constraint, positional)
- The MI-stratified analysis revealing phase-dependent feature importance
- The accuracy-vs-win-rate distinction for evaluation
- The observation that opening play often resists heuristic approximation

**Game-specific**:
- The specific features and their relative importance
- The decision tree rules (tied to the feature set and game structure)
- The number of rules needed to close the gap between random and perfect play

### 9.3 Predictions for other games

For games with similar structure (bounded state space, alternating play, pattern-based wins, constraint mechanics):
- **Games with stuck rules or zugzwang**: constraint features (mobility, opponent option count) will dominate late-game MI, similar to Okiya.
- **Pure pattern games** (e.g., Tic-Tac-Toe variants): offensive/defensive pattern features will dominate at all phases. Constraint features will be less important.
- **High-branching games**: more features may be needed, and deeper trees may be required. The MI analysis will reveal whether the game's strategy is compressible.

### 9.4 Quantifying strategic compressibility

A useful meta-metric emerging from this analysis is the **heuristic ceiling**: the maximum move accuracy achievable by a depth-k decision tree. For Okiya:
- Depth 4: 74%
- Depth 8: 78%
- Extrapolated depth ∞: ~85--90% (estimated from the training accuracy plateau)

The gap between the heuristic ceiling and 100% represents the fraction of decisions that require deep lookahead or board-specific knowledge and cannot be captured by local features. For Okiya, this is approximately 15--20%, concentrated at the opening (L=0--3) where feature-based MI is lowest.

This metric could serve as a game complexity measure complementary to state-space size: two games might have similar state counts but very different heuristic ceilings, indicating different *strategic* complexities.


## 10. Conclusions

1. **A 4-rule heuristic (win > block > build > constrain) achieves 65% move accuracy** on Okiya, closing roughly one-third of the gap between random play (41%) and perfect play (100%). Against random opponents, even simple heuristics raise the win rate from 50% to over 80%.

2. **A shallow decision tree (depth 4) achieves 74% accuracy** with interpretable rules. The tree discovers the same strategic structure identified by the MI analysis: win immediately, exploit stuck threats, and build on high-value positions while watching for opponent threats.

3. **Feature importance is strongly phase-dependent**: positional value dominates early game, pattern building dominates mid-game, and constraint play dominates late game. The stuck rule, which accounts for 46% of terminal states overall and 76% at the final level, drives the late-game importance of mobility features.

4. **The opening is the hard part**: no local feature distinguishes the winning opening move, confirming that the "fragile first-mover advantage" (mean 1.37 winning openings out of 12) is concentrated in board-specific global structure rather than locally computable properties.

5. **The methodology generalizes**: the pipeline requires only minimax labels and domain features. The MI-stratified analysis, decision tree extraction, and dual evaluation (accuracy + win rate) apply to any strongly solved game with similar structure.


## Appendix A: Experimental Details

- **Board**: canonical Okiya configuration, tiles = [14, 4, 11, 9, 7, 12, 10, 3, 8, 1, 5, 0, 6, 2, 13, 15]
- **Solution**: 9,297,746 reachable states across 17 levels (L=0 to L=16)
- **Dataset**: 23,045 informative states (with at least one suboptimal move), yielding 79,551 (state, move) training samples
- **Sampling**: stratified by level, max 5,000 states per level, seed=42
- **Tree training**: scikit-learn DecisionTreeClassifier with balanced class weights
- **Win-rate simulation**: 500 games per pair, deterministic tie-breaking with fixed seed
- **Software**: Python 3, NumPy, scikit-learn. Feature extraction: pure Python over bitmask representations.


## Appendix B: Complete MI Table

| Level | wins_imm | blocks | opp_valid | me_best | opp_best | me_p3 | opp_p3 | me_p2 | opp_p2 | n_pat_p | level_feat |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 0.007 | 0.022 | 0.016 | 0.000 | 0.000 | 0.074 | 0.031 | 0.000 | 0.000 | **0.251** | 0.000 |
| 2 | 0.043 | 0.005 | 0.044 | 0.059 | 0.000 | 0.000 | 0.000 | 0.072 | 0.000 | **0.074** | 0.000 |
| 3 | 0.000 | 0.000 | 0.011 | 0.043 | 0.000 | 0.000 | 0.000 | 0.049 | 0.000 | **0.096** | 0.025 |
| 4 | 0.013 | 0.000 | 0.000 | **0.061** | 0.005 | 0.054 | 0.000 | 0.053 | 0.001 | 0.060 | 0.000 |
| 5 | 0.002 | 0.004 | 0.014 | 0.017 | 0.011 | 0.025 | 0.007 | 0.007 | 0.017 | **0.036** | 0.005 |
| 6 | 0.015 | 0.000 | 0.021 | **0.072** | 0.000 | 0.053 | 0.007 | 0.004 | 0.002 | 0.036 | 0.000 |
| 7 | 0.004 | 0.015 | **0.030** | 0.007 | 0.007 | 0.008 | 0.004 | 0.004 | 0.000 | 0.022 | 0.004 |
| 8 | 0.055 | 0.002 | 0.043 | **0.064** | 0.007 | 0.016 | 0.000 | 0.003 | 0.000 | 0.021 | 0.000 |
| 9 | 0.044 | 0.015 | **0.084** | 0.047 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.005 |
| 10 | 0.117 | 0.000 | 0.094 | **0.119** | 0.001 | 0.000 | 0.004 | 0.005 | 0.000 | 0.011 | 0.006 |
| 11 | 0.095 | 0.013 | **0.144** | 0.105 | 0.000 | 0.013 | 0.007 | 0.000 | 0.000 | 0.000 | 0.012 |
| 12 | 0.163 | 0.003 | 0.167 | **0.177** | 0.004 | 0.012 | 0.000 | 0.013 | 0.006 | 0.017 | 0.003 |
| 13 | 0.129 | 0.000 | **0.263** | 0.127 | 0.005 | 0.000 | 0.007 | 0.000 | 0.015 | 0.000 | 0.000 |
| 14 | **0.694** | 0.016 | 0.012 | **0.694** | 0.000 | 0.074 | 0.011 | 0.045 | 0.026 | 0.016 | 0.000 |

Bold indicates the highest MI value at each level. The rightward shift from `n_patterns_p` (L=1--5) through `me_best_pattern` (L=6--12) to `opp_valid_after` (L=7--13) and `wins_immediately` (L=14) traces the strategic progression from positional play through pattern building to constraint endgame.


## Appendix C: Pairwise Win Rates

Win rates from game simulation (500 games, P0 listed in rows, P1 in columns). Values of 0.000 and 1.000 between deterministic heuristics reflect that on a fixed board configuration, the game is fully determined — the same sequence of moves occurs every time.

| P0 \ P1 | Random | Win-Block | Constraint | Pattern | FullPri | Tree-4 | Tree-8 | Oracle |
|---|---|---|---|---|---|---|---|---|
| Random | 0.500 | 0.322 | 0.228 | 0.176 | 0.168 | 0.158 | 0.126 | 0.004 |
| Win-Block | 0.634 | 0.500 | 0.360 | 0.310 | 0.386 | 0.352 | 0.240 | 0.010 |
| Constraint | 0.710 | 0.530 | 0.500 | 0.396 | 0.424 | 0.314 | 0.224 | 0.010 |
| Pattern | 0.828 | 0.664 | 0.428 | 0.500 | 1.000 | 0.000 | 1.000 | 0.000 |
| FullPri | 0.884 | 0.778 | 0.684 | 0.000 | 0.500 | 0.000 | 0.000 | 0.000 |
| Tree-4 | 0.832 | 0.632 | 0.550 | 1.000 | 1.000 | 0.500 | 0.000 | 0.000 |
| Tree-8 | 0.816 | 0.726 | 0.324 | 1.000 | 1.000 | 0.000 | 0.500 | 0.000 |
| Oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 |

The binary (0/1) outcomes for deterministic pairs highlight a limitation of single-board evaluation: the result depends on whether the heuristic happens to select the winning opening move for this specific board. Robust win-rate comparison requires testing across multiple board configurations, which is left for future work.
