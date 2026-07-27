# NBA Scout Intelligence Platform — Complete Project Context

## 1. Project Summary

| Field               | Description                                                                                                                                                                                                           |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Project name        | NBA Scout Intelligence Platform                                                                                                                                                                                       |
| Project type        | End-to-end machine learning and MLOps project                                                                                                                                                                         |
| Primary domain      | NBA player scouting, roster construction, player valuation, and decision support                                                                                                                                      |
| Primary users       | Scouts, basketball analysts, general managers, front-office staff, and coaching analysts                                                                                                                              |
| Main purpose        | Help users evaluate players, identify replacement candidates, estimate reasonable contract ranges, analyze roster fit, and compare acquisition options                                                                |
| Core product idea   | Given a target player, team context, budget, and strategy, return the top-K replacement candidates with basketball fit, expected performance, estimated salary range, acquisition difficulty, risks, and explanations |
| Product positioning | A decision-support platform, not an automated general manager and not a simple statistics dashboard                                                                                                                   |
| GitHub description  | An end-to-end ML and MLOps platform for NBA player scouting, replacement ranking, contract valuation, and roster-fit analysis.                                                                                        |

---

## 2. Problem Statement

NBA scouting decisions require more than comparing basic box-score statistics.

When a team evaluates a player, it must consider:

* The player’s current form.
* Whether recent performance is sustainable.
* The role the player performs.
* How similar the player is to an existing roster member.
* Whether the player complements the current roster.
* The player’s likely future contribution.
* A reasonable salary and contract range.
* The difficulty and cost of acquiring the player.
* Injury, age, contract, and performance risks.
* Whether another candidate provides better value for the same cost.

The platform should combine these factors into an explainable scouting recommendation.

The platform must not present recommendations as certain outcomes. It should communicate uncertainty, assumptions, confidence, and trade-offs.

---

## 3. Primary Product Question

The main product should answer:

> Given a target player, a team, a decision date, a roster-building strategy, and financial constraints, which available NBA players are the best replacement candidates?

Example:

> Find five players who could replace a meaningful portion of LeBron James’s role for the Los Angeles Lakers, prioritizing immediate competitiveness and an annual salary below a specified range.

The system should return candidates who can replace parts of the target player’s role.

It should not assume that a unique player such as LeBron James can be replaced perfectly by one person.

---

## 4. Main Use Cases

| Use case                     | User input                                                   | Expected output                                                                           |
| ---------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| Player analysis              | Player and decision date                                     | Current form, historical trend, role profile, projected performance, availability, risks  |
| Replacement search           | Target player, team, strategy, filters, budget               | Top-K replacement candidates with scores, salary ranges, risks, and explanations          |
| Need-based scouting          | Team needs, position, role, age, budget                      | Players who satisfy the specified basketball needs                                        |
| Player comparison            | Two to five players                                          | Side-by-side comparison of performance, role, fit, cost, risk, and future value           |
| Salary and contract analysis | Player and decision date                                     | Fair-market salary range, competitive range, contract-term range, comparable contracts    |
| Acquisition analysis         | Player, current contract, team context                       | Acquisition-difficulty tier, trade-value tier, likely asset-package category, constraints |
| Roster-fit analysis          | Player and target team                                       | Which team weaknesses the player addresses, role overlap, positional fit, and trade-offs  |
| Undervalued-player discovery | Candidate market and financial constraints                   | Players with strong projected impact relative to estimated cost                           |
| Strategy-specific scouting   | Win-now, rebuild, low-cost, defense-first, or shooting-first | Candidate ranking adjusted for the selected roster strategy                               |

---

## 5. Product Inputs

### 5.1. Core scouting request

| Input                    | Description                                                                    |
| ------------------------ | ------------------------------------------------------------------------------ |
| Target player            | Player whose role should be replaced or approximated                           |
| Target team              | Team considering the acquisition                                               |
| Decision date            | The historical or current date from which the analysis should be performed     |
| Strategy                 | Win-now, balanced, rebuild, low-cost, defense-first, shooting-first, or custom |
| Top K                    | Number of candidates to return                                                 |
| Maximum annual salary    | Optional financial constraint                                                  |
| Minimum and maximum age  | Optional age filters                                                           |
| Position or role filters | Optional position or archetype constraints                                     |
| Contract status          | Any, under contract, expiring, free agent, or team-controlled                  |
| Excluded players         | Players who should not appear in the result                                    |
| Ranking weights          | Optional custom importance assigned to each scoring dimension                  |

### 5.2. Example request

```json
{
  "target_player": "LeBron James",
  "target_team": "Los Angeles Lakers",
  "as_of_date": "2026-02-01",
  "strategy": "WIN_NOW",
  "top_k": 5,
  "maximum_annual_salary_usd": 30000000,
  "maximum_age": 31,
  "ranking_preferences": {
    "role_similarity": 0.30,
    "projected_impact": 0.25,
    "roster_fit": 0.20,
    "contract_value": 0.15,
    "future_value": 0.10
  }
}
```

---

## 6. Product Outputs

Each candidate recommendation should include:

| Output field            | Description                                                                     |
| ----------------------- | ------------------------------------------------------------------------------- |
| Rank                    | Candidate position in the final ranking                                         |
| Player identity         | Player name, team, age, and primary position                                    |
| Role similarity         | Similarity between the candidate’s basketball role and the target player’s role |
| Projected impact        | Expected future basketball contribution                                         |
| Roster fit              | Degree to which the player addresses the target team’s needs                    |
| Contract value          | Expected basketball value relative to the contract range                        |
| Future value            | Age, development trajectory, and long-term value                                |
| Availability            | Likelihood of being available and able to contribute                            |
| Salary range            | Estimated fair-market and competitive salary ranges                             |
| Contract-term range     | Reasonable expected contract length                                             |
| Acquisition difficulty  | Low, medium, high, or very high                                                 |
| Trade-value tier        | General category of assets likely required                                      |
| Risk level              | Injury, age, role, contract, or performance risk                                |
| Confidence              | Confidence level of the recommendation                                          |
| Strengths               | Reasons the player is recommended                                               |
| Trade-offs              | Capabilities lost relative to the target player                                 |
| Comparable players      | Similar players or comparable contracts                                         |
| Model and data metadata | Data date, feature version, and model versions used                             |

### Example recommendation

```json
{
  "rank": 1,
  "player": "Candidate A",
  "role_similarity": 0.88,
  "projected_impact_score": 0.82,
  "roster_fit_score": 0.90,
  "contract_value_score": 0.76,
  "salary_range": {
    "fair_market_low_usd": 21000000,
    "fair_market_high_usd": 26000000,
    "competitive_low_usd": 24000000,
    "competitive_high_usd": 29000000,
    "likely_term_min_years": 3,
    "likely_term_max_years": 4
  },
  "acquisition_difficulty": "HIGH",
  "risk_level": "MEDIUM",
  "confidence": 0.78,
  "strengths": [
    "Primary playmaking ability",
    "Strong transition creation",
    "Improves perimeter defense"
  ],
  "tradeoffs": [
    "Lower rebounding impact",
    "Less positional versatility",
    "Lower half-court rim pressure"
  ]
}
```

---

## 7. Important Product Principles

| Principle                 | Requirement                                                                      |
| ------------------------- | -------------------------------------------------------------------------------- |
| Decision support          | Recommendations assist human decision-makers and do not replace them             |
| Explainability            | Every recommendation must include reasons and trade-offs                         |
| Uncertainty               | Forecasts and valuations must include ranges or confidence                       |
| Context awareness         | Recommendations depend on team, strategy, date, role, and budget                 |
| Point-in-time correctness | Only information available at the decision date may be used                      |
| Financial realism         | Salary output must be a negotiation-aware range, not a single exact number       |
| Role-based comparison     | Players should be compared by basketball role, not only raw statistics           |
| Cost awareness            | The best basketball match may not be the best overall acquisition                |
| Candidate diversity       | Top-K results should include different types of solutions when appropriate       |
| Auditability              | The system should record which data and model versions produced a recommendation |

---

## 8. Basketball Role Representation

The system should represent each player as a role and contribution profile.

### Role dimensions

| Category              | Example dimensions                                                  |
| --------------------- | ------------------------------------------------------------------- |
| Scoring               | Scoring volume, shot creation, assisted scoring, isolation usage    |
| Rim pressure          | Rim attempts, free-throw generation, drives                         |
| Shooting              | Three-point volume, accuracy, catch-and-shoot, pull-up shooting     |
| Playmaking            | Assists, potential assists, ball-handling responsibility, turnovers |
| Rebounding            | Offensive and defensive rebound contribution                        |
| Perimeter defense     | Point-of-attack defense, steals, matchup difficulty                 |
| Interior defense      | Rim protection, blocks, defensive rebounding                        |
| Transition            | Transition frequency and efficiency                                 |
| Off-ball contribution | Cutting, movement, spacing, screening                               |
| Usage                 | Possession responsibility and offensive centrality                  |
| Versatility           | Number of positions and roles the player can perform                |
| Availability          | Games played, minutes stability, injury history                     |
| Context               | Team pace, teammates, role, starter or bench status                 |

The first version may use standardized feature vectors and weighted similarity.

Advanced representation-learning methods may be explored later, but they are not required for the initial product.

---

## 9. Machine Learning Components

The complete product may contain several independently evaluated ML components.

| Component                  | Purpose                                                 | Expected output                                               |
| -------------------------- | ------------------------------------------------------- | ------------------------------------------------------------- |
| Performance forecasting    | Estimate future player production                       | Expected PTS, AST, REB, efficiency, minutes, and availability |
| Player role representation | Represent player roles and playing styles               | Role vector or role-profile scores                            |
| Replacement retrieval      | Retrieve players similar to the target player           | Initial candidate pool                                        |
| Salary-range estimation    | Estimate a reasonable market range                      | Lower, median reference, and upper salary bounds              |
| Contract-term estimation   | Estimate reasonable contract duration                   | Expected contract-length range                                |
| Acquisition difficulty     | Estimate how difficult a player is to acquire           | Low, medium, high, or very high                               |
| Roster-fit scoring         | Estimate compatibility with a target team               | Fit score and addressed team needs                            |
| Risk estimation            | Evaluate injury, age, role, and contract risk           | Risk categories and confidence                                |
| Final ranking              | Combine all candidate information                       | Final top-K recommendations                                   |
| Explanation generation     | Convert model outputs into useful scouting explanations | Strengths, trade-offs, and reason codes                       |

These components should be evaluated independently before being combined into the final ranking.

---

## 10. Performance Forecasting

### Forecasting targets

* Points.
* Assists.
* Rebounds.
* Minutes.
* Usage.
* True shooting percentage.
* Turnovers.
* Games played or availability probability.
* Performance over the next 5 or 10 games.
* Performance over the remainder of a defined period.

### Candidate features

* Rolling averages.
* Exponentially weighted averages.
* Season-to-date performance.
* Recent minutes and usage.
* Opponent quality.
* Home or away.
* Rest days.
* Back-to-back status.
* Starting status.
* Team injuries.
* Team role.
* Age and career baseline.

### Required baselines

* Last-game performance.
* Last-5 average.
* Last-10 average.
* Season-to-date average.
* Exponentially weighted moving average.

A complex model should only be used if it improves on these baselines under temporal backtesting.

The existing NBA LSTM model may be reused as one candidate model, but it must compete against simpler baselines.

---

## 11. Salary and Contract Range

The salary component should not predict one exact salary.

The player can negotiate, multiple teams may compete, and contract structure depends on market conditions.

### Required output

| Output                     | Meaning                                                               |
| -------------------------- | --------------------------------------------------------------------- |
| Fair-market annual range   | Reasonable salary based on performance and comparable players         |
| Competitive annual range   | Range that may occur under stronger market competition                |
| Likely contract-term range | Reasonable number of contract years                                   |
| Total contract range       | Approximate total commitment                                          |
| Salary tier                | Minimum, rotation, starter, high-level starter, star, or maximum tier |
| Confidence                 | Reliability of the range                                              |
| Comparable contracts       | Historical contracts used as references                               |

### Example

```text
Fair annual salary range: $18M–$23M
Competitive market range: $21M–$27M
Likely contract term: 3–4 years
Confidence: Medium
```

The range may be created through quantile regression, conformal prediction, comparable-player retrieval, or a hybrid method.

---

## 12. Acquisition Cost

Salary and acquisition cost are separate concepts.

### Free-agent acquisition

Possible outputs:

* Expected salary range.
* Contract-length range.
* Cap-space requirement.
* Competition level.
* Signing difficulty.
* No trade assets required.

### Trade acquisition

Possible outputs:

* Existing salary commitment.
* Remaining contract years.
* Salary-matching difficulty.
* Trade-value tier.
* Likely asset-package category.
* Acquisition difficulty.
* Major transaction blockers.

### Example trade output

```text
Acquisition difficulty: High

Likely asset-package tier:
- Meaningful young rotation player
- Premium draft capital
- Salary-matching contract
```

The system should not claim that an exact package is certain.

---

## 13. Team Needs and Roster Fit

A team should be represented through a team-needs profile.

### Possible team-need dimensions

* Primary shot creation.
* Secondary playmaking.
* Three-point shooting.
* Perimeter defense.
* Rim protection.
* Rebounding.
* Transition offense.
* Bench scoring.
* Positional depth.
* Off-ball movement.
* Lineup size.
* Salary flexibility.

Roster fit should consider:

* Which team weaknesses the candidate addresses.
* Whether the candidate duplicates an existing role.
* Compatibility with core players.
* Positional balance.
* Offensive-system fit.
* Defensive-system fit.
* Salary and contract constraints.
* Win-now versus long-term value.

The system should describe fit as an estimate rather than a guaranteed causal impact.

---

## 14. Scouting Strategies

| Strategy       | Ranking priorities                                              |
| -------------- | --------------------------------------------------------------- |
| Win-now        | Current impact, playoff readiness, role fit, availability       |
| Balanced       | Current impact, future value, fit, and contract value           |
| Rebuild        | Age, potential, development trajectory, team control            |
| Low-cost       | Performance per dollar, contract value, acquisition feasibility |
| Defense-first  | Defensive versatility, perimeter defense, rim protection        |
| Shooting-first | Shooting volume, spacing, off-ball contribution                 |
| Custom         | User-defined weights and constraints                            |

The same target player should produce different rankings under different strategies.

---

## 15. Candidate Ranking

The final ranking may combine:

* Role similarity.
* Projected impact.
* Roster fit.
* Contract value.
* Future potential.
* Availability.
* Injury risk.
* Acquisition difficulty.

General concept:

```text
Final candidate score =
    role similarity contribution
  + projected impact contribution
  + roster-fit contribution
  + contract-value contribution
  + future-value contribution
  + availability contribution
  - injury-risk penalty
  - acquisition-difficulty penalty
```

Exact formulas and weights should be treated as explicit project decisions, not hidden assumptions.

The platform should preserve component scores rather than returning only one unexplained total score.

---

## 16. Candidate Diversity

The top-K list should not always contain candidates of the same type.

When appropriate, the result may contain:

| Candidate type     | Description                                      |
| ------------------ | ------------------------------------------------ |
| Best role match    | Most similar basketball role                     |
| Best value         | Strongest projected impact relative to cost      |
| Best immediate fit | Most suitable for the current roster             |
| Best future option | Younger candidate with higher long-term value    |
| Most attainable    | Candidate with lower acquisition difficulty      |
| High-upside option | Higher uncertainty but potentially strong return |

Candidate diversification should be transparent and should not hide lower ranking scores.

---

## 17. Data Requirements

### Core data entities

| Entity                   | Example information                                      |
| ------------------------ | -------------------------------------------------------- |
| Player                   | Identity, age, height, weight, position, experience      |
| Team                     | Team identity, pace, ratings, roster composition         |
| Game                     | Date, teams, location, result                            |
| Player game statistics   | Minutes, points, assists, rebounds, shooting, turnovers  |
| Player season statistics | Per-game, per-36, per-100, efficiency and impact metrics |
| Lineup                   | Players, minutes, offensive and defensive performance    |
| Contract                 | Salary, term, guarantees, options, status                |
| Transaction              | Trade, signing, extension, release, waiver               |
| Injury                   | Type, date, duration, games missed, recurrence           |
| Salary-cap environment   | Season cap, tax line, exceptions, financial context      |

### Data-layer concept

| Layer  | Purpose                               |
| ------ | ------------------------------------- |
| Bronze | Immutable raw source snapshots        |
| Silver | Cleaned, normalized, canonical tables |
| Gold   | Point-in-time ML features and labels  |

Each record should preserve enough metadata to identify:

* Source.
* Ingestion time.
* Source update time.
* Season.
* Snapshot.
* Feature version.
* Decision date.

---

## 18. Temporal Correctness and Leakage Rules

The system must follow point-in-time correctness.

For a prediction made at date `T`, only information available at or before `T` may be used.

The project must avoid:

* Using full-season averages to make midseason predictions.
* Using future games.
* Using future injury information.
* Using a contract outcome to predict that same contract.
* Using a completed trade to estimate its pre-trade acquisition difficulty.
* Random splitting of time-dependent player-game records.
* Fitting preprocessing transformations on the full dataset.
* Building historical candidate pools using present-day information.

Temporal leakage prevention is a core quality requirement.

---

## 19. Evaluation Requirements

### 19.1. Performance forecasting

Required evaluation:

* Temporal or walk-forward validation.
* MAE.
* RMSE.
* WAPE where appropriate.
* Baseline improvement.
* Prediction-interval coverage.
* Prediction-interval width.
* Error by position.
* Error by age group.
* Error by minutes tier.
* Error by starter or bench role.
* Error by usage tier.

### 19.2. Salary ranges

Required evaluation:

* Pinball loss.
* Median absolute error.
* Interval coverage.
* Average interval width.
* Coverage by salary tier.
* Underpricing rate.
* Overpricing rate.

A range that is always extremely wide should not be considered useful even if it has high coverage.

### 19.3. Replacement ranking

Required evaluation:

* Precision@K.
* Recall@K.
* NDCG@K.
* Mean reciprocal rank.
* Pairwise ranking accuracy.
* Top-K stability.
* Rank correlation under small weight changes.
* Human review through scouting case studies.

### 19.4. Acquisition difficulty

Required evaluation:

* Accuracy.
* Macro F1.
* Confusion matrix.
* Calibration.
* Error by acquisition tier.

### 19.5. End-to-end evaluation

Each end-to-end test case should include:

* Target player.
* Target team.
* Decision date.
* Strategy.
* Budget.
* Eligible candidate pool.
* Expected candidate groups.
* Important exclusions.

The final system should be evaluated for:

* Candidate relevance.
* Budget compliance.
* Candidate diversity.
* Explanation quality.
* Stability.
* Confidence calibration.
* Output latency.

---

## 20. Golden Scouting Evaluation Set

The project should eventually include a manually reviewed evaluation set.

Example scenarios:

* Replace a primary playmaker.
* Replace a high-usage scoring wing.
* Replace a three-and-D wing.
* Replace a stretch center.
* Replace a defensive point guard.
* Find a low-cost bench scorer.
* Find a young rebuilding asset.
* Find a playoff-ready veteran.
* Find a shooter who does not require high usage.
* Find a versatile defender under a salary constraint.

Each scenario should be manually judged for:

* Role relevance.
* Team fit.
* Cost realism.
* Candidate diversity.
* Explanation usefulness.
* Important missing trade-offs.

---

## 21. Explainability Requirements

Each recommendation should explain:

### Why the player is recommended

Examples:

* Similar primary playmaking responsibility.
* Strong transition creation.
* Provides better perimeter defense.
* Adds three-point spacing.
* Fits the specified salary range.
* Younger with stronger future value.
* More attainable than higher-ranked alternatives.

### Main trade-offs

Examples:

* Lower rebounding contribution.
* Less positional versatility.
* Lower rim pressure.
* Higher injury uncertainty.
* More limited half-court creation.
* Higher acquisition difficulty.
* Shorter expected competitive window.

Technical model explanations such as feature importance or SHAP may be used for internal analysis.

User-facing explanations should be expressed through clear basketball reason codes and natural-language summaries.

---

## 22. Confidence and Uncertainty

The platform must distinguish between:

* Observed factual data.
* Model prediction.
* Estimated range.
* Heuristic score.
* Human-defined constraint.
* Low-confidence inference.

Confidence should reflect factors such as:

* Data completeness.
* Sample size.
* Model calibration.
* Similarity to training examples.
* Injury uncertainty.
* Contract-market uncertainty.
* Availability of comparable transactions.

The system should not express high confidence when data is limited.

---

## 23. MLOps Scope

The complete project should demonstrate the following capabilities.

| MLOps area          | Project expectation                                                            |
| ------------------- | ------------------------------------------------------------------------------ |
| Data versioning     | Training and evaluation datasets should be reproducible                        |
| Feature versioning  | Models should record the feature definitions they use                          |
| Experiment tracking | Model parameters, metrics, artifacts, and code versions should be tracked      |
| Model registry      | Candidate, challenger, and production models should be identifiable            |
| Baseline comparison | Every model must be compared against an appropriate baseline                   |
| Evaluation gates    | Models should not be promoted based on one overall metric                      |
| Reproducibility     | A run should be tied to data, features, code, configuration, and random seed   |
| Monitoring          | Data quality, drift, prediction distribution, latency, and delayed performance |
| Retraining          | New models should be evaluated before replacing current models                 |
| Auditability        | Each recommendation should record relevant model and data versions             |

MLflow is expected to be used for experiment tracking and model lifecycle management.

The exact implementation and infrastructure decisions are outside this project-context document.

---

## 24. MLflow Experiment Groups

Potential experiment groups:

```text
nba-performance-forecast
nba-player-role-representation
nba-salary-range
nba-acquisition-difficulty
nba-roster-fit
nba-final-ranking
```

A model-training run should eventually record:

* Model family.
* Hyperparameters.
* Training period.
* Validation period.
* Data snapshot.
* Feature version.
* Target definition.
* Prediction horizon.
* Random seed.
* Overall metrics.
* Slice metrics.
* Baseline improvement.
* Evaluation reports.
* Error analysis.
* Model card.
* Code commit.

---

## 25. Monitoring Goals

### Data monitoring

* Missing values.
* Duplicate records.
* Schema changes.
* Invalid ranges.
* Player and team ID mismatches.
* Late-arriving data.
* Contract inconsistencies.

### Feature monitoring

* Age distribution.
* Minutes distribution.
* Usage distribution.
* Role-profile distribution.
* Position distribution.
* Salary distribution.
* Team-need distribution.

### Prediction monitoring

* Forecast distributions.
* Salary-range widths.
* Acquisition-tier distribution.
* Confidence distribution.
* Top-K candidate diversity.
* Frequency of repeated recommendations.

### Delayed ground-truth monitoring

* Forecast errors after games are played.
* Salary-range coverage after contracts are signed.
* Ranking quality on later reviewed cases.
* Error by player subgroup.
* Changes in model calibration.

### Service monitoring

* Request volume.
* Error rate.
* Latency.
* Failed model loading.
* Data freshness.
* Missing model-version metadata.

---

## 26. Project Phases

| Phase   | Main objective                          | Expected result                                            |
| ------- | --------------------------------------- | ---------------------------------------------------------- |
| Phase 0 | Define product and evaluation contracts | Clear scope, use cases, inputs, outputs, and non-goals     |
| Phase 1 | Build data foundation                   | Clean, versioned, point-in-time player and game data       |
| Phase 2 | Build performance intelligence          | Baselines, forecasting candidates, temporal evaluation     |
| Phase 3 | Build replacement finder                | Role vectors, candidate retrieval, top-K evaluation        |
| Phase 4 | Build salary and contract ranges        | Negotiation-aware salary intervals and comparables         |
| Phase 5 | Build acquisition analysis              | Acquisition tiers and transaction constraints              |
| Phase 6 | Build roster fit and final ranking      | Strategy-aware end-to-end recommendations                  |
| Phase 7 | Build product serving                   | API and user-facing scouting workflow                      |
| Phase 8 | Add production MLOps                    | Tracking, monitoring, retraining, and deployment processes |

---

## 27. MVP Definition

The minimum valuable product should support:

1. Selecting a target player.
2. Selecting a target team.
3. Choosing a roster strategy.
4. Applying age and salary constraints.
5. Retrieving a candidate pool.
6. Returning top-K replacement candidates.
7. Showing role similarity.
8. Showing projected performance.
9. Showing roster fit.
10. Showing a salary range.
11. Showing acquisition difficulty.
12. Showing risks, confidence, strengths, and trade-offs.
13. Recording the decision date and model/data versions.

The MVP does not need to solve every NBA transaction rule.

---

## 28. Non-Goals

The project should not initially attempt to:

* Replace professional scouts.
* Guarantee player performance.
* Predict one exact salary.
* Predict one exact trade package.
* Fully implement every NBA collective bargaining rule.
* Automatically execute trades.
* Claim causal changes in wins from adding a player.
* Model all lineup combinations in the first version.
* Use deep learning for every component.
* Build a production-scale distributed system before the core ML problems are validated.
* Create an LLM chatbot as the central product.
* Optimize for visual frontend complexity before model quality and evaluation are established.

---

## 29. Important Distinctions

| Concept A            | Concept B                 | Difference                                                           |
| -------------------- | ------------------------- | -------------------------------------------------------------------- |
| Player quality       | Player role               | A strong player may not perform the needed role                      |
| Role similarity      | Roster fit                | A player can resemble the target but fit the team poorly             |
| Salary range         | Acquisition cost          | Salary is only one part of obtaining a player                        |
| Projected impact     | Historical production     | Future expectations should not equal past averages                   |
| Market value         | Final negotiated contract | Negotiation can move the final value                                 |
| Recommendation score | Certainty                 | A high score does not guarantee success                              |
| Candidate retrieval  | Final ranking             | Retrieval finds plausible candidates; ranking evaluates full context |
| Correlation          | Causal impact             | Historical association does not prove roster improvement             |

---

## 30. Example End-to-End Scenario

### User request

> Find five possible replacements for LeBron James for the Los Angeles Lakers. Prioritize immediate competitiveness, secondary playmaking, rim pressure, and perimeter defense. Exclude players older than 31 and candidates whose expected annual salary range is mostly above $30M.

### System process

1. Resolve the target player and team.
2. Build the target player’s role profile as of the decision date.
3. Build the Lakers’ team-needs profile.
4. Apply age, salary, availability, and contract filters.
5. Retrieve players with similar role profiles.
6. Forecast each candidate’s future performance.
7. Estimate each candidate’s salary range.
8. Estimate acquisition difficulty.
9. Calculate roster-fit components.
10. Apply the win-now ranking strategy.
11. Diversify the final top-K list when appropriate.
12. Generate strengths, trade-offs, confidence, and metadata.

### Expected result categories

* Best direct role replacement.
* Best cost-adjusted option.
* Best defensive fit.
* Best younger option.
* Most attainable option.

---

## 31. Definition of a Successful Project

The project is successful when it demonstrates that the developer can:

* Translate a real decision problem into ML tasks.
* Design point-in-time correct datasets.
* Build and compare meaningful baselines.
* Evaluate forecasting, ranges, ranking, and classification correctly.
* Represent uncertainty rather than hiding it.
* Combine multiple models into an explainable product.
* Track experiments and model lineage.
* Monitor data and prediction behavior.
* Produce reproducible results.
* Explain why a candidate is recommended.
* Identify limitations and avoid unrealistic claims.

---

## 32. Instructions for Any Coding Assistant

When using this project context:

* Treat this document as the product and ML specification.
* Do not invent missing business rules silently.
* Do not create player-similarity weights without explicit instructions.
* Do not create salary formulas without explicit instructions.
* Do not choose evaluation splits without explicit instructions.
* Do not replace ranges with exact predictions.
* Do not assume that a more complex model is better.
* Do not add infrastructure or frameworks unless requested.
* Preserve point-in-time correctness.
* Separate observed data, predictions, heuristic scores, and user-defined constraints.
* Make uncertainty and model metadata visible.
* Ask for or leave explicit placeholders for unresolved product decisions.
* Implement only the component requested in the current task.
* Avoid generating the entire system from this document in one step.
* Prefer small, testable, reviewable changes.
