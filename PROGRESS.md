# Progress

## Current Status

- **Day:** 1
- **Date:** September 4, 2026
- **Phase:** 1 — Foundations

## Concepts Learned

### Mathematics
- (none yet)

### Machine Learning
- PPO (Proximal Policy Optimization): value loss, policy loss with clipping, entropy loss
- Neural networks as stacked logistic regressions
- Gradient descent in RL context (reward instead of labels)
- Training instability diagnosis via clip_fraction and approx_kl
- target_kl as early-stopping mechanism for policy updates
- explained_variance as RL equivalent of R-squared
- Statistical evaluation: sample size effect, coefficient of variation, root cause analysis

### Deep Learning
- (none yet)

### NLP
- (none yet)

### MLOps
- (none yet)

## Implementations

(none yet)

## Experiments

- `experiments/ppo_flappy_bird.ipynb` — PPO agent on FlappyBird-v0, 150k timesteps, reward from -7.6 to ~16. Environment: action_space=Discrete(2), observation_space=Box(0,1,(180,)). Diagnosed policy collapse, applied target_kl fix.

## Flagship Projects

(none yet)

## Interview Problems

(none yet)

## Benchmarks

(none yet)

## Weak Areas

To be identified as work begins.

## Next Actions

1. Retrain PPO with target_kl=0.05 and total_timesteps=800000
2. First learning objective: linear algebra — vectors and matrices
3. First implementation task: implement dot product, matrix multiplication, transpose from scratch
