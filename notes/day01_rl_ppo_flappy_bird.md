# Day 1 — RL Project Setup, PPO Training, and Debugging

## What Was Done

Trained a PPO agent on FlappyBird-v0 using Stable Baselines3. Migrated from local environment to Google Colab.

## Key Learnings

### 1. RL Project Setup and Debugging

- **Local to Colab migration**: `render_mode="human"` works locally but not in headless environments like Colab. Use `rgb_array` + video recording instead.
- **Kernel state**: Jupyter cells are not independent. After a restart, imports and variables are lost and must be re-run.
- **Virtual display (Xvfb)**: Rendering is possible on headless servers through a fake screen buffer via `pyvirtualdisplay`.
- **Video debugging**: 0-second video meant the metadata was not finalized (moov atom issue). Fixed with `ffmpeg -c copy` without quality loss.
- **Warnings vs Errors**: Gym deprecation warnings, `datetime.utcnow()` deprecation, and Monitor wrapper messages are all harmless. The observation-space warning (values outside expected range) was genuinely worth noting.

### 2. RL Fundamentals

- **Neural network as stacked logistic regressions**: Each layer is a mini logistic regression, chained together.
- **Gradient descent**: Same concept as supervised learning, but the loss function is more complex because there are no correct labels — only rewards.
- **PPO total loss has 3 parts**:
  - **Value loss**: Literally linear regression (MSE). Predicts future reward.
  - **Policy loss**: Similar to logistic regression, but with a clipping safety mechanism that prevents large gradient steps.
  - **Entropy loss**: Forces the policy to maintain exploration, preventing it from becoming overconfident too quickly.

### 3. Training Instability

- Observed **policy collapse**: `ep_rew_mean` dropped from 18 to 4-6. The signal came from spikes in `clip_fraction` and `approx_kl`.
- **`target_kl` fix**: Mechanism to early-stop policy updates when KL divergence crosses a threshold.
- **Trade-off learned**: `target_kl=0.02` was too strict. Training became stable but very slow (only 1-2 out of 10 epochs used per batch). A real example of over-correction.
- **`explained_variance`**: The RL equivalent of R-squared. Measures how well the value function fits the data.

### 4. Statistical Evaluation

- **Sample size effect**: 5 episodes (137.16 ± 70.32) vs 30 episodes (94.53 ± 54.26). More samples give a more trustworthy estimate (Law of Large Numbers), even if the number looks worse.
- **Coefficient of Variation (CV = std/mean)**: High CV indicates the policy is inconsistent, not just bad.
- **Root cause analysis**: Reading the training log revealed the variance issue was not from training instability but from learning becoming too slow due to an over-conservative `target_kl`.

## Biggest Takeaway

Applying a fix is not enough. You must also check for side effects. `target_kl=0.02` fixed the collapse but made training so slow that the curve was still rising without plateauing. This is the iterative debugging process that happens daily in real ML engineering: fix, measure, find the next bottleneck.

## Next Action

Train with `target_kl=0.05` and `total_timesteps=800000`. Check if `ep_rew_mean` reaches a higher plateau.
