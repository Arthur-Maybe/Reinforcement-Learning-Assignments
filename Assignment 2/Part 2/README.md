# Part II: Evaluation on Gymnasium Envs & Robotic Manipulation Domain

## Hyperparameters
- $\alpha = 0.1$
- $\epsilon = 0.1$
- $\gamma = 0.9$

---

## Part (b)(i): Deterministic Robotic Manipulation Domain
**Optimal Goal Sequence**: Starting from $\theta_1 = 20^\circ, \theta_2 = 20^\circ$, the agent requires **5 steps** to reach $\theta_1 = 40^\circ, \theta_2 = -60^\circ$, close the gripper on the bolt, and move to $\theta_1 = 40^\circ, \theta_2 = -40^\circ$ to insert the bolt into the hole.

| Algorithm | Planning Steps ($n$) | Avg Steps/Episode (Final) | Episodes to First Near-Optimal Performance |
| :--- | :--- | :--- | :--- |
| **Dyna-Q** | $n = 0$ | 5.0 | ~105 |
| **Dyna-Q** | $n = 5$ | 5.0 | ~65 |
| **Dyna-Q** | $n = 10$ | 5.0 | ~42 |
| **Dyna-Q** | $n = 20$ | 5.0 | ~25 |
| **Dyna-Q** | $n = 50$ | 5.0 | ~12 |
| **Prioritized Sweeping** | $n = 0$ | 5.0 | ~105 |
| **Prioritized Sweeping** | $n = 5$ | 5.0 | ~35 |
| **Prioritized Sweeping** | $n = 10$ | 5.0 | ~20 |
| **Prioritized Sweeping** | $n = 20$ | 5.0 | ~11 |
| **Prioritized Sweeping** | $n = 50$ | 5.0 | ~6 |

---

## Part (b)(ii): Non-Deterministic Domain ("robotic manipulation nd")
Actions fail with probability $0.1$ (resulting in no state change), requiring the agent to build a stochastic transition model $\hat{P}(s' \mid s, a)$ using normalized empirical transition frequencies.

| Algorithm | Planning Steps ($n$) | Avg Steps/Episode (Final) | Episodes to First Near-Optimal Performance |
| :--- | :--- | :--- | :--- |
| **Stochastic Dyna-Q** | $n = 0$ | ~5.5 | ~140 |
| **Stochastic Dyna-Q** | $n = 5$ | ~5.5 | ~85 |
| **Stochastic Dyna-Q** | $n = 10$ | ~5.5 | ~52 |
| **Stochastic Dyna-Q** | $n = 20$ | ~5.5 | ~32 |
| **Stochastic Dyna-Q** | $n = 50$ | ~5.5 | ~16 |

---

## Comparative Analysis & Discussion
1. **Sample Efficiency**: In both deterministic and non-deterministic domains, increasing $n$ dramatically accelerates learning by propagating goal rewards backward through the model without requiring real environment interactions.
2. **Prioritized Sweeping vs. Dyna-Q**: Prioritized Sweeping focuses simulated updates strictly on state-action pairs with high top-level queue priorities ($|\delta| > \theta$), achieving optimal policies in roughly half the number of episodes required by standard Dyna-Q.
3. **Robustness to Stochastic Dynamics**: The count-based transition distribution estimator enables Dyna-Q to learn accurate $Q$-values without making false deterministic assumptions during model planning.