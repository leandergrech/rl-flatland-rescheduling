### Arrival rate (%), held-out seeds, no malfunctions

| Policy | small | medium | large | xlarge |
|---|---|---|---|---|
| OR: PP+SIPP+MCP | 100.0 ± 0.0 | 99.3 ± 0.7 | 98.2 ± 0.9 | 95.7 ± 1.1 |
| PPO (compact obs) | 60.0 ± 13.9 | 35.0 ± 8.7 | 14.3 ± 2.6 | 7.5 ± 1.5 |
| BC from OR | 39.0 ± 7.1 | 18.0 ± 4.3 | 13.0 ± 2.2 | 7.9 ± 0.8 |
| BC then PPO | 63.0 ± 12.4 | 23.7 ± 4.5 | 13.3 ± 1.7 | 6.4 ± 1.2 |
| PPO (tree obs) | 25.0 ± 8.1 | 24.7 ± 4.8 | 16.5 ± 1.7 | 16.0 ± 1.5 |
| Reactive heuristic | 82.0 ± 9.4 | 46.3 ± 10.7 | 25.3 ± 4.3 | 11.1 ± 2.1 |
| Shortest path, no coordination | 36.0 ± 5.4 | 24.7 ± 4.7 | 8.3 ± 0.7 | 5.7 ± 1.0 |


### Normalised reward, held-out seeds, no malfunctions

| Policy | small | medium | large | xlarge |
|---|---|---|---|---|
| OR: PP+SIPP+MCP | 0.995 ± 0.004 | 0.991 ± 0.006 | 0.972 ± 0.010 | 0.953 ± 0.007 |
| PPO (compact obs) | 0.810 ± 0.062 | 0.724 ± 0.043 | 0.626 ± 0.020 | 0.582 ± 0.015 |
| BC from OR | 0.734 ± 0.038 | 0.679 ± 0.026 | 0.658 ± 0.015 | 0.623 ± 0.010 |
| BC then PPO | 0.823 ± 0.058 | 0.700 ± 0.028 | 0.646 ± 0.013 | 0.614 ± 0.011 |
| PPO (tree obs) | 0.664 ± 0.044 | 0.751 ± 0.020 | 0.744 ± 0.012 | 0.737 ± 0.013 |
| Reactive heuristic | 0.885 ± 0.041 | 0.766 ± 0.043 | 0.689 ± 0.024 | 0.615 ± 0.012 |
| Shortest path, no coordination | 0.728 ± 0.038 | 0.656 ± 0.030 | 0.564 ± 0.009 | 0.538 ± 0.012 |


### Deadlocked trains at episode end (mean per episode), held-out seeds, no malfunctions

| Policy | small | medium | large | xlarge |
|---|---|---|---|---|
| OR: PP+SIPP+MCP | 0.0 | 0.0 | 0.0 | 0.0 |
| PPO (compact obs) | 0.0 | 0.9 | 2.4 | 7.6 |
| BC from OR | 1.1 | 5.8 | 13.7 | 24.6 |
| BC then PPO | 0.8 | 2.5 | 4.9 | 8.2 |
| PPO (tree obs) | 0.0 | 0.2 | 0.2 | 1.0 |
| Reactive heuristic | 0.0 | 1.2 | 6.4 | 6.3 |
| Shortest path, no coordination | 3.6 | 17.9 | 30.7 | 61.7 |


### Arrival rate (%), same seeds with malfunctions

| Policy | small | medium | large | xlarge |
|---|---|---|---|---|
| OR: PP+SIPP+MCP | 100.0 ± 0.0 | 98.0 ± 1.4 | 93.8 ± 1.6 | 89.1 ± 2.2 |
| PPO (compact obs) | 60.0 ± 13.9 | 36.0 ± 8.3 | 14.2 ± 3.3 | 7.5 ± 1.6 |
| BC from OR | 41.0 ± 7.1 | 19.7 ± 4.2 | 15.2 ± 2.8 | 8.2 ± 1.1 |
| BC then PPO | 61.0 ± 11.3 | 20.7 ± 3.6 | 15.3 ± 2.3 | 6.9 ± 1.6 |
| PPO (tree obs) | 22.0 ± 7.7 | 22.7 ± 4.5 | 16.3 ± 1.6 | 15.0 ± 1.5 |
| Reactive heuristic | 76.0 ± 9.7 | 45.7 ± 9.7 | 22.3 ± 4.0 | 9.8 ± 1.3 |
| Shortest path, no coordination | 36.0 ± 5.4 | 23.7 ± 4.0 | 9.2 ± 0.7 | 5.2 ± 0.9 |


### Normalised reward, same seeds with malfunctions

| Policy | small | medium | large | xlarge |
|---|---|---|---|---|
| OR: PP+SIPP+MCP | 0.991 ± 0.004 | 0.984 ± 0.009 | 0.952 ± 0.013 | 0.922 ± 0.007 |
| PPO (compact obs) | 0.804 ± 0.060 | 0.729 ± 0.041 | 0.633 ± 0.020 | 0.580 ± 0.015 |
| BC from OR | 0.741 ± 0.039 | 0.677 ± 0.025 | 0.660 ± 0.014 | 0.595 ± 0.009 |
| BC then PPO | 0.821 ± 0.051 | 0.676 ± 0.026 | 0.654 ± 0.016 | 0.611 ± 0.014 |
| PPO (tree obs) | 0.656 ± 0.043 | 0.747 ± 0.019 | 0.742 ± 0.012 | 0.732 ± 0.014 |
| Reactive heuristic | 0.867 ± 0.040 | 0.761 ± 0.038 | 0.679 ± 0.024 | 0.612 ± 0.010 |
| Shortest path, no coordination | 0.727 ± 0.038 | 0.649 ± 0.026 | 0.570 ± 0.010 | 0.537 ± 0.013 |


### Deadlocked trains at episode end (mean per episode), same seeds with malfunctions

| Policy | small | medium | large | xlarge |
|---|---|---|---|---|
| OR: PP+SIPP+MCP | 0.0 | 0.0 | 0.0 | 0.0 |
| PPO (compact obs) | 0.0 | 0.9 | 2.0 | 7.8 |
| BC from OR | 1.1 | 5.9 | 15.4 | 30.9 |
| BC then PPO | 1.0 | 2.4 | 4.3 | 8.0 |
| PPO (tree obs) | 0.0 | 0.2 | 0.5 | 1.4 |
| Reactive heuristic | 0.0 | 1.0 | 5.5 | 6.1 |
| Shortest path, no coordination | 3.6 | 16.6 | 33.5 | 61.4 |


### Policy wall-clock per env step (ms, mean over episodes, no malfunctions)

| Policy | small | medium | large | xlarge |
|---|---|---|---|---|
| OR: PP+SIPP+MCP | 0.05 | 0.17 | 0.45 | 1.27 |
| PPO (compact obs) | 5.96 | 11.31 | 15.90 | 32.28 |
| BC from OR | 0.66 | 2.00 | 2.88 | 23.16 |
| BC then PPO | 0.84 | 2.27 | 3.54 | 5.86 |
| PPO (tree obs) | 2.56 | 11.34 | 16.22 | 16.01 |
| Reactive heuristic | 3.14 | 8.65 | 11.89 | 4.06 |
| Shortest path, no coordination | 0.06 | 0.13 | 0.27 | 0.45 |


### One-off setup per episode (s; the OR planner's initial plan)

| Policy | small | medium | large | xlarge |
|---|---|---|---|---|
| OR: PP+SIPP+MCP | 0.05 | 0.54 | 2.73 | 7.05 |
| PPO (compact obs) | 0.00 | 0.00 | 0.00 | 0.00 |
| BC from OR | 0.00 | 0.00 | 0.00 | 0.00 |
| BC then PPO | 0.00 | 0.00 | 0.00 | 0.00 |
| PPO (tree obs) | 0.00 | 0.00 | 0.00 | 0.00 |
| Reactive heuristic | 0.00 | 0.00 | 0.00 | 0.00 |
| Shortest path, no coordination | 0.03 | 0.17 | 0.43 | 0.61 |

