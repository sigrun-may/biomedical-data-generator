# Quickstart

```yaml
# config.yaml
n_samples: 200
n_features: 13
n_informative: 4
n_pseudo: 2
n_noise: 4
n_classes: 3
class_weights: [0.5, 0.3, 0.2]
effect_size: "medium"
corr_between: 0.2
corr_clusters:
  - size: 3
    rho: 0.7
    structure: "equicorrelated"
    anchor_role: "informative"
    anchor_beta: 1.0
random_state: 42

output:
  format: "csv"
```
    