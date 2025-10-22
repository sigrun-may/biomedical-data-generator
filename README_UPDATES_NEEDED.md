# VORGESCHLAGENE README-UPDATES

## Zeile 10 (Highlights) - KORRIGIERT:

```markdown
- **Multi-class generation** with exact `class_counts` control
```

ENTFERNT: ~~`weights` or~~ (existiert nicht mehr)

---

## Zeile 56-75 (Quickstart) - KORRIGIERT:

```python
from biomedical_data_generator import DatasetConfig, generate_dataset

cfg = DatasetConfig(
    n_samples=30,
    n_informative=5,
    n_pseudo=0,
    n_noise=0,
    n_classes=2,
    class_counts={0: 15, 1: 15},  # ← REQUIRED!
    class_sep=1.5,
    feature_naming="prefixed",
    random_state=42,
)

X, y, meta = generate_dataset(cfg, return_dataframe=True)
print(X.shape, y.shape)        # (30, 5), (30,)
print(meta.y_counts)           # {0: 15, 1: 15} (exact match!)
print(meta.feature_names[:5])  # ['i1', 'i2', 'i3', 'i4', 'i5']
```

---

## Zeile 90-110 (Correlated Clusters) - KORRIGIERT:

```python
from biomedical_data_generator import DatasetConfig, CorrCluster, generate_dataset

cfg = DatasetConfig(
    n_samples=200,
    n_informative=4,
    n_pseudo=1,
    n_noise=3,
    n_classes=3,
    class_counts={0: 70, 1: 70, 2: 60},  # ← REQUIRED!
    class_sep=1.2,
    corr_clusters=[
        CorrCluster(
            n_cluster_features=3,  # ← war "size"
            rho=0.7,
            structure="equicorrelated",
            anchor_role="informative",
            anchor_effect_size="large",  # ← war "anchor_beta"
            anchor_class=1,
        ),
        CorrCluster(
            n_cluster_features=2,  # ← war "size"
            rho=0.6,
            structure="toeplitz",
            anchor_role="pseudo",
        ),
    ],
    random_state=0,
)

X, y, meta = generate_dataset(cfg, return_dataframe=True)
```

---

## NEU: Class-specific Correlations Section

```markdown
## Class-specific correlations (class_rho)

Simulate biomarkers that **only correlate in diseased patients**:

```python
from biomedical_data_generator import DatasetConfig, CorrCluster, generate_dataset

cfg = DatasetConfig(
    n_samples=300,
    n_informative=0,
    n_pseudo=0,
    n_noise=0,
    corr_clusters=[
        CorrCluster(
            n_cluster_features=5,
            structure="equicorrelated",
            anchor_role="pseudo",
            class_rho={1: 0.9},    # Class 1 (diseased): HIGH correlation
            rho_baseline=0.1,      # Class 0 (healthy): LOW correlation
            label="Inflammation markers"
        )
    ],
    n_features=5,
    n_classes=2,
    class_counts={0: 150, 1: 150},
    random_state=42,
)

X, y, meta = generate_dataset(cfg)

# Result:
# - Healthy patients (class 0): biomarkers vary independently (ρ ≈ 0.1)
# - Diseased patients (class 1): biomarkers co-vary strongly (ρ ≈ 0.9)
```

**Use case:** Train ML models to detect subtle correlation patterns that emerge only in specific disease states (e.g., inflammation cascades, metabolic syndrome).
```

---

## Zeile 137-145 (Class balance) - KORRIGIERT:

```markdown
## Class balance and separability

- `n_classes`: number of classes (≥ 2)
- `class_counts`: **REQUIRED** - exact per-class sample counts
  (e.g., `{0: 50, 1: 30, 2: 20}`)
- `class_sep`: scales feature shifts → higher means easier separation

**Labels are generated deterministically** from `class_counts`. Each class gets
exactly the specified number of samples, shuffled randomly. Class separation is
achieved by shifting informative features and cluster anchors based on
`class_sep` and `anchor_effect_size`.
```

ENTFERNT: ~~`weights`~~ und ~~"softmax model"~~ Beschreibung

---

## Zeile 560-565 (Tips) - AKTUALISIERT:

```markdown
## Tips

- Prefer **omitting `n_features`**; let the config derive a consistent total.
- **Always specify `class_counts`** with exact per-class sizes
  (e.g., `{0: 100, 1: 50}` for 2:1 imbalance).
- Use `class_rho` to simulate disease-specific biomarker correlations.
- Start with `anchor_effect_size="medium"` and `class_sep≈1.2–1.5`;
  then tune for your lesson/demo.
- Keep `feature_naming="prefixed"` for didactics — it makes plots and
  tables easier to read.
```

---

## YAML Example - KORRIGIERT:

```yaml
n_samples: 200
n_classes: 3
class_counts:
  0: 70
  1: 70
  2: 60
class_sep: 1.2
n_informative: 4
n_pseudo: 1
n_noise: 3
feature_naming: prefixed
corr_clusters:
  - n_cluster_features: 3
    rho: 0.7
    structure: equicorrelated
    anchor_role: informative
    anchor_effect_size: large
    anchor_class: 1
  - n_cluster_features: 2
    rho: 0.6
    structure: toeplitz
    anchor_role: pseudo
random_state: 0
```

---

## Zusammenfassung der Änderungen:

### ENTFERNEN:
- ❌ Alle `weights` Referenzen
- ❌ "softmax model" Beschreibung
- ❌ `size` Parameter (→ `n_cluster_features`)
- ❌ `anchor_beta` Parameter (→ `anchor_effect_size`)

### HINZUFÜGEN:
- ✅ `class_counts` als **REQUIRED** markieren
- ✅ `class_counts` in ALLE Beispiele
- ✅ Neue Section für `class_rho`
- ✅ Aktualisierte CorrCluster Parameter
- ✅ Deterministische Label-Generierung beschreiben

### AKTUALISIEREN:
- ✅ Alle Code-Beispiele
- ✅ YAML-Beispiele
- ✅ API-Beschreibungen
- ✅ Tips Section
