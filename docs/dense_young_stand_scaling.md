---
icon: lucide/trees
---

# Dense young stand scaling

Some inventories record young stands with a very large number of stems.
Simulating such a stand as recorded is meaningless: no stand keeps that many stems as it grows up
([issue #105](https://github.com/Susi-pack/susi_26/issues/105),
[issue #312](https://github.com/Susi-pack/susi_26/issues/312)).

**Dense young stand scaling** is an option of `susi-xml-to-allometry` for those stands.
When it is on, the tool multiplies the stem count and basal area of a dense young stand by one
**scaling factor** before the growth model runs, so the stand's allometry file starts from the
scaled-down stand.

The option is **off by default**. Whether a stand's data should be corrected is your decision,
so you turn it on in the config file:

```toml
[dense_young_stand_scaling]
enabled = true
```

It is not a thinning. Nothing is harvested in any SUSI run, and it has nothing to do with the
`Thinning` cutting event: it only changes what the growth model is given when the allometry file
is made.

## The rule

A stand is scaled when **both** of these hold:

1. its mean diameter is **below** the diameter limit, and
2. its stem count is **above** the stem-count threshold for its main species.

A scaled stand starts from the target stem count for its main species.

The rule has five numbers, all set in the `[dense_young_stand_scaling]` table of the config file:

| Key | Default | Meaning |
|---|---|---|
| `max_mean_diameter` | `8.0` cm | Only stands with a mean diameter below this are scaled. |
| `stem_count_threshold_spruce` | `2200` stems/ha | A spruce-dominated stand is scaled when its stem count is above this. |
| `stem_count_threshold_other` | `2500` stems/ha | The same, for a pine- or deciduous-dominated stand. |
| `target_stem_count_spruce` | `1800` stems/ha | The stem count a scaled spruce-dominated stand starts from. |
| `target_stem_count_other` | `2000` stems/ha | The same, for a pine- or deciduous-dominated stand. |

Both comparisons are strict: a stand with exactly 2200 stems/ha, or a mean diameter of exactly
8.0 cm, is left alone.

A target can't be above its threshold. The tool refuses such a config file, because it would give
some stands more stems than the inventory recorded.

### What the rule reads

Everything comes from the stand's tree strata (one per species: pine, spruce, deciduous), not from
the stand-level summary the inventory also carries:

- **Stem count**: the sum of the three species' stem counts.
- **Mean diameter**: the three species' mean diameters, weighted by basal area. This is not the
  XML's `MeanDiameter` summary figure, which is computed differently and can disagree with it.
- **Main species**: the species with the largest basal area.

A stand with no stems or no basal area is never scaled: there is nothing to judge.

## What happens to a scaled stand

The scaling factor is the target stem count over the stand's stem count, so it is always below 1.
Every species' stem count and basal area are multiplied by that one factor. Age, mean diameter
and mean height are left as they are: the scaled stand has fewer trees of the same size, in the
same species proportions.

The growth model then runs on the scaled strata, and the allometry file is its output.

## Worked example: Paroninkorpi stand 20

Stand 20 of the Paroninkorpi example project is recorded like this:

| Species | Stems/ha | Basal area (m²/ha) | Mean diameter (cm) |
|---|---|---|---|
| Pine | 258 | 0.21 | 4.11 |
| Spruce | 949 | 0.84 | 4.97 |
| Deciduous | 1602 | 0.40 | 2.13 |
| **Stand** | **2809** | **1.45** | |

- Stem count: 258 + 949 + 1602 = **2809 stems/ha**.
- Mean diameter: (0.21 × 4.11 + 0.84 × 4.97 + 0.40 × 2.13) / 1.45 = **4.06 cm**.
- Main species: **spruce**, which has the largest basal area (0.84).

4.06 cm is below 8.0 cm, and 2809 stems/ha is above the spruce threshold of 2200, so the stand is
scaled. Its scaling factor is 1800 / 2809 = **0.641**:

| Species | Stems/ha | Basal area (m²/ha) |
|---|---|---|
| Pine | 165 | 0.13 |
| Spruce | 608 | 0.54 |
| Deciduous | 1027 | 0.26 |
| **Stand** | **1800** | **0.93** |

The first row of the stand's allometry file shows the difference. (The growth model works out
its own starting stem count and basal area from the strata it is given, so the row is close to
those figures, not equal to them.)

| `20.csv`, Year 0 | `N` (stems/ha) | `BA` (m²/ha) |
|---|---|---|
| Option off | 2781 | 1.8 |
| Option on | 1782 | 1.6 |

Stand 5 of the same project is denser still, with 3265 stems/ha, but its mean diameter is
8.98 cm. It is not a young stand by the rule, so it is not scaled.

## What the tool prints

With the option on, the Filtering section of the progress report lists every scaled stand:

```
Dense young stand scaling -- 1 stand(s) scaled down before the growth model runs:
  20: 2809 -> 1800 stems/ha (scaling factor 0.641)
```

With the option off, the tool warns about the stands the default numbers would scale, so you find
out about them without having to look:

```
Warning: 1 dense young stand(s): 20
  These are young stands with more stems than the default limits of dense young stand scaling, and they will be grown that way. [dense_young_stand_scaling] in the config file scales such stands down: https://susi-pack.github.io/susi_26/dense_young_stand_scaling/
```

The warning never stops the run, and a `--dry-run` shows it too.

It is always judged with the **default** numbers, on the stand as it is about to be grown. So if
you turn the option on with looser numbers of your own, a stand those numbers leave with more
stems than the defaults allow is still listed.

## What `stand_data.json` records

The figures recorded for a stand stay as the inventory reported them, scaled or not: its
`stem_count`, `basal_area`, `mean_diameter` and the per-species figures are never changed.
Two keys say what was done:

| Key | Meaning |
|---|---|
| `dense_young_stand_scaling_applied` | `true` when the stand was scaled before the growth model ran. |
| `dense_young_stand_scaling_factor` | The scaling factor, or `null` for a stand that was not scaled. |

For stand 20, with the option on:

```json
"stem_count": 2809.0,
"dense_young_stand_scaling_applied": true,
"dense_young_stand_scaling_factor": 0.6407974368102528
```

So a scaled stand's allometry file starts from fewer stems than its `stem_count`. The recorded
stem count times the scaling factor gives the stem count the growth model was given.
