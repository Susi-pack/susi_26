"""
Probability-sampling core for metsakeskus_to_allometry_stratified.py.

Deliberately free of susi imports and of any Metsakeskus specifics: every
function here takes plain numbers, so the design can be unit-tested and
replayed on a synthetic population without a GeoPackage. The tool module
adapts ParsedStand onto these.

The design, in one paragraph
---------------------------
Stands are stratified by (peat_type x fertilityclass x developmentclass). One
budget is allocated across ALL strata in proportion to stand AREA -- not a
separate budget per development class, which would decouple the sample's
composition from the region's. A stratum too thin to represent is EXCLUDED and
reported, rather than floored up to a minimum count. Within a stratum, stands
are drawn by Madow PPS-systematic selection from a frame sorted by basal area:
a random start, a fixed step through cumulative area, and "certainty" stands
(area >= step) taken with probability 1 and peeled off first.

Why this combination
--------------------
Area-proportional allocation makes the sample a picture of the region: each
stratum's share of the sample equals its share of the area, so every marginal
comparison (peat type, fertility class, development class) is both readable
raw and correctly composed. PPS within the stratum makes inclusion probability
proportional to area, so the unweighted sample estimates the area-weighted
population. Sorting by basal area is implicit stratification -- the systematic
sweep spreads picks across the basal-area range rather than clustering them,
which is what lets stand structure be used as a continuous modifier.

Two deliberate rejections:

  A per-development-class budget. Flat budgets per class decouple the sample
  from the landscape: a class holding 14% of the area but 50% of the budget
  makes every pooled statistic wrong unless weighted, for no gain.

  A minimum count per stratum. Flooring a stratum that holds 0.005% of the
  area up to three stands over-represents it several-hundred-fold and
  manufactures a category the region does not really have. A stratum too thin
  to sample proportionally is too thin to report; it is excluded, and the
  exclusion is printed and recorded.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# A stratum is identified by (peat_type, fertility class, development class).
# peat_type is a string so that a stand whose soiltype is missing can be
# placed in an explicit UNKNOWN_PEAT_TYPE stratum rather than being dropped.
StratumKey = tuple[str, int, int]

UNKNOWN_PEAT_TYPE = "unknown"


@dataclass(frozen=True)
class SamplingUnit:
    """One stand, reduced to what the design needs.

    unit_id    -- opaque, echoed back on the selection record
    area       -- hectares; the size measure for PPS. 0.0 when not recorded.
    basal_area -- the sort key for implicit stratification.
    stratum    -- (peat_type, fertilityclass, developmentclass)
    """

    unit_id: str
    area: float
    basal_area: float
    stratum: StratumKey


@dataclass(frozen=True)
class Selection:
    """One selected stand and its design quantities.

    analysis_weight is the number every pooled statistic multiplies by:
    hectares represented by this stand. It is area / pi, which within a
    stratum is constant for the swept stands and equal to the stand's own area
    for a certainty stand -- so "always weight" is a single rule that reduces
    to unweighted inside a stratum and corrects composition across strata.
    """

    unit_id: str
    stratum: StratumKey
    area: float
    inclusion_probability: float  # pi_i
    design_weight: float  # 1 / pi_i
    analysis_weight: float  # area_i / pi_i -- hectares represented
    certainty: bool  # True when taken with pi_i = 1


@dataclass(frozen=True)
class StratumReport:
    """What happened in one stratum -- printed, and written to the sidecar
    JSON, so a reader can see which strata were dropped and why, and check
    that the sample's composition matches the region's."""

    stratum: StratumKey
    n_available: int
    area_total: float
    area_share: float  # of the RETAINED area, i.e. of what was sampled
    n_allocated: int
    n_selected: int
    excluded: bool
    exclusion_reason: str | None


class SamplingConfigurationError(ValueError):
    """The configuration cannot produce a usable sample -- raised rather than
    silently degrading, because a sample nobody intended is worse than a run
    that stops."""


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------
def allocate_sample_sizes(
    units: list[SamplingUnit],
    budget_total: int,
    min_frame_stands_per_stratum: int = 200,
    min_stratum_area_share: float = 0.005,
) -> dict[StratumKey, StratumReport]:
    """One budget, allocated across all strata in proportion to stand area.

    A stratum is excluded when it has fewer than min_frame_stands_per_stratum
    stands in the frame, or holds less than min_stratum_area_share of the
    total area. Both tests are applied to the ORIGINAL totals, before
    renormalising, so excluding one stratum never cascades into excluding
    another.

    Shares are then computed over the retained area only, so the allocation
    sums to the budget rather than falling short by whatever the excluded
    strata would have taken.
    """
    if budget_total <= 0:
        raise SamplingConfigurationError(
            f"budget_total must be positive, got {budget_total}"
        )

    area_by_stratum: dict[StratumKey, float] = {}
    count_by_stratum: dict[StratumKey, int] = {}
    for unit in units:
        area_by_stratum[unit.stratum] = area_by_stratum.get(unit.stratum, 0.0) + max(
            unit.area, 0.0
        )
        count_by_stratum[unit.stratum] = count_by_stratum.get(unit.stratum, 0) + 1

    total_area = sum(area_by_stratum.values())
    if total_area <= 0:
        raise SamplingConfigurationError(
            "no stratum has any recorded area, so area-proportional allocation is "
            "undefined. Check that the stand layer's `area` column survived the "
            "filters."
        )

    excluded: dict[StratumKey, str] = {}
    for stratum, area in area_by_stratum.items():
        reasons = []
        if count_by_stratum[stratum] < min_frame_stands_per_stratum:
            reasons.append(
                f"only {count_by_stratum[stratum]} stands in frame "
                f"(< {min_frame_stands_per_stratum})"
            )
        if area / total_area < min_stratum_area_share:
            reasons.append(
                f"{area / total_area:.4%} of area (< {min_stratum_area_share:.2%})"
            )
        if reasons:
            excluded[stratum] = "; ".join(reasons)

    retained = [s for s in area_by_stratum if s not in excluded]
    if not retained:
        raise SamplingConfigurationError(
            f"every stratum was excluded (min_frame_stands_per_stratum="
            f"{min_frame_stands_per_stratum}, min_stratum_area_share="
            f"{min_stratum_area_share}). Loosen the thresholds."
        )
    retained_area = sum(area_by_stratum[s] for s in retained)

    reports: dict[StratumKey, StratumReport] = {}
    for stratum, area in area_by_stratum.items():
        if stratum in excluded:
            reports[stratum] = StratumReport(
                stratum=stratum,
                n_available=count_by_stratum[stratum],
                area_total=area,
                area_share=0.0,
                n_allocated=0,
                n_selected=0,
                excluded=True,
                exclusion_reason=excluded[stratum],
            )
            continue
        share = area / retained_area
        # At least 1: a stratum kept by the thresholds above is worth
        # representing, and rounding must not silently drop it.
        allocated = max(1, round(budget_total * share))
        reports[stratum] = StratumReport(
            stratum=stratum,
            n_available=count_by_stratum[stratum],
            area_total=area,
            area_share=share,
            n_allocated=allocated,
            n_selected=min(allocated, count_by_stratum[stratum]),
            excluded=False,
            exclusion_reason=None,
        )
    return reports


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def pps_systematic(
    units: list[SamplingUnit], n: int, rng: random.Random
) -> list[Selection]:
    """Madow PPS-systematic selection of n units, with size = area and the
    frame sorted by basal area.

    Three stages:
      1. Certainty units. A unit whose area is at least the sampling step
         cannot have pi < 1. Those are taken outright and removed, which
         raises the step for the rest, so the peel repeats until it finds
         none. Without this, one large stand can absorb several selection
         points and be "selected" more than once.
      2. Systematic sweep. ONE random start in [0, step) -- drawing a start
         per cut would make the cuts unevenly spaced and reintroduce exactly
         the duplicate selection stage 1 exists to prevent -- then one pick
         per step through the cumulative area.
      3. pi_i = k * a_i / A for the swept units, 1.0 for certainty units.

    With no area recorded anywhere in the stratum, sizes fall back to equal,
    which degenerates to systematic equal-probability sampling from a
    basal-area-sorted list: still a probability sample, just not
    area-proportional.
    """
    if n <= 0 or not units:
        return []

    def certainty(unit: SamplingUnit) -> Selection:
        return Selection(
            unit_id=unit.unit_id,
            stratum=unit.stratum,
            area=unit.area,
            inclusion_probability=1.0,
            design_weight=1.0,
            analysis_weight=max(unit.area, 0.0),
            certainty=True,
        )

    if n >= len(units):
        return [certainty(u) for u in units]

    sizes = {u.unit_id: max(u.area, 0.0) for u in units}
    if sum(sizes.values()) <= 0:
        sizes = {u.unit_id: 1.0 for u in units}

    # 1. peel certainty units
    pool = list(units)
    selections: list[Selection] = []
    remaining = n
    while pool and remaining > 0:
        step = sum(sizes[u.unit_id] for u in pool) / remaining
        big = [u for u in pool if sizes[u.unit_id] >= step]
        if not big:
            break
        selections += [certainty(u) for u in big]
        big_ids = {u.unit_id for u in big}
        pool = [u for u in pool if u.unit_id not in big_ids]
        remaining -= len(big)

    if remaining <= 0 or not pool:
        return selections
    if remaining >= len(pool):
        return selections + [certainty(u) for u in pool]

    # 2. systematic sweep over cumulative size, frame sorted by basal area
    ordered = sorted(pool, key=lambda u: (u.basal_area, u.unit_id))
    total = sum(sizes[u.unit_id] for u in ordered)
    step = total / remaining
    start = rng.uniform(0.0, step)
    cuts = [start + j * step for j in range(remaining)]

    cumulative = 0.0
    cut_index = 0
    for unit in ordered:
        cumulative += sizes[unit.unit_id]
        while cut_index < remaining and cuts[cut_index] <= cumulative:
            pi = remaining * sizes[unit.unit_id] / total
            selections.append(
                Selection(
                    unit_id=unit.unit_id,
                    stratum=unit.stratum,
                    area=unit.area,
                    inclusion_probability=pi,
                    design_weight=1.0 / pi,
                    analysis_weight=max(unit.area, 0.0) / pi,
                    certainty=False,
                )
            )
            cut_index += 1
    return selections


def select_stratified_sample(
    units: list[SamplingUnit],
    budget_total: int,
    min_frame_stands_per_stratum: int = 200,
    min_stratum_area_share: float = 0.005,
    seed: int = 19950101,
) -> tuple[list[Selection], dict[StratumKey, StratumReport]]:
    """Allocation then selection, stratum by stratum.

    The RNG is re-seeded per stratum from (seed, stratum) so that the draw in
    one stratum does not depend on how many stands happened to be in the
    strata processed before it -- adding a stand to one fertility class must
    not reshuffle another.
    """
    reports = allocate_sample_sizes(
        units, budget_total, min_frame_stands_per_stratum, min_stratum_area_share
    )

    grouped: dict[StratumKey, list[SamplingUnit]] = {}
    for unit in units:
        grouped.setdefault(unit.stratum, []).append(unit)

    selections: list[Selection] = []
    for stratum in sorted(grouped):
        report = reports[stratum]
        if report.excluded:
            continue
        rng = random.Random(f"{seed}|{stratum}")
        selections += pps_systematic(grouped[stratum], report.n_allocated, rng)
    return selections, reports


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def format_allocation_table(reports: dict[StratumKey, StratumReport]) -> str:
    """The allocation table, printed every run: each retained stratum's share
    of the area beside its share of the sample. Under area-proportional
    allocation those two columns should agree to within rounding -- that
    agreement IS the claim that the sample pictures the region, so it is
    printed rather than asserted."""
    retained = {k: r for k, r in reports.items() if not r.excluded}
    total_selected = sum(r.n_selected for r in retained.values()) or 1

    header = (
        f"{'peat':<8}{'FC':>4}{'DC':>4}{'frame':>8}{'area_ha':>12}"
        f"{'% area':>9}{'n':>5}{'% sample':>10}"
    )
    lines = [header, "-" * len(header)]
    for stratum in sorted(retained):
        r = retained[stratum]
        peat, fertility, development = stratum
        lines.append(
            f"{peat:<8}{fertility:>4}{development:>4}{r.n_available:>8,}"
            f"{r.area_total:>12,.1f}{r.area_share:>9.1%}{r.n_selected:>5}"
            f"{r.n_selected / total_selected:>10.1%}"
        )
    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL':<8}{'':>4}{'':>4}"
        f"{sum(r.n_available for r in retained.values()):>8,}"
        f"{sum(r.area_total for r in retained.values()):>12,.1f}"
        f"{1.0:>9.1%}{total_selected:>5}{1.0:>10.1%}"
    )

    dropped = {k: r for k, r in reports.items() if r.excluded}
    if dropped:
        lines.append("")
        lines.append("Excluded strata (too thin to represent, not floored up):")
        for stratum in sorted(dropped):
            r = dropped[stratum]
            lines.append(
                f"  {r.stratum[0]}/FC{r.stratum[1]}/DC{r.stratum[2]}: "
                f"{r.n_available:,} stands, {r.area_total:,.1f} ha -- {r.exclusion_reason}"
            )
        total_area = sum(r.area_total for r in reports.values())
        lines.append(
            f"  together {sum(r.area_total for r in dropped.values()):,.1f} ha = "
            f"{sum(r.area_total for r in dropped.values()) / total_area:.2%} of the region's area"
        )
    return "\n".join(lines)


def format_marginal_table(reports: dict[StratumKey, StratumReport]) -> str:
    """Sample share beside area share for each factor ON ITS OWN -- peat type,
    fertility class, development class. These are the comparisons the design
    is for, and this table is the evidence that each one is correctly
    composed."""
    retained = [r for r in reports.values() if not r.excluded]
    total_area = sum(r.area_total for r in retained) or 1.0
    total_n = sum(r.n_selected for r in retained) or 1

    lines = ["Marginal composition -- sample share vs area share:"]
    for label, index in (("peat type", 0), ("fertility class", 1), ("development class", 2)):
        by_level_area: dict[object, float] = {}
        by_level_n: dict[object, int] = {}
        for r in retained:
            level = r.stratum[index]
            by_level_area[level] = by_level_area.get(level, 0.0) + r.area_total
            by_level_n[level] = by_level_n.get(level, 0) + r.n_selected
        parts = [
            f"{level}: n={by_level_n[level]:<4} "
            f"({by_level_n[level] / total_n:>5.1%} vs {by_level_area[level] / total_area:>5.1%} area)"
            for level in sorted(by_level_area, key=str)
        ]
        lines.append(f"  {label:<18}" + "   ".join(parts))
    lines.append(
        "  A level whose n is too small to compare should be reported as not estimable, "
        "not rescued by reallocation -- its share of the sample is its share of the region."
    )
    return "\n".join(lines)
