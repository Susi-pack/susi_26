"""Tests for the sampling design.

These are about the DESIGN's properties, not about Metsakeskus data, which is
why sampling_design.py has no susi imports: the whole design runs on a
synthetic population in a couple of seconds.

Run: pytest test_sampling_design.py -q
"""

import math
import random
import statistics

import pytest

from tools.shared_allometry_tool_utils.sampling_design import (
    SamplingConfigurationError,
    SamplingUnit,
    allocate_sample_sizes,
    format_marginal_table,
    pps_systematic,
    select_stratified_sample,
)

BUDGET = 180


def synthetic_population(n: int = 3000, seed: int = 42) -> list[SamplingUnit]:
    rng = random.Random(seed)
    return [
        SamplingUnit(
            unit_id=f"u{i}",
            area=rng.lognormvariate(0.4, 1.0),
            basal_area=rng.lognormvariate(3.0, 0.5),
            stratum=(
                rng.choice(["A", "A", "A", "S"]),
                rng.choice([2, 3, 3, 4, 4, 4, 5]),
                rng.choice([2, 2, 3]),
            ),
        )
        for i in range(n)
    ]


# --- selection mechanics ---------------------------------------------------
def test_no_unit_is_ever_selected_twice():
    """The failure the certainty-unit peel exists to prevent. A single random
    start per sweep is what keeps the cuts exactly one step apart; drawing one
    per cut lets a unit absorb two of them."""
    population = synthetic_population()
    for seed in range(200):
        selections, _ = select_stratified_sample(population, BUDGET, seed=seed)
        assert len({s.unit_id for s in selections}) == len(selections)


def test_selection_is_reproducible_from_the_seed():
    population = synthetic_population()
    first, _ = select_stratified_sample(population, BUDGET, seed=5)
    second, _ = select_stratified_sample(population, BUDGET, seed=5)
    assert [s.unit_id for s in first] == [s.unit_id for s in second]


def test_a_stratum_draw_does_not_depend_on_other_strata():
    population = synthetic_population()
    target = ("A", 4, 3)
    before = {
        s.unit_id
        for s in select_stratified_sample(population, BUDGET, seed=1)[0]
        if s.stratum == target
    }
    augmented = population + [SamplingUnit("extra", 5.0, 20.0, ("S", 2, 2))]
    after = {
        s.unit_id
        for s in select_stratified_sample(augmented, BUDGET, seed=1)[0]
        if s.stratum == target
    }
    assert before == after


def test_a_dominant_stand_does_not_swallow_the_stratum():
    """The old quantile selector returned the giant plus the five SMALLEST
    stands here, because it back-filled from the start of the sorted frame."""
    units = [SamplingUnit(f"p{i}", 1.0, float(i), ("A", 3, 2)) for i in range(1, 21)]
    units[14] = SamplingUnit("p15", 200.0, 15.0, ("A", 3, 2))

    selections = pps_systematic(units, 6, random.Random(0))
    basal_areas = {u.unit_id: u.basal_area for u in units}
    picked = sorted(basal_areas[s.unit_id] for s in selections)

    assert len(selections) == 6
    assert sum(s.certainty for s in selections) == 1  # the 200 ha stand
    assert picked[3] > 5  # NOT five smallest + the giant


def test_small_stratum_is_taken_whole():
    units = [SamplingUnit(f"s{i}", 1.0, float(i), ("S", 5, 2)) for i in range(2)]
    selections = pps_systematic(units, 3, random.Random(0))
    assert len(selections) == 2
    assert all(s.certainty and s.design_weight == 1.0 for s in selections)


# --- the two properties the design exists for ------------------------------
def test_sample_composition_matches_area_composition():
    """Area-proportional allocation means every marginal split's share of the
    sample equals its share of the area. This is the 'picture of the region'
    claim, tested rather than asserted."""
    population = synthetic_population()
    _, reports = select_stratified_sample(population, BUDGET, seed=1)
    retained = [r for r in reports.values() if not r.excluded]
    total_area = sum(r.area_total for r in retained)
    total_n = sum(r.n_selected for r in retained)

    for index in (0, 1, 2):  # peat type, fertility class, development class
        area_by, n_by = {}, {}
        for r in retained:
            level = r.stratum[index]
            area_by[level] = area_by.get(level, 0.0) + r.area_total
            n_by[level] = n_by.get(level, 0) + r.n_selected
        for level, area in area_by.items():
            assert abs(n_by[level] / total_n - area / total_area) < 0.02, (
                index,
                level,
            )


def test_unweighted_sample_estimates_the_area_weighted_mean():
    """PPS within a stratum: pi proportional to area makes the UNWEIGHTED
    sample mean an estimator of the AREA-weighted population mean."""
    population = synthetic_population()
    group = [u for u in population if u.stratum == ("A", 4, 3)]
    rng = random.Random(0)
    y = {u.unit_id: 2.0 * u.basal_area + rng.gauss(0, 3) for u in group}
    truth = sum(y[u.unit_id] * u.area for u in group) / sum(u.area for u in group)

    pps_means, srs_means = [], []
    for trial in range(2000):
        pps = pps_systematic(group, 26, random.Random(trial))
        pps_means.append(statistics.mean(y[s.unit_id] for s in pps))
        srs = random.Random(10_000 + trial).sample(group, 26)
        srs_means.append(statistics.mean(y[u.unit_id] for u in srs))

    pps_bias = abs(statistics.mean(pps_means) - truth)
    srs_bias = abs(statistics.mean(srs_means) - truth)
    standard_error = statistics.stdev(pps_means) / math.sqrt(len(pps_means))

    assert pps_bias < srs_bias / 4
    assert pps_bias < 5 * standard_error


def test_analysis_weight_is_constant_within_a_stratum():
    """Why 'always weight' needs no switch: inside a stratum the weights are
    equal, so a weighted statistic equals the unweighted one; across strata
    they correct composition. Certainty stands are the documented exception."""
    population = synthetic_population()
    selections, _ = select_stratified_sample(population, BUDGET, seed=3)
    by_stratum: dict[tuple, list] = {}
    for s in selections:
        if not s.certainty:
            by_stratum.setdefault(s.stratum, []).append(s.analysis_weight)
    checked = 0
    for weights in by_stratum.values():
        if len(weights) < 2:
            continue
        assert max(weights) - min(weights) < 1e-6 * max(weights)
        checked += 1
    assert checked >= 3


def test_the_pooled_sample_estimates_the_region_weighted_or_not():
    """The payoff of area-proportional allocation: the design is
    SELF-WEIGHTING, so the raw pooled mean and the weighted pooled mean both
    estimate the area-weighted mean of the sampled region, and agree with each
    other. The weights are carried for provenance and for the cases where
    proportionality breaks (a stratum clamped by n_available, certainty
    stands), not because the headline numbers need rescuing.

    The target is the retained area, not the whole frame: excluded strata are
    deliberately outside the population this sample describes, which is why
    they are reported.
    """
    population = synthetic_population()
    rng = random.Random(1)
    y = {u.unit_id: 2.0 * u.basal_area + rng.gauss(0, 3) for u in population}

    _, reports = select_stratified_sample(population, BUDGET, seed=0)
    retained = {k for k, r in reports.items() if not r.excluded}
    in_scope = [u for u in population if u.stratum in retained]
    truth = sum(y[u.unit_id] * u.area for u in in_scope) / sum(
        u.area for u in in_scope
    )

    weighted, raw = [], []
    for seed in range(300):
        selections, _ = select_stratified_sample(population, BUDGET, seed=seed)
        w = [s.analysis_weight for s in selections]
        v = [y[s.unit_id] for s in selections]
        weighted.append(sum(a * b for a, b in zip(w, v)) / sum(w))
        raw.append(statistics.mean(v))

    assert abs(statistics.mean(weighted) - truth) < 0.02 * truth
    assert abs(statistics.mean(raw) - truth) < 0.02 * truth
    # self-weighting: the two agree, which is what "picture of the region" means
    assert abs(statistics.mean(weighted) - statistics.mean(raw)) < 0.01 * truth


# --- exclusion, not flooring -----------------------------------------------
def test_thin_stratum_is_excluded_and_reported_not_floored():
    population = synthetic_population()
    population += [SamplingUnit(f"tiny{i}", 0.01, 5.0, ("S", 2, 3)) for i in range(4)]
    reports = allocate_sample_sizes(
        population, BUDGET, min_frame_stands_per_stratum=200
    )
    tiny = reports[("S", 2, 3)]
    assert tiny.excluded
    assert tiny.n_selected == 0
    assert tiny.exclusion_reason is not None
    assert "in frame" in tiny.exclusion_reason


def test_excluded_strata_are_absent_from_the_sample():
    population = synthetic_population()
    population += [SamplingUnit(f"tiny{i}", 0.01, 5.0, ("S", 2, 3)) for i in range(4)]
    selections, reports = select_stratified_sample(population, BUDGET, seed=1)
    dropped = {k for k, r in reports.items() if r.excluded}
    assert dropped
    assert not any(s.stratum in dropped for s in selections)


def test_retained_shares_renormalise_to_the_full_budget():
    """Excluding a stratum must not leave the sample short by that stratum's
    would-be allocation."""
    population = synthetic_population()
    population += [SamplingUnit(f"tiny{i}", 0.01, 5.0, ("S", 2, 3)) for i in range(4)]
    reports = allocate_sample_sizes(population, BUDGET)
    retained = [r for r in reports.values() if not r.excluded]
    assert abs(sum(r.area_share for r in retained) - 1.0) < 1e-9
    assert abs(sum(r.n_allocated for r in retained) - BUDGET) <= len(retained)


def test_no_area_anywhere_raises_rather_than_degrading():
    units = [SamplingUnit(f"n{i}", 0.0, float(i % 17), ("A", 3, 2)) for i in range(400)]
    with pytest.raises(SamplingConfigurationError, match="no stratum has any recorded area"):
        allocate_sample_sizes(units, BUDGET)


def test_everything_excluded_raises():
    population = synthetic_population()
    with pytest.raises(SamplingConfigurationError, match="every stratum was excluded"):
        allocate_sample_sizes(population, BUDGET, min_frame_stands_per_stratum=10_000)


def test_marginal_table_mentions_every_factor():
    population = synthetic_population()
    _, reports = select_stratified_sample(population, BUDGET, seed=1)
    rendered = format_marginal_table(reports)
    for label in ("peat type", "fertility class", "development class"):
        assert label in rendered
