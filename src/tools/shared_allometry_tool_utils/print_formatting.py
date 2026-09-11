"""Console progress-report formatting shared by xml_to_allometry.py and
metsakeskus_to_allometry.py: the sectioned Reading/Filtering/Writing
headings, and the StandSkipped type + printer for reporting which stands
were dropped and why instead of silently vanishing.
"""

from dataclasses import dataclass

from susi.io.load_output_data import StandID


@dataclass(frozen=True)
class StandSkipped:
    """A stand that does not survive some filter or check, so no allometry
    file gets written for it."""

    stand_id: StandID
    reason: str


def print_section(title: str) -> None:
    """Marks one phase of main()'s reading -> filtering -> writing pipeline
    in the console output, so the phases are visually separated."""
    print()
    print(title)
    print("-" * len(title))


def print_skips(skips: list[StandSkipped], label: str) -> None:
    if not skips:
        return
    print(f"{label}: {len(skips)}")
    for skip in skips:
        print(f"  {skip.stand_id}: {skip.reason}")
    print()
