from dataclasses import dataclass, field

from citsigol import Map


@dataclass
class BifurcationDiagram:
    map_class: type[Map]
    parameter_bounds: list[tuple[float, float]]
    initial_values: list[float] = field(default_factory=lambda: [0.5])
    steps_to_skip: int = 100
