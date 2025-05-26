"""CPG brains for modular robots."""

from ._brain_cpg_instance import BrainCpgInstance
from ._brain_cpg_instance_locomotion import BrainCpgInstanceLocomotion
from ._brain_cpg_instance_locomotion_newstate import BrainCpgInstanceLocomotionNewstate
from ._brain_cpg_network_neighbor import BrainCpgNetworkNeighbor
from ._brain_cpg_network_neighbor_random import BrainCpgNetworkNeighborRandom
from ._brain_cpg_network_static import BrainCpgNetworkStatic
from ._brain_cpg_network_locomotion import BrainCpgNetworkLocomotion
from ._brain_cpg_network_locomotion_newstate import BrainCpgNetworkLocomotionNewstate

from ._cpg_network_structure import CpgNetworkStructure
from ._make_cpg_network_structure_neighbor import (
    active_hinges_to_cpg_network_structure_neighbor,
)

__all__ = [
    "BrainCpgInstance",
    "BrainCpgInstanceLocomotion",
    "BrainCpgInstanceLocomotionNewstate",
    "BrainCpgNetworkNeighbor",
    "BrainCpgNetworkNeighborRandom",
    "BrainCpgNetworkStatic",
    "BrainCpgNetworkLocomotion",
    "BrainCpgNetworkLocomotionNewstate",
    "CpgNetworkStructure",
    "active_hinges_to_cpg_network_structure_neighbor",
]
