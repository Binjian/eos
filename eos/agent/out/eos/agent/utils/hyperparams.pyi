from typing import Dict, NamedTuple

from _typeshed import Incomplete

class HyperParam(NamedTuple):
    ModelName: Incomplete
    BatchSize: Incomplete
    NStates: Incomplete
    NActions: Incomplete
    HiddenUnitsState0: Incomplete
    HiddenUnitsState1: Incomplete
    HiddenUnitsAction: Incomplete
    HiddenUnitsOut: Incomplete
    ActionBias: Incomplete
    NLayerActor: Incomplete
    NLayerCritic: Incomplete
    Gamma: Incomplete
    TauActor: Incomplete
    TauCritic: Incomplete
    ActorLR: Incomplete
    CriticLR: Incomplete
    CkptInterval: Incomplete
    PaddingValue: Incomplete

hyper_param_list: Incomplete
hyper_param_by_name: Dict[str, HyperParam]
