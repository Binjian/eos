from __future__ import annotations
from collections import namedtuple
from typing import Dict

HYPER_PARAM = namedtuple(
    "HYPER_PARAM",
    [
        "ModelName",  # name of the model
        "BatchSize",  # batch size for training
        "NStates",  # number of states in the state space
        "NActions",  # number of actions in the action space
        "HiddenUnitsState0",  # hidden unit number for the first layer of the state network
        "HiddenUnitsState1",  # hidden unit number for the second layer of the state network
        "HiddenUnitsAction",  # hidden unit number for the action input layer
        "HiddenUnitsOut",  # hidden unit number for the output layer
        "ActionBias",  # bias for action output`
        "NLayerActor",  # number of layers for the actor network
        "NLayerCritic",  # number of layers for the critic network
        "Gamma",  # Gamma value for RL discount
        "TauActor",  # Tau value for Polyak averaging for the actor network
        "TauCritic",  # Tau value for Polyak averaging for the actor network
        "ActorLR",  # learning rate for the actor network
        "CriticLR",  # learning rate for the critic network
        "CkptInterval",  # checkpoint interval
        "PaddingValue",  # padding value for the input, impossible value for observation, action or reward
    ],
)


hyper_param_list = [
    HYPER_PARAM(
        ModelName="DEFAULT",  # name of the model
        BatchSize=4,  # batch size for training
        NStates=600,  # number of states in the state space
        NActions=68,  # number of actions in the action space
        HiddenUnitsState0=16,  # hidden unit number for the first layer of the state network
        HiddenUnitsState1=32,  # hidden unit number for the second layer of the state network
        HiddenUnitsAction=256,  # hidden unit number for the action input layer
        HiddenUnitsOut=256,  # hidden unit number for the output layer
        ActionBias=0.0,  # bias for action output`
        NLayerActor=2,  # number of layers for the actor network
        NLayerCritic=2,  # number of layers for the critic network
        Gamma=0.99,  # Gamma value for RL discount
        TauActor=0.005,  # Tau value for Polyak averaging for the actor network
        TauCritic=0.005,  # Tau value for Polyak averaging for the actor network
        ActorLR=0.001,  # learning rate for the actor network
        CriticLR=0.002,  # learning rate for the critic network
        CkptInterval=5,  # checkpoint interval
        PaddingValue=0,  # padding value for the input, impossible value for observation, action or reward
    ),
    HYPER_PARAM(
        ModelName="DDPG",  # name of the model
        BatchSize=4,  # batch size for training
        NStates=600,  # number of states in the state space
        NActions=68,  # number of actions in the action space
        HiddenUnitsState0=16,  # hidden unit number for the first layer of the state network
        HiddenUnitsState1=32,  # hidden unit number for the second layer of the state network
        HiddenUnitsAction=256,  # hidden unit number for the action input layer
        HiddenUnitsOut=256,  # hidden unit number for the output layer
        ActionBias=0,  # bias for action output`
        NLayerActor=2,  # number of layers for the actor network
        NLayerCritic=2,  # number of layers for the critic network
        Gamma=0.99,  # Gamma value for RL discount factor
        TauActor=0.005,  # Tau value for Polyak averaging for the actor network
        TauCritic=0.005,  # Tau value for Polyak averaging for the actor network
        ActorLR=0.001,  # learning rate for the actor network
        CriticLR=0.002,  # learning rate for the critic network
        CkptInterval=5,  # checkpoint interval
        PaddingValue=0,  # padding value for the input, impossible value for observation, action or reward
    ),
    HYPER_PARAM(
        ModelName="RDPG",  # name of the model
        BatchSize=4,  # batch size for training
        NStates=600,  # number of states in the state space
        NActions=68,  # number of actions in the action space
        HiddenUnitsState0=16,  # hidden unit number for the first layer of the state network
        HiddenUnitsState1=32,  # hidden unit number for the second layer of the state network
        HiddenUnitsAction=256,  # hidden unit number for the action input layer
        HiddenUnitsOut=256,  # hidden unit number for the output layer
        ActionBias=0,  # bias for action output`
        NLayerActor=2,  # number of layers for the actor network
        NLayerCritic=2,  # number of layers for the critic network
        Gamma=0.99,  # Gamma value for RL discount
        TauActor=0.005,  # Tau value for Polyak averaging for the actor network
        TauCritic=0.005,  # Tau value for Polyak averaging for the actor network
        ActorLR=0.001,  # learning rate for the actor network
        CriticLR=0.002,  # learning rate for the critic network
        CkptInterval=5,  # checkpoint interval
        PaddingValue=-10000,  # padding value for the input, impossible value for observation, action or reward
    ),
]

hyper_param_by_name: Dict[str, HYPER_PARAM] = {
    hyper_param.ModelName: hyper_param for hyper_param in hyper_param_list
}
