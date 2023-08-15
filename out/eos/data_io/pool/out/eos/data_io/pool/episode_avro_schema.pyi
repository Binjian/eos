from _typeshed import Incomplete as Incomplete

from eos.data_io.struct.eos_data import ObservationMeta as ObservationMeta

state_unit_fields_schema: Incomplete
state_specs_fields_schema: Incomplete
action_specs_fields_schema: Incomplete
reward_specs_fields_schema: Incomplete
episode_meta_fields_schema: Incomplete
observation_meta_fields_schema: Incomplete
state_fields_schema: Incomplete

def gen_torque_table_schema(obs_meta: ObservationMeta): ...

action_fields_schema: Incomplete
reward_fields_schema: Incomplete

def gen_episode_array_fields_schema(obs_meta: ObservationMeta): ...
def gen_episode_schema(obs_meta: ObservationMeta) -> dict: ...
