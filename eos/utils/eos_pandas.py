import pandas as pd
import numpy as np
from typing import Dict, cast, Union
from datetime import datetime
from functools import reduce


def assemble_state_ser(motion_power: pd.DataFrame) -> pd.Series:
    """
    assemble state df from motion_power df
    order is vital for the model:
    "timestep, velocity, thrust, brake"
    contiguous storage in each measurement
    due to sort_index, output:
    [col0: brake, col1: thrust, col2: timestep, col3: velocity]
    """
    state: pd.Series = cast(
        pd.Series,
        (
            motion_power.loc[:, ['timestep', 'velocity', 'thrust', 'brake']]
            .stack()
            .swaplevel(0, 1)
        ),
    )
    state.name = 'state'
    state.index.names = ['rows', 'idx']
    state.sort_index(
        inplace=True
    )  # sort by rows and idx (brake, thrust, timestep, velocity)

    return state


def assemble_reward_ser(
    motion_power: pd.DataFrame, obs_sampling_rate: int
) -> pd.Series:
    """
    assemble reward df from motion_power df
    order is vital for the model:
    contiguous storage in each row, due to sort_index, output:
    [timestep, work]
    """

    pow_t = motion_power.loc[:, ['current', 'voltage']]
    ui_sum = pow_t.prod(axis=1).sum()
    wh = (
        ui_sum / 3600.0 / obs_sampling_rate
    )  # rate 0.05 for kvaser, 0.02 remote # negative wh
    # self.logc.info(
    #     f'wh: {wh}',
    #     extra=self.dictLogger,
    # )
    work = wh * (-1.0)
    reward_ts = pd.to_datetime(datetime.now())
    reward: pd.Series = cast(
        pd.Series,
        (
            pd.DataFrame({'work': work, 'timestep': reward_ts}, index=[0])
            .stack()
            .swaplevel(0, 1)
            .sort_index()  # columns oder (timestep, work)
        ),
    )
    reward.name = 'reward'
    reward.index.names = ['rows', 'idx']
    return reward


def assemble_action_ser(
    torque_map_line: np.ndarray,
    torque_table_row_names: list[str],
    table_start: int,
    flash_start_ts: pd.Timestamp,
    flash_end_ts: pd.Timestamp,
    torque_table_row_num_flash: int,
    torque_table_col_num: int,
    speed_scale: tuple,
    pedal_scale: tuple,
) -> pd.Series:
    """
    generate action df from torque_map_line
    order is vital for the model:
    contiguous storage in each row, due to sort_index, output:
    "r0, r1, r2, r3, ..., ,speed, throttle(map),timestep"
    """
    # assemble_action_df
    row_num = torque_table_row_num_flash
    speed_ser = pd.Series(
        speed_scale[table_start : table_start + torque_table_row_num_flash],
        name='speed',
    )
    throttle_ser = pd.Series(pedal_scale, name='throttle')
    torque_map = np.reshape(
        torque_map_line,
        [torque_table_row_num_flash, torque_table_col_num],
    )
    df_torque_map = pd.DataFrame(torque_map).transpose()  # row to columns
    df_torque_map.columns = pd.Index(torque_table_row_names)  # index: [r0, r1, ...]

    span_each_row = (flash_end_ts - flash_start_ts) / row_num
    flash_timestamps_ser = pd.Series(
        pd.to_datetime(
            flash_start_ts + np.linspace(1, row_num, row_num) * span_each_row
        ),
        name='timestep',
    )

    dfs: list[Union[pd.DataFrame, pd.Series]] = [
        df_torque_map,
        flash_timestamps_ser,
        speed_ser,
        throttle_ser,
    ]
    action_df: pd.DataFrame = cast(
        pd.DataFrame,
        reduce(
            lambda left, right: pd.merge(
                left,
                right,
                how='outer',
                left_index=True,
                right_index=True,
            ),
            dfs,
        ),
    )

    action = cast(
        pd.Series, (action_df.stack().swaplevel(0, 1).sort_index())
    )  # columns order (r0, r1, ..., speed, throttle, timestep)
    action.name = 'action'
    action.index.names = ['rows', 'idx']
    action.columns.names = []

    return action


def nest(d: dict) -> dict:
    """
    Convert a flat dictionary with tuple key to a nested dictionary through to the leaves
    arrays will be converted to dictionaries with the index as the key
    no conversion of pd.Timestamp
    """
    result: Dict = {}
    for key, value in d.items():
        target = result
        for k in key[:-1]:
            target = target.setdefault(k, {})
        target[key[-1]] = value
    return result


def df_to_nested_dict(df_multi_indexed_col: pd.DataFrame) -> dict:
    """
    Convert a dataframe with multi-indexed columns to a nested dictionary
    """
    d = df_multi_indexed_col.to_dict(
        'index'
    )  # for multi-indexed dataframe, the index in the first level of the dictionary is still a tuple!
    return {k: nest(v) for k, v in d.items()}


def eos_df_to_nested_dict(episode: pd.DataFrame) -> dict:
    """
    Convert an eos dataframe with multi-indexed columns to a nested dictionary
    Remove all the levels of the multi-indexed columns except for 'timestamp'
    Keep only the timestamp as the single key for the nested dictionary
    """
    dict_nested = df_to_nested_dict(
        episode
    )  # for multi-indexed dataframe, the index in the first level of the dictionary is still a tuple!
    indices_dict = [
        {episode.index.names[i]: level for i, level in enumerate(levels)}
        for levels in episode.index
    ]  # all elements in the array should have the same vehicle, driver, episodestart
    single_key_dict = {
        idx['timestamp']: dict_nested[key]
        for idx, key in zip(indices_dict, dict_nested)
    }

    return single_key_dict


def ep_nest(d: Dict) -> Dict:
    """
    Convert a flat dictionary with tuple key to a nested dictionary with arrays at the leaves
    convert pd.Timestamp to millisecond long integer
    """
    result: Dict = {}
    for key, value in d.items():
        target = result
        for k in key[:-2]:
            target = target.setdefault(k, {})
        if key[-2] not in target:
            target[key[-2]] = []

        if isinstance(value, pd.Timestamp):
            value = value.timestamp() * 1e6  # convert to microsecond long integer
        target[key[-2]].append(value)

    return result


def df_to_ep_nested_dict(df_multi_indexed_col: pd.DataFrame) -> dict:
    """
    Convert a dataframe with multi-indexed columns to a nested dictionary
    """
    d = df_multi_indexed_col.to_dict(
        'index'
    )  # for multi-indexed dataframe, the index in the first level of the dictionary is still a tuple!
    return {k: ep_nest(v) for k, v in d.items()}


def avro_ep_encoding(episode: pd.DataFrame) -> list[Dict]:
    """
    avro encoding,
    parsing requires a schema defined in "data_io/pool/episode_avro_schema.py"

    Convert an eos dataframe with multi-indexed columns to a nested dictionary
    Remove all the levels of the multi-indexed columns except for 'timestamp'
    Keep only the timestamp as the single key for the nested dictionary
    ! Convert Timestamp to millisecond long integer!! for compliance to the  avro storage format
    as flat as possible
    PEP20: flat is better than nested!
    """
    dict_nested = df_to_ep_nested_dict(
        episode
    )  # for multi-indexed dataframe, the index in the first level of the dictionary is still a tuple!
    indices_dict = [
        {episode.index.names[i]: level for i, level in enumerate(levels)}
        for levels in episode.index
    ]  # all elements in the array should have the same vehicle, driver, episodestart
    array_of_dict = [
        {
            'timestamp': idx['timestamp'].timestamp()
            * 1e6,  # convert to microsecond long integer
            **dict_nested[
                key
            ],  # merge the nested dict with the timestamp, as flat as possible
        }
        for (idx, key) in zip(indices_dict, dict_nested)
    ]

    return array_of_dict


def avro_ep_decoding(episodes: list[Dict]) -> list[pd.DataFrame]:
    """
    avro decoding,

    Convert a list of nested dictionaries to DataFrame with multi-indexed columns and index
    ! Convert microsecond long integer to Timestamp!
    (avro storage format stores timestamp as long integer in keys but
    seem to have DateTime with timezone in the values.)
    """

    df_episodes_list = []
    for ep in episodes:
        dict_observations = [
            {
                (
                    ep['meta']['episode_meta']['vehicle'],
                    ep['meta']['episode_meta']['driver'],
                    pd.to_datetime(
                        ep['meta']['episode_meta']['episodestart'], unit='us'
                    ),
                    pd.to_datetime(step['timestamp'], unit='us'),
                    qtuple,
                    rows,
                    idx,
                ): item
                for qtuple, obs in step.items()
                if qtuple != 'timestamp'
                for rows, value in obs.items()
                for idx, item in enumerate(value)
            }
            for step in ep['sequence']
        ]

        dict_ep = {k: v for d in dict_observations for k, v in d.items()}

        ser_decoded = pd.Series(dict_ep)
        ser_decoded.index.names = [
            'vehicle',
            'driver',
            'episodestart',
            'timestamp',
            'qtuple',
            'rows',
            'idx',
        ]
        df_decoded = ser_decoded.unstack(level=['qtuple', 'rows', 'idx'])  # type: ignore
        df_episodes_list.append(df_decoded)

    return df_episodes_list


def decode_mongo_documents(
    df: pd.DataFrame,
    torque_table_row_names: list[str],
) -> tuple[
    list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]
]:
    """
    decoding the batch RECORD observations from mongodb nested dicts to pandas dataframe
    (EPISODE doesn't need decoding, it is already a dataframe)
    TODO need to check whether sort_index is necessary
    """

    dict_observations_list = (
        [  # list of observations as dict with tuple key suitable as MultiIndex
            {
                (
                    meta['episodestart'],
                    meta['vehicle'],
                    meta['driver'],
                    meta['timestamp'],
                    qtuple,
                    rows,
                    idx,
                ): value
                for qtuple, obs1 in obs.items()
                for rows, obs2 in obs1.items()
                for idx, value in obs2.items()
            }
            for meta, obs in zip(df['meta'], df['observation'])
        ]
    )

    df_actions = []
    df_states = []
    df_nstates = []
    ser_rewards = []
    idx = pd.IndexSlice
    for dict_observations in dict_observations_list:  # decode each measurement from
        ser_decoded = pd.Series(dict_observations)
        ser_decoded.index.names = [
            'episodestart',
            'vehicle',
            'driver',
            'timestamp',
            'qtuple',
            'rows',
            'idx',
        ]

        # decode state
        ser_state = ser_decoded.loc[
            idx[:, :, :, :, 'state', ['brake', 'thrust', 'velocity', 'timestep']]
        ]
        df_state = ser_state.unstack(level=[0, 1, 2, 3, 4, 5])  # type: ignore
        multiindex = df_state.columns
        df_state.set_index(multiindex[-1], inplace=True)  # last index has timestep
        df_states.append(df_state)

        # decode action
        ser_action = ser_decoded.loc[
            idx[:, :, :, :, 'action', [*torque_table_row_names, 'throttle']]
        ]
        df_action = ser_action.unstack(level=[0, 1, 2, 3, 4, 5])  # type: ignore
        multiindex = df_action.columns
        df_action.set_index(multiindex[-1], inplace=True)  # last index has throttle

        action_timestep = ser_decoded.loc[idx[:, :, :, :, 'action', 'timestep']]
        action_speed = ser_decoded.loc[idx[:, :, :, :, 'action', 'speed']]
        action_multi_col = [
            (*column, speed, timestep)  # swap speed and timestep
            for column, timestep, speed in zip(
                df_action.columns, action_timestep, action_speed
            )
        ]
        df_action.columns = pd.MultiIndex.from_tuples(
            action_multi_col,
            names=[
                'episodestart',
                'vehicle',
                'driver',
                'timestamp',
                'qtuple',
                'rows',
                'speed',
                'timestep',
            ],
        )
        df_actions.append(df_action)

        # decode reward
        ser_reward = ser_decoded.loc[idx[:, :, :, :, 'reward', ['work', 'timestep']]]
        df_reward = ser_reward.unstack([0, 1, 2, 3, 4, 5])  # type: ignore
        multiindex = df_reward.columns
        df_reward.set_index(multiindex[-1], inplace=True)  # last index has timestep
        # df_reward
        ser_rewards.append(df_reward)

        # decode nstate
        ser_nstate = ser_decoded.loc[
            idx[:, :, :, :, 'nstate', ['brake', 'thrust', 'velocity', 'timestep']]
        ]
        df_nstate = ser_nstate.unstack([0, 1, 2, 3, 4, 5])  # type: ignore
        multiindex = df_nstate.columns
        df_nstate.set_index(multiindex[-1], inplace=True)
        df_nstates.append(df_nstate)

    return df_states, df_actions, ser_rewards, df_nstates


def decode_dataframe_from_parquet(df: pd.DataFrame):
    """
    decode the dataframe from parquet with flat column indices to MultiIndexed DataFrame
    """

    multi_tpl = [tuple(col.split('_')) for col in df.columns]
    multi_col = pd.MultiIndex.from_tuples(multi_tpl)
    i1 = multi_col.get_level_values(0)
    i1 = pd.Index(
        ['' if str(idx) in (str(pd.NA), 'nan', '') else idx for idx in i1]
    )  # convert index of level 2 type to int and '' if NA
    i2 = multi_col.get_level_values(
        1
    )  # must be null string instead of the default pd.NA or np.nan
    i2 = pd.Index(
        ['' if str(idx) in (str(pd.NA), 'nan', '') else idx for idx in i2]
    )  # convert index of level 2 type to int and '' if NA
    i3 = multi_col.get_level_values(
        2
    )  # must be null string instead of the default pd.NA or np.nan
    i3 = pd.Index(
        ['' if str(idx) in (str(pd.NA), 'nan', '') else int(idx) for idx in i3]
    )  # convert index of level 2 type to int and '' if NA

    multi_col = pd.MultiIndex.from_arrays([i1, i2, i3])
    multi_col.names = ['qtuple', 'rows', 'idx']
    df.columns = multi_col

    df = df.set_index(['vehicle', 'driver', 'episodestart', df.index])  # type: ignore

    return df
