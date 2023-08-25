from datetime import datetime
from functools import reduce
from typing import Dict, List, Union, cast

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


def assemble_state_ser(state_columns: pd.DataFrame) -> pd.Series:
    """
    assemble state df from state_columns dataframe
    order is vital for the model:
    "timestep, velocity, thrust, brake"
    contiguous storage in each measurement
    due to sort_index, output:
    [col0: brake, col1: thrust, col2: timestep, col3: velocity]
    """
    state: pd.Series = cast(
        pd.Series,
        (state_columns.stack().swaplevel(0, 1)),
    )
    state.name = 'state'
    state.index.names = ['rows', 'idx']
    state.sort_index(
        inplace=True
    )  # sort by rows and idx (brake, thrust, timestep, velocity)

    return state


def assemble_reward_ser(
    power_columns: pd.DataFrame, obs_sampling_rate: int
) -> pd.Series:
    """
    assemble reward df from motion_power df
    order is vital for the model:
    contiguous storage in each row, due to sort_index, output:
    power_columns: ['current', 'voltage']
    [timestep, work]
    """

    ui_sum = power_columns.prod(axis=1).sum()
    wh = (
        ui_sum / 3600.0 / obs_sampling_rate
    )  # rate 0.05 for kvaser, 0.02 remote # negative wh
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
    "r0, r1, r2, r3, ..., speed, throttle(map),timestep"
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
    # action.column.names = []

    return action


def nest(d: dict) -> dict:
    """
    Convert a flat dictionary with tuple key to a nested dictionary through to the leaves
    arrays will be converted to dictionaries with the index as the key
    no conversion of pd.Timestamp
    only for use in mongo records
    """
    result: Dict = {}
    for key, value in d.items():
        target = result
        for k in key[:-1]:
            target = target.setdefault(k, {})
        target[str(key[-1])] = value   # for mongo only string keys are allowed.
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


def decode_mongo_records(
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
                df_action.columns, action_timestep, action_speed  # type: ignore
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


def decode_mongo_episodes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    decoding the batch RECORD observations from mongodb nested dicts to pandas dataframe
    (EPISODE doesn't need decoding, it is already a dataframe)
    TODO need to check whether sort_index is necessary"""

    dict_observations = {
        (
            meta['vehicle'],
            meta['driver'],
            meta['episodestart'],
            timestamp,
            qtuple,
            rows,
            idx,
        ): value
        for meta, obs in zip(df['meta'], df['observation'])
        for timestamp, obs1 in obs.items()
        for qtuple, obs2 in obs1.items()  # (state, action, reward, next_state)
        for rows, obs3 in obs2.items()  # (velocity, thrust, brake), (r0, r1, r2, ...),
        for idx, value in obs3.items()  # (0, 1, 2, ...)
    }
    ser_decoded = pd.Series(dict_observations)
    ser_decoded.index.names = [
        'vehicle',
        'driver',
        'episodestart',
        'timestamp',
        'qtuple',
        'rows',
        'idx',
    ]
    batch = ser_decoded.unstack(level=['qtuple', 'rows', 'idx'])  # type: ignore

    return batch


def encode_dataframe_from_parquet(df: pd.DataFrame):
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


def decode_episode_dataframes_to_padded_arrays(
    batch: pd.DataFrame, padding_value: float = -10000.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    decode the dataframes to 3D numpy arrays [B, T, F] for states, actions, rewards, next_states
    episodes with variable lengths will turn into ragged arrays with the same raggedness, thus the same maximum length
    after padding the arrays will have the same shape and padding pattern.
    """

    episodestart_index = batch.index.unique(level='episode_start')
    batch.sort_index(inplace=False, axis=0).sort_index(inplace=True, axis=1)

    # array of rewards for minibatch
    idx = pd.IndexSlice
    df_rewards = batch.loc[:, idx['reward', 'work']]  # type: ignore
    rewards_list = [
        df_rewards.loc[idx[:, :, ep_start, :]].values.tolist()  # type: ignore
        for ep_start in episodestart_index
    ]
    r_n_t = keras.utils.pad_sequences(
        rewards_list, padding='post', dtype=np.float32, value=padding_value
    )

    # array of states for minibatch
    df_states = batch.loc[
        :, idx['state', ['velocity', 'thrust', 'brake']]  # type: ignore
    ]  # same order as inference !!!
    states_list = [
        df_states.loc[idx[:, :, ep_start, :]].values.tolist()  # type: ignore
        for ep_start in episodestart_index
    ]
    s_n_t = keras.utils.pad_sequences(
        states_list, padding='post', dtype=np.float32, value=padding_value
    )

    # array of actions for minibatch
    df_actions = batch.loc[:, idx['action', self.torque_table_row_names]]  # type: ignore
    actions_list = [
        df_actions.loc[idx[:, :, ep_start, :]].values.tolist()  # type: ignore
        for ep_start in episodestart_index
    ]
    a_n_t = keras.utils.pad_sequences(
        actions_list, padding='post', dtype=np.float32, value=padding_value
    )

    # array of next_states for minibatch
    df_nstates = df_decoded.loc[:, idx['nstate', ['velocity', 'thrust', 'brake']]]  # type: ignore
    nstates_list = [
        df_nstates.loc[idx[:, :, ep_start, :]].values.tolist()  # type: ignore
        for ep_start in episodestart_index
    ]
    ns_n_t = keras.utils.pad_sequences(
        nstates_list, padding='post', dtype=np.float32, value=padding_value
    )

    return s_n_t, a_n_t, r_n_t, ns_n_t


def encode_episode_dataframe_from_series(
    observations: List[pd.Series],
    torque_table_row_names: List[str],
    episode_start_dt: datetime,
    driver_str: str,
    truck_str: str,
) -> pd.DataFrame:
    """
    encode the list of observations as a dataframe with multi-indexed columns
    """
    episode = pd.concat(
        observations, axis=1
    ).transpose()  # concat along columns and transpose to DataFrame, columns not sorted as (s,a,r,s')
    episode.columns.names = ["tuple", "rows", "idx"]
    episode.set_index(("timestamp", "", 0), append=False, inplace=True)
    episode.index.name = "timestamp"
    episode.sort_index(axis=1, inplace=True)

    # convert columns types to float where necessary
    state_cols_float = [("state", col) for col in ["brake", "thrust", "velocity"]]
    action_cols_float = [
        ("action", col) for col in [*torque_table_row_names, "speed", "throttle"]
    ]
    reward_cols_float = [("reward", "work")]
    nstate_cols_float = [("nstate", col) for col in ["brake", "thrust", "velocity"]]
    for col in (
        action_cols_float + state_cols_float + reward_cols_float + nstate_cols_float
    ):
        episode[col[0], col[1]] = episode[col[0], col[1]].astype(
            "float"
        )  # float16 not allowed in parquet

    # Create MultiIndex for the episode, in the order 'episodestart', 'vehicle', 'driver'
    episode = pd.concat(
        [episode],
        keys=[pd.to_datetime(episode_start_dt)],
        names=["episodestart"],
    )
    episode = pd.concat([episode], keys=[driver_str], names=["driver"])
    episode = pd.concat([episode], keys=[truck_str], names=["vehicle"])
    episode.sort_index(inplace=True)  # sorting in the time order of timestamps

    return episode
