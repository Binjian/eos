import pandas as pd


def nest(d: dict) -> dict:
    """
    Convert a flat dictionary with tuple key to a nested dictionary
    """
    result = {}
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
    Convert a eos dataframe with multi-indexed columns to a nested dictionary
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


def decode_mongo_documents(
    df: pd.DataFrame,
    torque_table_row_names: list[str],
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.Series], list[pd.DataFrame]]:
    """
    decoding the batch observations from mongodb nested dicts to pandas dataframe
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
        df_state = ser_state.unstack([0, 1, 2, 3, 4, 5])
        multiindex = df_state.columns
        df_state.set_index(multiindex[-1], inplace=True)  # last index has timestep
        df_states.append(df_state)

        # decode action
        ser_action = ser_decoded.loc[
            idx[:, :, :, :, 'action', [*torque_table_row_names, 'throttle']]
        ]
        df_action = ser_action.unstack(level=[0, 1, 2, 3, 4, 5])
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
        ser_reward = ser_reward.unstack([0, 1, 2, 3, 4, 5])
        multiindex = ser_reward.columns
        ser_reward.set_index(multiindex[-1], inplace=True)  # last index has timestep
        # ser_reward
        ser_rewards.append(ser_reward)

        # decode nstate
        ser_nstate = ser_decoded.loc[
            idx[:, :, :, :, 'nstate', ['brake', 'thrust', 'velocity', 'timestep']]
        ]
        df_nstate = ser_nstate.unstack([0, 1, 2, 3, 4, 5])
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
    i1 = [
        '' if str(idx) in (str(pd.NA), 'nan', '') else idx for idx in i1
    ]  # convert index of level 2 type to int and '' if NA
    i2 = multi_col.get_level_values(
        1
    )  # must be null string instead of the default pd.NA or np.nan
    i2 = [
        '' if str(idx) in (str(pd.NA), 'nan', '') else idx for idx in i2
    ]  # convert index of level 2 type to int and '' if NA
    i3 = multi_col.get_level_values(
        2
    )  # must be null string instead of the default pd.NA or np.nan
    i3 = [
        '' if str(idx) in (str(pd.NA), 'nan', '') else int(idx) for idx in i3
    ]  # convert index of level 2 type to int and '' if NA

    multi_col = pd.MultiIndex.from_arrays([i1, i2, i3])
    multi_col.names = ['qtuple', 'rows', 'idx']
    df.columns = multi_col

    df = df.set_index(['vehicle', 'driver', 'episodestart', df.index])

    return df
