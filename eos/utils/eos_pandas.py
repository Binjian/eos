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
    d = df_multi_indexed_col.to_dict('index')
    return {k: nest(v) for k, v in d.items()}


def decode_mongo_documents(
    df: pd.DataFrame,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.Series], list[pd.DataFrame]]:

    """
    decoding the batch observations from mongodb nested dicts to pandas dataframe
    TODO need to check whether sort_index is necessary
    """

    dict_observations_list = [
        {
            (
                meta['episodestart'],
                meta['vehicle'],
                meta['driver'],
                meta['timestamp'],
                key1,
                key2,
                key3,
            ): value
            for key1, obs1 in obs.items()
            for key2, obs2 in obs1.items()
            for key3, value in obs2.items()
        }
        for meta, obs in zip(df['meta'], df['observation'])
    ]

    df_actions = []
    df_states = []
    df_nstates = []
    ser_rewards = []
    idx = pd.IndexSlice
    for dict_observations in dict_observations_list:
        ser_decoded = pd.Series(dict_observations)
        ser_decoded.index.names = [
            'episodestart',
            'vehicle',
            'driver',
            'timestamp',
            'tuple',
            'rows',
            'idx',
        ]

        # decode action
        ser_action = ser_decoded.loc[idx[:, :, :, :, 'action', 'timestep']]
        df_action = ser_action.unstack(level=[0, 1, 2, 3, 4, 5])
        multiindex = df_action.columns
        df_action.set_index(multiindex[-1], inplace=True)
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
                'tuple',
                'rows',
                'speed',
                'timestep',
            ],
        )
        df_actions.append(df_action)

        # decode state
        ser_state = ser_decoded.loc[
            idx[:, :, :, :, 'state', ['brake', 'thrust', 'velocity', 'timestep']]
        ]
        df_state = ser_state.unstack([0, 1, 2, 3, 4, 5])
        multiindex = df_state.columns
        df_state.set_index(multiindex[-1], inplace=True)
        df_states.append(df_state)

        # decode nstate
        ser_nstate = ser_decoded.loc[
            idx[:, :, :, :, 'nstate', ['brake', 'thrust', 'velocity', 'timestep']]
        ]
        df_nstate = ser_nstate.unstack([0, 1, 2, 3, 4, 5])
        multiindex = df_nstate.columns
        df_nstate.set_index(multiindex[-1], inplace=True)
        df_nstates.append(df_nstate)

        # decode reward
        ser_reward = ser_decoded.loc[idx[:, :, :, :, 'reward', ['work', 'timestep']]]
        ser_reward = ser_reward.unstack([0, 1, 2, 3, 4, 5])
        multiindex = ser_reward.columns
        ser_reward.set_index(multiindex[-1], inplace=True)
        # ser_reward
        ser_rewards.append(ser_reward)

        return df_states, df_actions, ser_rewards, df_nstates
