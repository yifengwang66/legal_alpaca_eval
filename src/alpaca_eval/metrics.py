import logging
from typing import Sequence, Union

import pandas as pd


def pairwise_to_winrate(preferences: Union[pd.Series, Sequence]) -> dict[str, int]:
    """Extract head2head metrics (n_wins, n_counts, win_rate) from a sequence preference.
    This assumes that the preference is encoded as 0 for draw, 1 for base win, 2 when the model to compare wins.
    """
    if not isinstance(preferences, pd.Series):
        series_preferences = pd.Series(preferences)
    else:
        series_preferences = preferences.copy()

    is_preference = series_preferences.isin([0, 1, 2])
    n_not_pair = sum(~is_preference)
    if n_not_pair > 0:
        logging.info(f"drop {n_not_pair} outputs that are not[0, 1, 2]")
    series_preferences = series_preferences[is_preference].astype(int).copy()

    n_draws = (series_preferences == 0).sum()
    n_wins_base = (series_preferences == 1).sum()
    n_wins = (series_preferences == 2).sum()
    n_total = len(series_preferences)
    series_preferences[series_preferences == 0] = 1.5
    series_preferences -= 1
    win_rate = series_preferences.mean()

    return dict(
        win_rate=win_rate * 100,
        standard_error=series_preferences.sem() * 100,
        n_wins=n_wins,
        n_wins_base=n_wins_base,
        n_draws=n_draws,
        n_total=n_total,
    )


def multiwise_to_avg_rank(rank_obj_list):
    """
    rank_obj_list example:
    [
        [
            {
                "model": "model_1",
                "rank": 1,
            },
            {
                "model": "model_2",
                "rank": 2,
            },
            ...
        ],
        [
            {
                "model": "model_1",
                "rank": 2,
            },
            {
                "model": "model_2",
                "rank": 1,
            },
            ...
        ],
    ]

    """
    df = pd.DataFrame([item for per_rank in rank_obj_list for item in per_rank])
    # 计算每个model的平均rank
    average_rank = df.groupby('model')['rank'].mean().rename('avg_rank')

    # 计算每个model获得的每个排名的次数
    rank_counts = df.groupby(['model', 'rank']).size().unstack(fill_value=0)

    return pd.merge(
        average_rank,
        rank_counts,
        on="model"
    )
