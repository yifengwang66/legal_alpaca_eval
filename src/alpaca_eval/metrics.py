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


def round_rank_evaluate(preferences):
    """
        preferences example:
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
            ],
            [
                {
                    "model": "model_1",
                    "rank": 0,
                },
                {
                    "model": "model_2",
                    "rank": 0,
                },
            ],
        ]
    """
    win_list = []
    models_score_record = {}
    result_dic = {}
    initial_metrics = dict(
        win_rate=0,
        n_wins=0,
        n_draws=0,
        n_total=len(preferences),
    )
    for i, per_rank in enumerate(preferences):
        if models_score_record.get(per_rank[0]["model"]) is None:
            models_score_record[per_rank[0]["model"]] = []
        if models_score_record.get(per_rank[1]["model"]) is None:
            models_score_record[per_rank[1]["model"]] = []
        # 两个模型都存在显著错误的情况
        if per_rank[0]["rank"] == 0 and per_rank[1]["rank"] == 0:
            win_list.append("both_wrong")
            models_score_record[per_rank[0]["model"]].append(0)
            models_score_record[per_rank[1]["model"]].append(0)
            win_name = per_rank[0]["model"]
            lose_name = per_rank[1]["model"]
        else:
            win_name, lose_name = (per_rank[0]["model"], per_rank[1]["model"]) \
                if per_rank[0]["rank"] == 1 \
                else (per_rank[1]["model"], per_rank[0]["model"])
            models_score_record[win_name].append(1)
            models_score_record[lose_name].append(0)
            if result_dic.get(win_name) is None:
                result_dic[per_rank[0]["model"]] = initial_metrics.copy()
                result_dic[per_rank[1]["model"]] = initial_metrics.copy()
            result_dic[win_name]["n_wins"] += 1
            result_dic[lose_name]["n_draws"] += 1
            win_list.append(win_name)
        if i == len(preferences) - 1:
            result_dic[win_name]["win_rate"] = round(
                result_dic[win_name]["n_wins"] / result_dic[win_name]["n_total"], 4
            ) * 100
            result_dic[lose_name]["win_rate"] = round(
                result_dic[lose_name]["n_wins"] / result_dic[lose_name]["n_total"], 4
            ) * 100
            result_dic["win_name"], result_dic["lose_name"] = (win_name, lose_name) \
                if result_dic[win_name]["win_rate"] > result_dic[lose_name]["win_rate"] \
                else (lose_name, win_name)
    return win_list, result_dic, models_score_record
