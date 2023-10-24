import math
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

ACTIVITY_NAMES = [
    'nonproduction',
    'input',
    'remove/cut',
    'paste',
    'replace',
    'move'
]
ACTIVITY2IDX = dict(zip(ACTIVITY_NAMES, range(len(ACTIVITY_NAMES))))
TEXT_CHANGE_NAMES = ['alphanum', 'other']
TEXT_CHANGE2IDX = dict(zip(TEXT_CHANGE_NAMES, range(len(TEXT_CHANGE_NAMES))))
ACTION_TIME_COLS = [
    'action_time_max_log',
    'action_time_mean_log',
    'action_time_std_log'
]


def process_activity(data: pd.Series) -> List[float]:
    """Count activity frequencies

    Parameters
    ----------
    data : pd.Series
        activity data

    Returns
    -------
    List[float]
        activity frequencies
    """

    activity = data.apply(lambda x: 'move' if 'Move' in x else x.lower())
    activity2freq = activity.value_counts(normalize=True)
    res = [0] * len(ACTIVITY_NAMES)
    for name, freq in activity2freq.items():
        res[ACTIVITY2IDX[name]] = freq

    return res


def process_text_change(data: pd.Series) -> List[float]:
    text_change = data.apply(lambda x: 'alphanum' if x == 'q' else 'other')
    text_change2freq = text_change.value_counts(normalize=True)
    res = [0] * len(TEXT_CHANGE_NAMES)
    for name, freq in text_change2freq.items():
        res[TEXT_CHANGE2IDX[name]] = freq

    return res


def prepare_data(
    df: pd.DataFrame,
    labels: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Prepare feature matrix and (optional) labels for a given portion of data

    Parameters
    ----------
    df : pd.DataFrame
        Events DataFrame
    labels : Optional[pd.DataFrame], optional
        Labels DataFrame, by default None

    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray]]
        X, y
    """

    data = []
    n_groups = df['id'].nunique()
    for log_id, group in tqdm(df.groupby('id'), total=n_groups):
        tmp = []

        # id
        tmp.append(log_id)

        # action_time
        tmp.extend([
            math.log(group['action_time'].max() + 1),
            math.log(group['action_time'].mean() + 1),
            math.log(group['action_time'].std() + 1)
        ])

        # activity
        tmp.extend(process_activity(data=group['activity']))

        # text_change
        tmp.extend(process_text_change(data=group['text_change']))

        data.append(tmp)
    res = pd.DataFrame(
        data=data,
        columns=['id', *ACTION_TIME_COLS, *ACTIVITY_NAMES, *TEXT_CHANGE_NAMES]
    )
    if labels is not None:
        res = res.merge(labels, on='id', how='left')

    return res
