"""Rolling history sequence generation for next-item prediction without target leakage."""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd


def build_rolling_sequences(
    user_interactions: pd.DataFrame,
    max_history_length: int = 20,
    min_history: int = 1,
    padding_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate rolling history sequences and next-item targets for training.
    
    Guarantees:
    1. Chronological order strictly preserved.
    2. Zero target leakage: target item is NEVER present in the input history.
    3. Padding with `padding_idx` on the left/beginning of sequence.
    4. Truncation to the most recent `max_history_length` items.
    
    Args:
        user_interactions: DataFrame containing ['user_idx', 'movie_idx', 'timestamp'] sorted chronologically.
        max_history_length: Maximum number of previous items in history.
        min_history: Minimum required items in history to form an example (default 1).
        padding_idx: Token ID used for padding (0).
        
    Returns:
        histories: np.ndarray of shape [N, max_history_length] (int64)
        targets: np.ndarray of shape [N] (int64)
        user_ids: np.ndarray of shape [N] (int64)
    """
    histories_list = []
    targets_list = []
    users_list = []

    # Group by user_idx
    for user_idx, group in user_interactions.groupby("user_idx", sort=False):
        movie_sequence = group["movie_idx"].tolist()
        n_items = len(movie_sequence)
        if n_items <= 1:
            continue

        # Rolling window: at each step t from min_history to n_items-1
        # history is items up to t-1, target is item at t
        for t in range(min_history, n_items):
            target_item = movie_sequence[t]
            history = movie_sequence[:t]

            # Truncate to most recent max_history_length items
            if len(history) > max_history_length:
                history = history[-max_history_length:]

            # Left pad to max_history_length with padding_idx
            padded_history = [padding_idx] * (max_history_length - len(history)) + history

            histories_list.append(padded_history)
            targets_list.append(target_item)
            users_list.append(user_idx)

    if not histories_list:
        return np.zeros((0, max_history_length), dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    return (
        np.array(histories_list, dtype=np.int64),
        np.array(targets_list, dtype=np.int64),
        np.array(users_list, dtype=np.int64),
    )


def build_eval_sequences(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    max_history_length: int = 20,
    padding_idx: int = 0,
    prior_target_df: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build evaluation examples (one target per user) using historical interactions available at evaluation time.
    
    For Validation:
      History = Train interactions. Target = Val item.
    For Test:
      History = Train interactions + Val interaction. Target = Test item.
      
    Args:
        train_df: Training interactions.
        target_df: Validation or Test interactions.
        max_history_length: Maximum history length.
        padding_idx: Padding token (0).
        prior_target_df: Optional prior target dataframe to append to history (e.g., validation df when evaluating test).
        
    Returns:
        histories: np.ndarray of shape [num_eval_users, max_history_length]
        targets: np.ndarray of shape [num_eval_users]
        user_ids: np.ndarray of shape [num_eval_users]
    """
    # Build history mapping from train_df
    histories_by_user = {}
    for user_idx, group in train_df.groupby("user_idx", sort=False):
        histories_by_user[user_idx] = group["movie_idx"].tolist()

    if prior_target_df is not None:
        for user_idx, group in prior_target_df.groupby("user_idx", sort=False):
            if user_idx in histories_by_user:
                histories_by_user[user_idx].extend(group["movie_idx"].tolist())
            else:
                histories_by_user[user_idx] = group["movie_idx"].tolist()

    histories_list = []
    targets_list = []
    users_list = []

    for _, row in target_df.iterrows():
        u = int(row["user_idx"])
        target = int(row["movie_idx"])
        user_history = histories_by_user.get(u, [])

        if len(user_history) > max_history_length:
            user_history = user_history[-max_history_length:]

        padded_history = [padding_idx] * (max_history_length - len(user_history)) + user_history
        histories_list.append(padded_history)
        targets_list.append(target)
        users_list.append(u)

    return (
        np.array(histories_list, dtype=np.int64),
        np.array(targets_list, dtype=np.int64),
        np.array(users_list, dtype=np.int64),
    )
