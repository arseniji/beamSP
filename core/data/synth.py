import numpy as np
import pandas as pd

from .loader import profile_ids as _real_profile_ids

def synthesize(df, samples_per_profile, noise, feature_cols, seed=1488):
    rng = np.random.default_rng(seed)
    df = df.copy().reset_index(drop=True)
    df["is_synth"] = 0
    df["profile_id"] = _real_profile_ids(df, feature_cols)
    synth_rows = []
    for _, row in df.iterrows():
        for _ in range(samples_per_profile):
            new = row.copy()
            for col, amp in noise.items():
                new[col] = row[col] + rng.uniform(-amp, amp)
            new["is_synth"] = 1
            synth_rows.append(new)
    if synth_rows:
        df = pd.concat([df, pd.DataFrame(synth_rows)], ignore_index=True)
    return df
