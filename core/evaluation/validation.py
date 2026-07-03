import numpy as np

def leave_one_group_out(model_factory, X, y, groups, is_synth=None):
    preds = np.full(len(y), np.nan)
    if is_synth is None:
        is_synth = np.zeros(len(y), dtype=bool)
    is_synth = np.asarray(is_synth, bool)

    for g in np.unique(groups):
        test_mask = (groups == g) & (~is_synth)
        train_mask = groups != g
        if test_mask.sum() == 0:
            continue
        m = model_factory()
        m.fit(X[train_mask], y[train_mask])
        preds[test_mask] = m.predict(X[test_mask])
    return preds
