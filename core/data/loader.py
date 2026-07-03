import pandas as pd

def load_dataset(path):
    raw = pd.read_excel(path, header=None)
    rows = []
    for _, r in raw.iterrows():
        material = r[3]
        if material not in ("композитный", "стальной"):
            continue
        rows.append({
            "shifr": r[2],
            "a_h0": float(r[1]),
            "material": material,
            "is_steel": 1 if material == "стальной" else 0,
            "H": float(r[4]),
            "s": float(r[5]),
            "R": float(r[6]),
            "E": float(r[7]),
            "Qдв_СП63": float(r[9]),
            "Qдв_рук78": float(r[10]),
        })
    return pd.DataFrame(rows)


def profile_ids(df, feature_cols):
    geom = [c for c in feature_cols if c != "a_h0"]
    ids, _ = pd.factorize(df[geom].astype(str).agg("|".join, axis=1))
    return ids
