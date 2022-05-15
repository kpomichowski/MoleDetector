import pandas as pd

train_df_path = r"../data/train_df.csv"

df = pd.read_csv(train_df_path, index_col="Unnamed: 0")


def replicate_rows_by_k_factor(df: pd.DataFrame, k_factors: list) -> pd.DataFrame:

    assert len(df.dx.unique()) == len(
        k_factors
    ), "inaproperiate length of k, must equals to number of columns in df"
    df_copy = df.copy(deep=True)
    for k_index in range(len(k_factors)):
        df_copy = df_copy.append(
            [df_copy.loc[df_copy["lesion_type"] == k_index, :]] * (k_factors[k_index]),
            ignore_index=True,
        )

    return df_copy


# 'akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'
k_factors = [27, 26, 9, 100, 17, 0, 80]
replicated_df = replicate_rows_by_k_factor(df, k_factors)
replicated_df.to_csv("../data/r_train_df.csv", index=True)
