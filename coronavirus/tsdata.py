def split_for_training(df, freq=1):
    n = len(df)

    m = freq * n
    if m - 1 != df.index[-1]:
        raise Exception(
            """
            Ensure that the dataframe is incrementally indexed (default integer index)
            df.reset_index()
            """
        )

    train_df = df[0 : int(m * 0.7)]
    validation_df = df[int(m * 0.7) : int(m * 0.9)]
    test_df = df[int(m * 0.9) :]

    return train_df, validation_df, test_df