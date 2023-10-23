def show_fold_diag(df):
    for fold, df0 in df.groupby('fold'):
        counts = {}
        counts['all'] = len(df0)
        for diag, df1 in df0.groupby('diag'):
            counts[diag] = len(df1)
        print(fold, ' '.join(f'{k}:{v}' for k, v in counts.items()))
