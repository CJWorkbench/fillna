def render(table, params):
    # Get columns as an array
    cols = params['colnames']
    if cols is None:
        return table    

    cols = [c.strip() for c in cols.split(',')]
    if cols == [] or cols == ['']:
        return table

    import numpy as np

    def empty_str_to_none(s):
        if isinstance(s, str) and s=='':
            return None
        else:
            return s

    # Convert empty strings to None so fillna sees them
    # Except if we have an entirely empty string column, in which case do nothing
    for c in table.columns:
        if table[c].dtype == np.object:
            if not (table[c]=='').all():
                table[c] = table[c].apply(empty_str_to_none)


    method = params['method']
    if method == 0:  # down
        table[cols] = table[cols].fillna(axis=0, method='ffill')
    else:
        table[cols] = table[cols].fillna(axis=0, method='bfill')

    # Convert any remaining None cells in string cols to empty string
    # (can happen when first/last row is empty)
    def none_to_empty_str(s):
        return s if s is not None else ''

    for c in table.columns:
        if table[c].dtype == np.object:
            if table[c].iloc[0] is None or table[c].iloc[-1] is None:
                table[c] = table[c].apply(none_to_empty_str)


    return table
