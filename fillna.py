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

    content_type = params['contenttype']

    # Convert empty strings to None so fillna sees them
    # If the value is provided by the user, convert everything
    # Else don't convert entirely empty column as that is pointless
    for c in table.columns:
        if table[c].dtype == np.object:
            if (content_type == 0) or (content_type ==1 and not (table[c]=='').all()):
                table[c] = table[c].apply(empty_str_to_none)

    if content_type == 0:
        # User supplied value
        fill_val = params['fillvalue'].strip()
        if fill_val == '':
            # If no actual value is supplied, NOP
            return table
        for col in cols:
            # Fill in the value according to the data type
            #if table[col].isnull().all():
                # If entire column is None, skip it.
            #    continue
            if not table[col].isnull().any():
                # If entire column has no Nones, skip it.
                continue
            if table[col].dtype == np.float64:
                # If type is float, try converting value to that.
                # If that doesn't work, fill and coerce to string.
                # (No need to worry about int, as int64 columns cannot have NaN)
                try:
                    num_val = float(fill_val)
                    table[col] = table[col].fillna(num_val)
                except:
                    table[col] = table[col].fillna(str(fill_val)).astype(str)
            else:
                # For other types, fill in the string directly
                table[col] = table[col].fillna(str(fill_val))

    elif content_type == 1:
        # Adjacent value
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
