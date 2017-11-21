def render(table, params):
    # Get columns as an array
    cols = params['colnames']
    if cols is None:
        return table    
    cols = [c.strip() for c in cols.split(',')]
    if cols == [] or cols == ['']:
        return table

    method = params['method']
    if method == 'Down':
        table[cols] = table[cols].fillna(axis = 0, method='ffill')
    else:
        table[cols] = table[cols].fillna(axis=0, method='bfill')

    return table
