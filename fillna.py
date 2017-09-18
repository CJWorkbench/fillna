class Importable:
    @staticmethod
    def __init__(self):
        pass

    @staticmethod
    def event():
        pass

    @staticmethod
    def render(wf_module, table):
        cols = wf_module.get_param_string('colnames').split(',')
        cols = [c.strip() for c in cols]

        if cols == [] or cols == ['']:
            return table

        for c in cols:
            if not c in table.columns:
                wf_module.set_error('There is no column named %s' % c)
                return None

        method = wf_module.get_param_menu_string('method')
        if method == 'Down':
            table[cols] = table[cols].fillna(axis = 0, method='ffill')
        else:
            table[cols] = table[cols].fillna(axis=0, method='bfill')

        wf_module.set_ready(notify=False)
        return table