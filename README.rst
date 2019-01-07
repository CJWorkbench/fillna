fillna
------

Workbench module that fills in missing values.

Developing
----------

First, get up and running:

1. ``pip3 install pipenv``
2. ``pipenv sync`` # to download dependencies
3. ``pipenv run python ./test_fillna.py`` # to test

To add a feature on the Python side:

1. Write a test in ``test_fillna.py``
2. Run ``pipenv run python ./test_fillna.py`` to prove it breaks
3. Edit ``fillna.py`` to make the test pass
4. Run ``pipenv run python ./test_fillna.py`` to prove it works
5. Commit and submit a pull request

To develop continuously on Workbench:

1. Check this code out in a sibling directory to your checked-out Workbench code
1. Start Workbench with ``bin/dev start``
2. In a separate tab in the Workbench directory, run ``pipenv run ./manage.py develop-module ../fillna https://github.com/CJWorkbench/fillna.git``
3. Edit this code; the module will be reloaded in Workbench immediately
4. When viewing the module in Workbench, modify parameters to re-render output
