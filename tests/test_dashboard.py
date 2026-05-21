import os


def test_dashboard_exists():

    assert os.path.exists("src/dashboard/app.py")
