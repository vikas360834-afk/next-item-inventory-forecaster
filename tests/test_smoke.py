from app.core import *


def test_import_core():
    import app.core
    assert app.core is not None
