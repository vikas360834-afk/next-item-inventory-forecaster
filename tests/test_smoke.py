def test_streamlit_app_compiles():
    import py_compile
    py_compile.compile("app/main.py",doraise=True)
