import sys
try:
    import xlrd
    import importlib
    print('python_exec:', sys.executable)
    print('xlrd_version:', xlrd.__version__)
    print('xlrd_file:', xlrd.__file__)
except Exception as e:
    print('IMPORT_ERROR', str(e))

