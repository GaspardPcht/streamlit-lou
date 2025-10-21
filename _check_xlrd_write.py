import sys
out = {}
try:
    import xlrd
    out['xlrd_version'] = getattr(xlrd, '__version__', 'unknown')
    out['xlrd_file'] = getattr(xlrd, '__file__', 'unknown')
    out['python_exec'] = sys.executable
    out['ok'] = True
except Exception as e:
    out['ok'] = False
    out['error'] = str(e)

with open('xlrd_info.txt', 'w', encoding='utf-8') as f:
    for k,v in out.items():
        f.write(f"{k}: {v}\n")
print('WROTE xlrd_info.txt')

