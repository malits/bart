from distutils.core import setup, Extension
import os
import re

BART_PATH = os.environ['TOOLBOX_PATH']

src_dirs = []
for root, dirs, files in list(os.walk(f"{BART_PATH}/src")):
       for d in dirs:
              curr = os.path.join(root, d)
              src_files = list(os.walk(curr))[0][2]
              for f in src_files:
                     if re.match(r'[A-Za-z0-9]+\.c\b', f):
                            name = os.path.join(curr, f)
                            src_dirs.append(name)

phantom_module = Extension('_phantom',
                           include_dirs=['../src/'],
                           sources=src_dirs,
                           libraries=['lapacke']
                           )

setup (name = 'phantom',
       version = '0.1',
       author      = "BART",
       description = """Simple swig example from docs""",
       ext_modules = [phantom_module],
       py_modules = ["phantom"],
       )