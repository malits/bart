import os
from distutils.core import setup, Extension

bart_module = Extension ("bart", 
    sources=["bartmodule.c"],
    runtime_library_dirs=["bart.so"]
)

def main():
    setup(
        name="bart",
        version="0.0",
        description="BART: Toolbox for Computational Magnetic Resonance Imaging (MRI)",
        author="Max Litster",
        author_email="maxlitster@berkeley.edu",
        ext_modules=[bart_module]
    )

if __name__=="__main__":
    main()
