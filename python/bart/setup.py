import os
from distutils.core import setup, Extension

bart_module = Extension ("bart", 
    sources=["bartmodule.c"],
    library_dirs=[os.path.join(os.environ["TOOLBOX_PATH"], "lib/lib*.a")],
    extra_link_args=["-whole-archive"]
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