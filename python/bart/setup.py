from distutils.core import setup, Extension

def main():
    setup(
        name="bart",
        version="0.0",
        description="BART: Toolbox for Computational Magnetic Resonance Imaging (MRI)",
        author="Max Litster",
        author_email="maxlitster@berkeley.edu",
        ext_modules=[Extension("bart", ["bartmodule.c"])]
    )

if __name__=="__main__":
    main()