import os
from os import listdir
from os.path import isfile, join
from distutils.core import setup, Extension


TOOLBOX_PATH = os.environ['TOOLBOX_PATH']


def get_libraries():
    library_dir = join(TOOLBOX_PATH, 'lib')
    libraries = [lib[3:-2] for lib in listdir(library_dir) if isfile(join(library_dir, lib)) \
                and lib != '.gitignore'] #TODO: Handle .gitignore in a cleaner way
    return libraries + ['lapack', 'blas', 'lapacke', 'blas', 'quadmath', 'z', 'png', 'fftw3f',\
                        'fftw3f_threads', 'm', 'uuid']


def bart_extension():
    libraries = get_libraries()
    return Extension ("bart", 
                    extra_compile_args=["-static"],
                    sources=["bartmodule.c"],
                    libraries=libraries,
                    library_dirs=[f"{TOOLBOX_PATH}/lib/"]
                )

def main():
    setup(
        name="bart",
        version="0.0",
        description="BART: Toolbox for Computational Magnetic Resonance Imaging (MRI)",
        author="Max Litster",
        author_email="maxlitster@berkeley.edu",
        ext_modules=[bart_extension()]
    )

if __name__=="__main__":
    main()
