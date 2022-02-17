from distutils.core import setup

setup(
    name="memai",
    version="0.0.1",
    description="MemAI: A reproducible environment to train memory management AIs.",
    author="Nicolas Denoyelle, Balazs Gerofi",
    author_email="ndenoyelle@anl.gov, bgerofi@riken.jp",
    url="https://github.com/bgerofi/Heterogeneous-Memory-Management",
    license="MIT",
    requires=[
        "numpy(==1.22.0)",
        "intervaltree(==3.1.0)",
        "pandas(==1.3.5)",
    ],
    packages=["memai"],
    scripts=["memai/estimator.py"],
    package_data={
        "memai": [
            "data/lammps-7-12-2016/src/USER-INTEL/TEST/*.feather",
            "data/lulesh2.0.2/*.feather",
            "data/miniFE/miniFE-2.0.1_openmp_opt/*.feather",
            "data/nekbone-2.3.4/test/example1/*.feather",
        ]
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.10",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Benchmark",
    ],
)
