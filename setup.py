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
        # For the pebs trace environment implementation
        "gym(==0.21.0)",
        # For building placement models.
        "pfrl(==0.3.0)",
        # Pinned dependency for pfrl.
        "pytorch(==1.8.2)",
        # For quick computations on arrays of data.
        "numpy(==1.22.0)",
        # For observations processing
        "scipy(==1.8.0)",
        # For building page tables and tracking contiguous memory chunks
        "intervaltree(==3.1.0)",
        # For reading PEBS traces and exporting preprocessed traces.
        "pandas(==1.3.5)",
        # To show progress on preprocessing and training.
        "tqdm(==4.62.3)",
    ],
    packages=["memai"],
    scripts=[
        # For testing estimations for different mappings and evaluate estimator
        # runtime overhead.
        "memai/estimator.py",
        # To preprrocess PEBS traces into AI input.
        "memai/preprocessing.py",
        # To run a gym environment with dummy actions and evaluate runtime
        # overhead of runing the environment compared to executing the
        # application.
        "memai/env.py",
        # To train and evaluate AI models.
        "memai/ai.py",
    ],
    package_data={
        # Traces obtained from applications run.
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
