import os
from setuptools import setup

version = os.environ.get("MODULEVER", "0.0")

setup(
    #    install_requires = ['cothread'], # require statements go here
    name="pediip",
    version=version,
    description="Module",
    author="Melanie Vollmar",
    author_email="melanie.vollmar@diamond.ac.uk",
    packages=["modules",
              "modules.create_mr_set",
#              "module.cnn",
#              "modules.db_files"
              ],
    install_requires=[
#         "procrunner",
         "pybind11",
         "Gemmi",
         "biopython",
         "xraydb",
         "pandas",
#         "six==1.15.0",
         "six",
#         "numpy==1.21.0",
#         "numpy==1.19.2",
         "numpy",
#         "tensorflow==2.5.3",
         "tensorflow",
#         "h5py==3.1.0",
         "h5py",
#         "tensorflow<=1.13",
         "Keras",
         "Pillow",
         "PyYaml",
         "scikit-learn",
         "mrcfile",
         "logconfig",
         "matplotlib",
         "configargparse",
#         "scipy==1.4.1",
         "scipy",
#         "h5py==2.10.0",
    ],
    scripts=[
            "bin/create_mr_set",
            "bin/populate_database",
            ],
    
    entry_points={
        "console_scripts": [
#            "pediip.create_mr_set = modules.create_mr_set.create_mr_set:main",#need to fix command line
            "pediip.prepare = modules.cnn.command_line_preparation:main",
            "pediip.prepare_binary = modules.cnn.command_line_preparation_binary:main",
            "pediip.prepare_random_combined = modules.cnn.command_line_preparation_random_pick_combined:main",
#            "topaz3.test_split = topaz3.train_test_split:command_line",
#            "topaz3.predict_from_maps = topaz3.predictions:command_line",
#            "topaz3.filter = topaz3.filters:filter_command_line",
        ]
    },
    #    entry_points = {'console_scripts': ['test-python-hello-world = topaz3.topaz3:main']}, # this makes a script
    #    include_package_data = True, # use this to include non python files
    license="BSD3 license",
    zip_safe=False,
)
