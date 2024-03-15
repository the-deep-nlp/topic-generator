from setuptools import setup, find_packages

PACKAGE_VERSION = "0.1"
MINOR_RELEASE = "0"

setup(
    name="topic_generator",
    version=f"{PACKAGE_VERSION}.{MINOR_RELEASE}",
    packages=find_packages(where="src"),  # include all packages under src
    package_dir={"": "src"},
    include_package_data=True,

    install_requires=[
        "numba==0.56.4",
        "bertopic",
        "umap-learn",
        "hdbscan",
        "scikit-learn",
        "torch@http://download.pytorch.org/whl/cpu/torch-2.1.0%2Bcpu-cp310-cp310-linux_x86_64.whl#sha256=5077921fc2b54e69a534f3a9c0b98493c79a5547c49d46f5e77e42da3610e011"
        ],

    author="",
    author_email="",
    description="This package contains topic modeling utilities based on BertTopic library",
    classifiers=[
        ""
    ]
)
