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
        "numpy",
        "pandas",
        "bertopic",
        "umap-learn",
        "hdbscan",
        "spacy",
        "scikit-learn"
        ],

    author="",
    author_email="",
    description="This package contains topic modeling utilities based on BertTopic library",
    classifiers=[
        ""
    ]
)