from pathlib import Path

from setuptools import find_packages, setup


def read_requirements(path):
    return list(Path(path).read_text().splitlines())


base_reqs = read_requirements("ts_combine/requirements/core.txt")
torch_reqs = read_requirements("ts_combine/requirements/torch.txt")

all_reqs = base_reqs + torch_reqs
all_reqs = base_reqs

with open("README.md") as fh:
    LONG_DESCRIPTION = fh.read()

URL = "https://github.com/Chaoqun-Guo/ts_combine"

PROJECT_URLS = {
    "Bug Tracker": "https://github.com/Chaoqun-Guo/ts_combine/issues",
    "Documentation": URL,
    "Source Code": "https://github.com/Chaoqun-Guo/ts_combine",
}

setup(
    name="ts_combine",
    version="0.0.1",
    description="A time series analysis integration toolkit in Python.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    project_urls=PROJECT_URLS,
    url=URL,
    maintainer="Chaoqun Guo",
    maintainer_email="chaoqunguo317@outlook.com",
    license="Apache License 2.0",
    packages=find_packages(),
    install_requires=all_reqs,
    package_data={
        "ts_combine": ["py.typed"],
    },
    zip_safe=False,
    python_requires=">=3.7",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    keywords="time series forecasting",
)
