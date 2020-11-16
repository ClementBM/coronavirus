# Introduction
Template repository for python package

Manuel steps to generate and publish the package to TestPyPI with poetry, documentation from [packaging.python](https://python-poetry.org/docs/)

Build the package, generate distribution archives
```shell
poetry build
```

Add Test PyPI as an alternate package repository
```shell
poetry config repositories.testpypi https://test.pypi.org/legacy/
```

Upload/publish package/distribution archive to TestPyPI (a separate instance of the Python Package Index)
```shell
poetry publish -r testpypi
```

# Installation
```shell
pip install --index-url https://test.pypi.org/simple/ coronavirus
```
or
```shell
pip3 install --index-url https://test.pypi.org/simple/ coronavirus
```

# Usage
```python
from coronavirus import main
```

# Code of Conduct

# History (changelog)
