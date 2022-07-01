# openeo_processes
`openeo_processes` provides implementations of many [openEO processes](https://github.com/Open-EO/openeo-processes) in Python.
Processes are currently aligned with openEO API version 1.0.

## Installation
TODO, will be installable through PyPI by the next release.
**This library requires the GDAL library to be present.**

## Development Environment
### Managing dependencies
This project uses [poetry](https://github.com/python-poetry/poetry) to manage dependencies through virtual environments. The poetry CLI can be installed easily following the instructions in the [official documentation](https://python-poetry.org/docs/master/#installing-with-the-official-installer). Note that poetry is already installed and setup on the provided devcontainer. 

To install this project and its dependencies into a fresh virtual environment run:
- `poetry install` to install all dependencies (core + development)
or
- `poetry install --no-dev` to install only the core dependencies

To add a dependency run `poetry add <PACKAGENAME>@<VERSION>`. Use the optional `--dev` flag to add it as a development dependency. 
Note: When adding new dependencies, please do not pin to specific versions unless absolutely necessary (see discussion in #91). Usage of the caret-operator is preferred for specifying versions, this will allow versions to range up to the next major version (`^1.2.3` is equivalent to	`>=1.2.3 <2.0.0`, see [poetry documentation on caret requirements](https://python-poetry.org/docs/master/dependency-specification#caret-requirements) for additional examples).

The `poetry.lock` file is only included in the source to speed up dependency resolution during CI, this can be ignored on a local build. 

To run a shell command within this Poetry-managed virtual environment use `poetry run [CMD]`, e.g. `poetry run python -m pytest`.

For advanced options, please see the documentation at https://python-poetry.org/docs/.

### Devcontainer (recommended)
Several processes depend on the GDAL library for I/O, which requires a range of C libraries to be present on the system.
In order to ensure a reproducible development environment across contributors, this project comes with a development container image that has the following dependencies preinstalled:
- GDAL
- poetry for dependency management
- nox and older versions of python to test against 

This image is also what is used to run the CI pipeline.

How to use?
1) Install the [`VSCode Remote - Containers`](https://code.visualstudio.com/docs/remote/containers) extension
2) Open the repo in VSCode
3) Run the `Remote-Containers: Rebuild and Reopen in Container` command or click the `Open in Container` pop-up
4) Optional: Mount your ssh keys as shown [here](https://code.visualstudio.com/docs/remote/containers#_sharing-git-credentials-with-your-container) to authenticate with Github.
5) Done, you should now be able to run this library entirely from the isolation of a container.

This devcontainer is intended to be used in conjunction with the extension, which lets you use it as a full-featured development environment. Note that this isn't a strict requirement, but a strong recommendation that will make life easier. Also note that this package is distributed in the standard formats through PyPI and can be installed using pip without any need for Docker or poetry - provided that GDAL is already correctly installed on the target system.

Building the devcontainer without VSCode is not recommended, it is **not enough** to only run `docker build -f ./.devcontainer/Dockerfile .`. You will also have to mount the source code into the container and afterwards run the `postCreateCommand` from `devcontainer.json` in order for everything to be set up.

### Virtual Environment
If you already have GDAL installed on your system and don't want to develop in a container, it is possible to setup a local virtual environment using
```
poetry install
```

Testing across multiple versions of Python using `nox` might not work if those versions are not installed on your system. See the `.devcontainer/Dockerfile` for an example of how this can be done on Ubuntu. 


## Continuous Integration
CI on this project runs the following checks against all push events:
- Build the devcontainer
- Run the following nox sessions on multiple versions of Python inside the devcontainer:
    - mypy: static type checking
    - test: run testsuite using pytest

Because the CI uses the devcontainer as the base environment to run nox sessions, they can also be run locally from the devcontainers shell, e.g. 
`nox --session tests --python=3.8` to run the tests session on a specific version of python.

## FAQ
### `Permission denied` errors with devcontainer 
From [the docs on VSCode Remote - Containers](https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user):
> Inside the container, any mounted files/folders will have the exact same permissions as outside the container - including the owner user ID (UID) and group ID (GID). Because of this, your container user will either need to have the same UID or be in a group with the same GID. The actual name of the user / group does not matter. The first user on a machine typically gets a UID of 1000, so most containers use this as the ID of the user to try to avoid this problem.

By default, the devcontainer uses UID=1000 and GID=1000, but if the files on your host system belong to a different user, you can configure these values using the `USER_UID` and `USER_GID` docker build arguments.  
