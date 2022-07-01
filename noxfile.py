from nox_poetry import session

python_versions=["3.8", "3.9"]

@session(python=python_versions)
def tests(session):
    session.install("pytest", ".")
    session.run("pytest")

@session(python=python_versions)
def mypy(session):
    session.install("mypy")
    session.run("mypy", "--install-types", "--non-interactive")
