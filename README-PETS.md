# PETs challenge setup

## Calling Go from Python

[pets/go.py](pets/go.py) provides a helper util
to build (if necessary) the Go executable at runtime from Python,
and to run it with the given command-line arguments.

The only external runtime dependency to enable that is the Go compiler
itself, which can be installed via `dependencies -> go` section
of [pets/environment-cpu.yml](pets/environment-cpu.yml).

## Vendoring Go dependencies

Since [pets/environment-cpu.yml](pets/environment-cpu.yml) doesn't
provide a way to install Go packages, we can install them at runtime using
the "vendoring" mechanism, as mentioned in
[PETs problem description](https://www.drivendata.org/competitions/141/uk-federated-learning-2-pandemic-forecasting-federated/page/642/#how-can-i-include-software-dependencies-that-my-solution-depends).

This means all of the Go package dependencies must be pre-fetched and stored
alongside our submission code. The following commands automatically
create [vendor](vendor) directory with them:

```
go mod tidy
go mod download
go mod vendor
```

**_Caveat_**: these commands might need to be run under Linux;
we've run into issues when we first run them on macOS and
used that folder inside the (Linux) Docker container.

## Centralized Code Submission

Python APIs for [training](https://www.drivendata.org/competitions/141/uk-federated-learning-2-pandemic-forecasting-federated/page/644/#training)
and [testing](https://www.drivendata.org/competitions/141/uk-federated-learning-2-pandemic-forecasting-federated/page/644/#test)
only pass _paths_ to the input and output CSV files and the model directory,
so we don't need to install any Python dependencies for it.

See an example solution in
[pets/solution_centralized.py](pets/solution_centralized.py).

## Federated Code Submission
