# U.S. PETs Prize Challenge: Phase 2 (Pandemic Forecasting–Federated)

## Code layout and implementation

Our solution introduces **_MusCAT_**, a multi-scale federated system
for privacy-preserving pandemic risk prediction. It is implemented in Python and Go.

1. Centralized solution uses Python.

   **_Outline the algorithm and the libraries used_**

2. Federated solution uses both Python and Go. The latter is needed for
   performance optimization, and uses the following high-performance Go libraries:

   - [hcholab/lattigo](https://github.com/hcholab/lattigo/tree/petschal), a custom fork of
     [tuneinsight/lattigo](https://github.com/tuneinsight/lattigo) library
     for lattice-based homomorphic encryption

   - [dedis/onet](https://github.com/dedis/onet), Overlay Network library
     for simulation and deployment of decentralized, distributed protocols

   **_Outline the algorithm and the libraries used_**

## Building

Our solution incorporates both Python and Go source code, along with the Go binary named `petchal`, which is pre-compiled from this code.

We suggest that this solution is executed as-is with these files, to ensure it works as tested in lab conditions.

### Rebuilding Go code

If you would rather like to reproduce the binary from Go code, you can do so using either:

- `go build` command, if you are on a **Linux** system with **Go 1.19** installed

- `docker run --rm -t --platform linux/amd64 -v "$PWD:/work" -w /work golang:1.19 go build`
  if you are on **macOS**, to ensure binary compatibility with the runtime Linux environment

Additionally, please use `CPU_OR_GPU=cpu` for the runtime.

### Local testing/development

The following instructions are _only_ needed for local testing/development,
and to (re)generate `submission.zip`:

- Clone the challenge repository:

  ```sh
  cd $(mktemp -d) # use a temporary (or an empty) directory
  git clone https://github.com/drivendataorg/pets-prize-challenge-runtime .
  ```

- Download the `va-*` data files into `data/pandemic/{centralized,federated}/{test,train}/` directories

  - one can avoid data duplication by creating hard links to the associated files, e.g.
    ```sh
    ls /path/to/downloaded/data/*.{gz,csv} | while read f ; do
      for p in "centralized/test" "centralized/train" \
            "scenario01/test/client01" "scenario01/train/client01" \
            "scenario01/test/client02" "scenario01/train/client02" \
            "scenario01/test/client03" "scenario01/train/client03" ; do
        sudo ln -f "$f" "data/pandemic/$p/$(basename $f)"
      done
    done
    ```

- Build and run test submission:

  ```sh
  # Download the official challenge Docker image
  make pull

  # Prepare runtime variables
  export CPU_OR_GPU=cpu
  export SUBMISSION_TRACK=pandemic
  export SUBMISSION_TYPE=centralized # or federated

  # (Re)build the Go code directly, if you're on a Linux system with Go 1.19 installed
  ( cd "submission_src/pandemic" &&  go build )

  # Alternatively, (re)generate the x86_64/amd64 binary if you're on macOS (with or without Apple Silicon)
  docker run --rm -t --platform linux/amd64 \
    -v "$PWD/submission_src/pandemic:/work" \
    -w /work golang:1.19 go build

  # Remove any previous submission archive
  rm -f submission/submission.zip

  # (Re)pack the submission code
  make pack-submission

  # Test the submission
  make test-submission
  ```
