# U.S. PETs Prize Challenge: Phase 2 (Pandemic Forecasting–Federated)

## Code layout and implementation

Our solution introduces **_MusCAT_**, a multi-scale federated system
for privacy-preserving pandemic risk prediction. It is implemented in Python and Go.

- Centralized solution uses Python:

  - [solution_centralized.py](solution_centralized.py) represents the entrypoint to the solution. It defines the main functions required by the framework (e.g., fit()) and implements MusCat's general workflow.

  - [muscat_model.py](muscat_model.py) constructs the MusCAT model and defines each step of MusCat's workflow (called by solution_centralize.py). 

- Federated solution uses both Python and Go. The latter is needed for cryptographic operations, and uses a custom fork of [tuneinsight/lattigo](https://github.com/tuneinsight/lattigo) library for lattice-based homomorphic encryption.

  - [solution_federated.py](solution_federated.py) represents the entrypoint to the solution. It defines the main functions required by the framework (e.g., fit(), configure_fit()...) and implements MusCat's general federated workflow.
      - fit() in class TrainClient implements the core of our model training, executed by the clients, with the computation of global statistics (W0-W3 in Section 3.4 of our manuscript) and the poisson regression (W4). 
      - aggregate_fit() in class TrainStrategy defines the operations of the server, i.e., securely aggregating encrypted information for the collaboration among the clients.
      - fit() and evaluate() in class TestClient implement the clients' part of the inference (W5 in our manuscript)
      - configure_fit and aggregate_fit() define the server functions for the same operations
      

  - [muscat_model.py](muscat_model.py) constructs the MusCAT model and defines each step of MusCat's federated workflow (called by solution_federated.py).

  - [muscat_privacy.py](muscat_privacy.py) contains static parameters and functions specific for
    Differential Privacy (DP)

  - [muscat_workflow.py](muscat_workflow.py) contains static parameters for
    the secure and plaintext training and testing workflows. It notably defines the training parameters and the order of the rounds to train a model. 

  - [mhe_routines.go](mhe_routines.go) represents the Go entrypoint that
    parses command-line arguments passed to it from Python, and executes
    a computation corresponding to its step in the Python workflow.
    This takes the form:

    ```sh
    petchal <command> <arg1> [<arg2> ...]
    ```

    where `<command>` designates a step in the workflow,
    and `<arg1> [<arg2> ...]` represents various parameters, which
    specify either path(s) to the data directory(s), or numeric parameters. It currently enables the setup of the cryptographic parameters and the execution of the *Collective Aggregation and Decryption* (defined in multiple operations for the clients and for the server).

  - [mhe/crypto.go](mhe/crypto.go) contains cryptographic utilities
    for Multiparty Homomorphic Encryption (MHE, e.g., vectors encryption and decryption), along with some functions
    to handle disk I/O (e.g., to save and read cryptographic parameters and keys), which is needed for passing data from/to Python.

  - [mhe/protocols.go](mhe/protocols.go) provides high-level functions
    that implement disk-assisted client-server communication protocol

  - [mhe/utilities.go](mhe/utilities.go) contains auxiliary utilities,
    including functions to (de)serialize data vectors and matrices
    from/to disk, in order to pass them from/to Python

  - [go.mod](go.mod) and [go.sum](go.sum) configure third-party Go
    dependencies.

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
