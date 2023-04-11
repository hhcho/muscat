# MusCAT

This repository includes software for Team MusCAT's solution to the U.S. PETs Prize Challenge Phase 2 (Pandemic Forecasting). 

## Problem Setting

Predictive models that can assess an individual's risk of infection in a given day can be a useful tool for pandemic response. Training these models at scale can be challenging due to difficulties in sharing personally identifying information across administrative or organizational boundaries. The Challenge task was to develop a federated learning system that can jointly leverage a distributed set of private datasets, including a broad range of information about the individuals (e.g., demographics, daily activities, and social contacts), to train and apply risk prediction models while rigorously protecting the privacy of individuals in the dataset. See [here](https://www.drivendata.org/competitions/103/nist-federated-learning-2-pandemic-forecasting-federated/) for the official Challenge page introducing the problem.

## Our Technical Approach

We introduce MusCAT, a multi-scale federated system for privacy-preserving pandemic risk prediction. We leverage the key insight that predictive information can be divided into components operating at different scales of the problem, such as individual contacts, shared locations, and population-level risks. These components are individually learned using a combination of privacy-enhancing technologies to best optimize the tradeoff between privacy and model accuracy. Based on the Challenge dataset, we show that our solution enables improved risk prediction with formal privacy guarantees, while maintaining practical runtimes even with many federation units.

![Graphical illustration of MusCAT](https://www.dropbox.com/s/s7tgd1unyk7pbw6/PETsChallenge_MusCAT_Graphic.png?dl=0)

Our white paper describing the solution is available [here](https://www.dropbox.com/s/dzyc8himjtcu05j/PETsChallenge_MusCAT_Report.pdf?dl=0)

## Software Components and Methodology

Our solution is implemented in Python and Go.

- Centralized solution uses Python:

  - [solution_centralized.py](solution_centralized.py) represents the entrypoint to the solution. It defines the main functions required by the framework (`fit()` and `predict()`) and implements MusCAT's general workflow, similar to the one described in _Section 3.4 (Privacy-Preserving Federated System for Individual Risk Prediction)_ of our manuscript.

  - [muscat_model.py](muscat_model.py) constructs the MusCAT model and defines each step of MusCAT's workflow (called by [solution_centralized.py](solution_centralized.py)). See _Sections 3.4 and 5.1 (Centralized performance)_ for discussions of the workflow and its benchmarks.

- Federated solution uses both Python and Go. The latter is needed for cryptographic operations, and uses a [custom fork](https://github.com/hcholab/lattigo/tree/petschal) of [Lattigo](https://github.com/tuneinsight/lattigo) library for lattice-based homomorphic encryption, as discussed in _Section 5.2 (Federated Performance → Implementation Details)_.

  - [solution_federated.py](solution_federated.py) represents the entrypoint to the solution. It defines the main functions required by the framework (e.g., `fit()`, `configure_fit()`, ...) and implements MusCAT's general federated workflow.

    - `fit()` in class `TrainClient` implements the core of our model training, executed by the clients, with the computation of global statistics (**W0-W3** in _Section 3.4_) and the Poisson regression (**W4**).
    - `aggregate_fit()` in class `TrainStrategy` defines the operations of the server, i.e., securely aggregating encrypted information for the collaboration among the clients, as described in _Section 5 (Experimental Results)._
    - `fit()` and `evaluate()` in class `TestClient` implement the clients' part of the inference (**W6**)
    - `configure_fit()` and `aggregate_fit()` define the server functions for the same operations. See _Section 5._

  - [muscat_model.py](muscat_model.py) constructs the MusCAT model and defines each step of MusCAT's federated workflow (called by [solution_federated.py](solution_federated.py)), as described in _Section 3.4_.

  - [muscat_privacy.py](muscat_privacy.py) contains static parameters and functions specific for Differential Privacy (DP). See _Sections 3.4, 4 (Privacy Analysis → DP Training),_ and _5.2 (Federated Performance → Privacy)_ for a discussion of DP, its implementation and performance.

  - [dpmean.py](dpmean.py) provides `multivariate_mean_iterative()` that implements CoinPress algorithm for private mean estimation (called by [solution_federated.py](solution_federated.py)), as described in _section 5.2_ _(Federated Performance → Privacy)_.

  - [muscat_workflow.py](muscat_workflow.py) contains static parameters for
    the secure and plaintext training and testing workflows. It notably defines the training parameters and the order of the rounds to train a model. See _Section 3.4_ on the workflow.

  - [mhe_routines.go](mhe_routines.go) represents the Go entrypoint that
    parses command-line arguments passed to it from Python, and executes
    a computation corresponding to its step in the Python workflow.
    This takes the form:

    ```sh
    muscat <command> <arg1> [<arg2> ...]
    ```

    where `<command>` designates a step in the workflow,
    and `<arg1> [<arg2> ...]` represents various arguments, which
    specify either path(s) to the data directory(s), or numeric parameters. It currently enables the setup of the cryptographic parameters and the execution of the _Collective Aggregation and Decryption_ (used during MusCAT's workflow for secure aggregation of the clients' local results by the server), as discussed in _Sections 3.4 and 5_.

  - [mhe/crypto.go](mhe/crypto.go) contains cryptographic utilities
    for Multiparty Homomorphic Encryption (MHE, e.g., vectors encryption and decryption), along with some functions to handle disk I/O (e.g., to save and read cryptographic parameters and keys), which is needed for passing data from/to Python. See Section _5.2.1 (Efficiency & Scalability → MHE Operations)_ on the use of these cryptographic primitives.

  - [mhe/protocols.go](mhe/protocols.go) provides high-level functions
    that implement disk-assisted client-server communication protocol. See _Section 5.2.1 (Efficiency & Scalability)_ for this protocol implementation.

  - [mhe/utilities.go](mhe/utilities.go) contains auxiliary utilities,
    including functions to (de)serialize data vectors and matrices
    from/to disk, in order to pass them from/to Python. See _Section 5.2.1_ for relevant discussions.

  - [go.mod](go.mod) and [go.sum](go.sum) configure third-party Go
    dependencies.

## Running

1. Download and partition the `va-*` data files as described in
   the `pandemic-partitioning-example.ipynb` notebook
   on the [Data Download page](https://www.drivendata.org/competitions/103/nist-federated-learning-2-pandemic-forecasting-federated/data/).

2. Install Docker and run the following command:
   ```sh
   export SUBMISSION_TYPE=centralized # or federated
   docker run --rm -it \
     --env SUBMISSION_TRACK=${SUBMISSION_TYPE} \
     --mount type=bind,source="$(pwd)"/data/${SUBMISSION_TYPE},target=/code_execution/data,readonly \
     ghcr.io/hhcho/muscat ${SUBMISSION_TYPE}
   ```

## Local testing/development

```sh
docker build --platform linux/amd64 -t muscat .
```
