FROM nvidia/cuda:11.0.3-base-ubuntu20.04 AS runtime
# adapated from build file for pangeo images
# https://github.com/pangeo-data/pangeo-docker-images

ENV CONDA_VERSION=4.10.3-2 \
    CONDA_ENV=condaenv \
    RUNTIME_USER=appuser \
    RUNTIME_UID=1000 \
    SHELL=/bin/bash \
    LANG=C.UTF-8  \
    LC_ALL=C.UTF-8 \
    CONDA_DIR=/opt/conda \
    DEBIAN_FRONTEND=noninteractive

ENV HOME=/home/${RUNTIME_USER} \
    PATH=${CONDA_DIR}/bin:${PATH}

# ======================== root ========================
# initialize paths we will use
RUN mkdir -p /code_execution

# Create appuser user, permissions, add conda init to startup script
RUN echo "Creating ${RUNTIME_USER} user..." \
    && groupadd --gid ${RUNTIME_UID} ${RUNTIME_USER}  \
    && useradd --create-home --gid ${RUNTIME_UID} --no-log-init --uid ${RUNTIME_UID} ${RUNTIME_USER} \
    && echo ". ${CONDA_DIR}/etc/profile.d/conda.sh ; conda activate ${CONDA_ENV}" > /etc/profile.d/init_conda.sh \
    && chown -R ${RUNTIME_USER}:${RUNTIME_USER} /opt /code_execution

# Install base packages
ARG DEBIAN_FRONTEND=noninteractive
COPY runtime/apt.txt /home/${RUNTIME_USER}
RUN rm -f /etc/apt/sources.list.d/cuda.list && rm -f /etc/apt/sources.list.d/nvidia-ml.list
RUN echo "Installing base packages..." \
    && apt-get update --fix-missing \
    && apt-get install -y apt-utils 2> /dev/null \
    && apt-get install -y wget zip tzdata \
    && apt-get install -y libgl1-mesa-glx \
    && xargs -a /home/${RUNTIME_USER}/apt.txt apt-get install -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /home/${RUNTIME_USER}/apt.txt
# ======================================================

# ======================== user ========================
USER ${RUNTIME_USER}

# Install conda
RUN echo "Installing Miniforge..." \
    && URL="https://github.com/conda-forge/miniforge/releases/download/${CONDA_VERSION}/Miniforge3-${CONDA_VERSION}-Linux-x86_64.sh" \
    && wget --quiet ${URL} -O /home/${RUNTIME_USER}/miniconda.sh \
    && /bin/bash /home/${RUNTIME_USER}/miniconda.sh -u -b -p ${CONDA_DIR} \
    && rm /home/${RUNTIME_USER}/miniconda.sh \
    && conda install -y -c conda-forge mamba \
    && mamba clean -afy \
    && find ${CONDA_DIR} -follow -type f -name '*.a' -delete \
    && find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete

# Switch back to root for installing conda packages
COPY runtime/environment.yml /home/${RUNTIME_USER}/environment.yml
RUN export CONDA_OVERRIDE_CUDA= \
    && mamba env create --name ${CONDA_ENV} -f /home/${RUNTIME_USER}/environment.yml  \
    && mamba clean -afy \
    && conda run pip cache purge \
    && rm /home/${RUNTIME_USER}/environment.yml \
    && find ${CONDA_DIR} -follow -type f -name '*.a' -delete \
    && find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete \
    && find ${CONDA_DIR} -follow -type f -name '*.js.map' -delete

# Copy run script into working dir and set it as the working doie
WORKDIR /code_execution
RUN mkdir -p data predictions submission /home/${RUNTIME_USER}/.config/procps
COPY --chown=appuser:appuser runtime/tests /code_execution/tests
COPY --chown=appuser:appuser runtime/supervisor.py /code_execution/supervisor.py
COPY --chown=appuser:appuser runtime/main_federated_train.py /code_execution/main_federated_train.py
COPY --chown=appuser:appuser runtime/main_federated_test.py /code_execution/main_federated_test.py
COPY --chown=appuser:appuser runtime/post_federated.py /code_execution/post_federated.py
COPY --chown=appuser:appuser runtime/main_centralized_train.py /code_execution/main_centralized_train.py
COPY --chown=appuser:appuser runtime/main_centralized_test.py /code_execution/main_centralized_test.py
COPY --chown=appuser:appuser runtime/post_centralized.py /code_execution/post_centralized.py
COPY --chown=appuser:appuser runtime/entrypoint.sh /code_execution/entrypoint.sh
COPY --chown=appuser:appuser runtime/toprc /home/${RUNTIME_USER}/.config/procps/toprc
COPY --chown=appuser:appuser src/*.py /code_execution/src/

# Build Go package
FROM golang:1.19 AS go

WORKDIR /src
COPY src/ .
RUN go build

# Build the final image
FROM runtime AS final

ENV SUBMISSION_TRACK=pandemic
COPY --chown=appuser:appuser --from=go /src/muscat /code_execution/src/

ENTRYPOINT ["/bin/bash", "/code_execution/entrypoint.sh"]

# Test the final image
FROM final

RUN conda run --no-capture-output -n condaenv python -m pytest tests

# Use the final image
FROM final
