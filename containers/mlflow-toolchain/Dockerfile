#
#    Copyright 2021 EPAM Systems
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#

FROM continuumio/miniconda3:4.10.3

ARG ODAHUFLOW_DIR=/opt/odahu-flow

# nvidia CUDA needed for tensorflow-gpu
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PATH="/usr/local/nvidia/bin:/opt/conda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}"

RUN apt-get update --fix-missing && \
    apt-get install -y wget git bzip2 ca-certificates \
            # numpy install requirements
            gfortran && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY mlflow/requirements.txt "${ODAHUFLOW_DIR}/"
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r "${ODAHUFLOW_DIR}/requirements.txt"

COPY mlflow/ "${ODAHUFLOW_DIR}/"
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    python3 -m pip install -e "${ODAHUFLOW_DIR}"

CMD [ "/bin/bash" ]
