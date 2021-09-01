#
#    Copyright 2019 EPAM Systems
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

FROM python:3.7.6

ARG MLFLOW_VERSION=1.20.1
ARG PSYCOPG2_VERSION=2.8.6
ENV MLFLOW_ARTIFACT_ROOT=/mlruns 
ENV MLFLOW_BACKEND_STORE=/mlruns

RUN pip install --no-cache-dir mlflow==${MLFLOW_VERSION} psycopg2==${PSYCOPG2_VERSION} google-cloud-storage azure-storage-blob

RUN mkdir -p ${MLFLOW_ARTIFACT_ROOT}

EXPOSE 5000

ENTRYPOINT ["bash", "-c"]
CMD ["mlflow server --backend-store-uri ${MLFLOW_BACKEND_STORE} --default-artifact-root ${MLFLOW_ARTIFACT_ROOT} --host 0.0.0.0 -p 5000"]
