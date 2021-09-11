FROM continuumio/miniconda3:latest as build

COPY environment.yml .
RUN conda env create -f environment.yml

# Install conda-pack:
RUN conda install -c conda-forge conda-pack

# Use conda-pack to create a standalone enviornment
# in /venv:
RUN conda-pack -n workshop -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
  rm /tmp/env.tar

# We've put venv in same path it'll be in final image,
# so now fix up paths:
RUN /venv/bin/conda-unpack

FROM debian:buster AS runtime

# Copy /venv from the previous stage:
COPY --from=build /venv /venv

RUN mkdir -p /workshop /workshop/data/cnn-models/ /workshop/data/heat-maps/
ADD EpitopeWorkshop /workshop/EpitopeWorkshop
COPY entry.sh setup.py /workshop/
ENV CNN_DIR /workshop/data/cnn-models/
ENV HEAT_MAP_DIR /workshop/data/heat-maps/
COPY data/cnn.pth /workshop/data/cnn-models/

RUN cd /workshop && /bin/bash -c "source /venv/bin/activate && python setup.py develop"
WORKDIR /workshop/EpitopeWorkshop

# When image is run, run the code with the environment
# activated:
SHELL ["/bin/bash"]
ENTRYPOINT /workshop/entry.sh