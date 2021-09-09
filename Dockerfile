FROM continuumio/miniconda3:latest

RUN mkdir -p /workshop
ADD EpitopeWorkshop /workshop/EpitopeWorkshop

COPY environment.yml setup.py /workshop/
RUN conda env create -f /workshop/environment.yml

WORKDIR /workshop/EpitopeWorkshop
RUN conda run -n workshop pip install -e /workshop

SHELL ["conda", "run", "-n", "workshop", "/bin/bash", "-c"]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "workshop", "python", "main.py"]
