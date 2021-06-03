FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY ml_env.yml .
RUN conda env create -f ml_env.yml 

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "ml_env", "/bin/bash", "-c"]

# Make sure the environment is activated:
RUN echo "Make sure flask is installed:"
RUN python -c "import flask"

# The code to run when container is started:
#COPY run.py .
COPY app.py .
COPY utils.py .
COPY transformers.py .
COPY test.csv .
COPY nyc_stacked_model.sav .
COPY catpipe.sav .
COPY numpipe.sav .
COPY scaler.sav .
COPY templates/ ./templates
COPY static/ ./static

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ml_env", "gunicorn",  "--bind", "0.0.0.0:5000", "app:app"]
