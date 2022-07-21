#!/bin/bash

# Run Jupyterlab
#cd /home/jupyter/notebook
#sudo runuser -l jupyter -c "/opt/conda/bin/jupyter lab --port=9999 --ip=0.0.0.0 --no-browser"
#jupyter-lab
#sudo runuser -l jupyter -c "/usr/local/bin/jupyter lab --allow-root --port=8282 --ip=0.0.0.0 --no-browser"
sudo runuser -l jupyter -c "jupyter lab --ip='*' --NotebookApp.token='' --NotebookApp.password='' --allow-root --port=8282  --no-browser"

