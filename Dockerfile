FROM python:3.9

WORKDIR /

COPY . ./

RUN apt-get update -y && \ 
  apt-get install apt-utils pandoc -y && \
  apt-get install libopenmpi-dev cmake libz-dev build-essential libssl-dev libffi-dev python-dev -y

RUN pip install --upgrade pip


RUN pip install --upgrade setuptools cython
#RUN pip install numpy>=1.19.2 pandas>=1.0.4 cvxpy>=1.1.15 statsmodels>=0.10.1 tensorflow>=2.6.0 matplotlib>=3.3.3
#RUN pip install pipreqs && pipreqs ./tensortrade

#RUN pip install -r ./requirements.txt
#RUN pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl
#RUN pip install -e .[docs,tests]

RUN pip install -e .[tensorforce,baselines,fbm]
#RUN pip install tensortrade[tf,tensorforce,baselines,ccxt,fbm]@git+https://github.com/lakhwindersingh/tensortrade.git

RUN pip install -r ./examples/requirements.txt

# Setup for Jupyter Notebook
RUN groupadd -g 1000 jupyter && \
    useradd -g jupyter -m -s /bin/bash jupyter && \
    mkdir -p /etc/sudoers.d /root/.jupyter  /home/jupyter/.jupyter /home/jupyter/notebook && \
    echo "jupyter:jupyter" | chpasswd && \
    echo "jupyter ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/jupyter && \
    chmod -R a+rwX  /etc/sudoers.d/jupyter /home/jupyter && \
    echo "/usr/lib" >  /etc/ld.so.conf.d/nbquant.conf && \
    echo "/usr/local/lib" >>  /etc/ld.so.conf.d/nbquant.conf && \
    #sudo mkdir /data && sudo chmod 777 /data && \
    ldconfig  && \
    echo "export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:$LD_LIBRARY_PATH" > /etc/profile.d/jupyter.sh && \
    echo "export PATH=/usr/local/bin:/usr/bin:$PATH" >> /etc/profile.d/jupyter.sh && \
    cp /etc/profile.d/jupyter.sh /root/.bashrc && \
    # Below file enable password access instead of token
    echo "c.NotebookApp.token = 'jupyter'" > /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.notebook_dir = '/home/jupyter/notebook/'" >> /root/.jupyter/jupyter_notebook_config.py && \
    cp /root/.jupyter/jupyter_notebook_config.py /home/jupyter/.jupyter

USER jupyter
RUN pip install --user jupyter jupyterlab

USER root
# Add shell script to start postfix and jupyter

COPY entrypoint.sh /usr/local/bin
RUN chmod +x /usr/local/bin/entrypoint.sh
EXPOSE 8282 9000 443
WORKDIR /home/jupyter/
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD []