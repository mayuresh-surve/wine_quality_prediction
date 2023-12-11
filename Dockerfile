FROM centos:7

RUN yum -y update && yum -y install python3 python3-dev python3-pip python3-virtualenv \
	java-1.8.0-openjdk wget

RUN python -V
RUN python3 -V

ENV PYSPARK_DRIVER_PYTHON python3
ENV PYSPARK_PYTHON python3

RUN pip3 install --upgrade pip
RUN pip3 install numpy panda
RUN pip3 install pyspark

RUN wget https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
RUN tar -xzf spark-3.5.0-bin-hadoop3.tgz && mv spark-3.5.0-bin-hadoop3 spark
RUN rm spark-3.5.0-bin-hadoop3.tgz

RUN (echo 'export SPARK_HOME=/spark' >> ~/.bashrc && echo 'export PATH=$SPARK_HOME/bin:$PATH' >> ~/.bashrc && echo 'export PYSPARK_PYTHON=python3' >> ~/.bashrc)

RUN mkdir /wineapp

WORKDIR /wineapp

COPY wine_prediction_docker.py /wineapp/ 
COPY ValidationDataset.csv /wineapp/
ADD models/ /wineapp/

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN /bin/bash -c "source ~/.bashrc"
RUN /bin/sh -c "source ~/.bashrc"

ENTRYPOINT ["/spark/bin/spark-submit", "--packages", "org.apache.hadoop:hadoop-aws:3.3.6", "wine_prediction_docker.py"]