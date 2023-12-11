# Wine quality prediction

### Create S3 bucket
1. Under AWS dashboard select s3 service
2. Click on **Create bucket**
3. Give name to your bucket and scroll down to click on **Create bucket**

### Create EMR cluster
1. Go to AWS dashboard
2. Search/Select EMR service
3. Click on **Create Cluster**
4. Set name for the cluster
5. Under Application bundle select **Spark Interactive**
6. Under Cluster provisioning and scaling increase the instance size of the Task-1 to 4.
7. Under cluster logs give path to the S3 bucket you have created. (Example path is **s3://{bucket-name}/logs**)
8. Under security configuration create new key-pair.
9. Under Identity and access management choose **EMR_DefaultRole** and **EMR_EC2_DefaultRole**
10. Click on the **Create cluster** to create new cluster
11. Go to the EC2 dashboard search for the master EC2 instance
12. For the master EC2 instance edit inbound security rules to allow SSH.

### Transfer your files to the EMR
1. From the EMR dashboard click on the cluster you have created.
2. Copy the **Primary node public DNS**.
3. On your local machine navigate to the folder where you have store key-pair for the EMR. 
4. Run the command `sftp -i <your-pem-file-name> hadoop@<your-primary-node-public-DNS>`
5. Now run the command `put [file-name]` for each file you want to add to the EMR. 

### SSH to the EMR and run training
1. Run the command `ssh -i <your-pem-file-name> hadoop@<your-primary-node-public-DNS>`
2. This will log you into the EMR cluster created.
3. If you run the command `ls` you should be able to see all the files you copied from the local machine. 
4. Run the following command to copy files from master node to HDFS
    - `hadoop fs -put \<file-name> /user/hadoop/\<file-name>`
5. Repeat this for each file
6. Launch training model using command `spark-submit wine_training.py`
7. This will store ML model to the s3 bucket. 

### Run the prediction without docker
1. ssh to the EMR cluster
2. run the command `spark-submit wine_prediction.py --test_file ./ValidationDataset.csv`

### Run the prediction with docker
1. sudo systemctl start docker
2. sudo systemctl enable docker
3. sudo systemctl status docker 
4. docker pull browdex/wine-prediction
5. docker run browdex/wine-prediction