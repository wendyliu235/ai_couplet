```python
# import libraries
import boto3, re, sys, math, json, os, sagemaker, urllib.request
from sagemaker import get_execution_role
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.display import display
from time import gmtime, strftime
from sagemaker.predictor import csv_serializer

# Define IAM role
role = get_execution_role()
prefix = 'sagemaker/DEMO-xgboost-dm'
my_region = boto3.session.Session().region_name # set the region of the instance

# this line automatically looks for the XGBoost image URI and builds an XGBoost container.
xgboost_container = sagemaker.image_uris.retrieve("xgboost", my_region, "latest")

print("Success - the MySageMakerInstance is in the " + my_region + " region. You will use the " + xgboost_container + " container for your SageMaker endpoint.")
```

    Success - the MySageMakerInstance is in the us-west-2 region. You will use the 433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest container for your SageMaker endpoint.



```python
bucket_name = 'darkalexwang-s3-bucket-mysagemaker0-tutorial' # <--- CHANGE THIS VARIABLE TO A UNIQUE NAME FOR YOUR BUCKET
s3 = boto3.resource('s3')
try:
    if  my_region == 'us-east-1':
      s3.create_bucket(Bucket=bucket_name)
    else: 
      s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={ 'LocationConstraint': my_region })
    print('S3 bucket created successfully')
except Exception as e:
    print('S3 error: ',e)
```

    S3 bucket created successfully



```python
try:
  urllib.request.urlretrieve ("https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv", "bank_clean.csv")
  print('Success: downloaded bank_clean.csv.')
except Exception as e:
  print('Data load error: ',e)

try:
  model_data = pd.read_csv('./bank_clean.csv',index_col=0)
  print('Success: Data loaded into dataframe.')
except Exception as e:
    print('Data load error: ',e)
```

    Success: downloaded bank_clean.csv.
    Success: Data loaded into dataframe.



```python
train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
print(train_data.shape, test_data.shape)
```

    (28831, 61) (12357, 61)



```python
pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('train.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')
```


```python
sess = sagemaker.Session()
xgb = sagemaker.estimator.Estimator(xgboost_container,role, instance_count=1, instance_type='ml.m4.xlarge',output_path='s3://{}/{}/output'.format(bucket_name, prefix),sagemaker_session=sess)
xgb.set_hyperparameters(max_depth=5,eta=0.2,gamma=4,min_child_weight=6,subsample=0.8,silent=0,objective='binary:logistic',num_round=100)
```


```python
xgb.fit({'train': s3_input_train})
```

    2021-11-08 02:20:21 Starting - Starting the training job...
    2021-11-08 02:20:46 Starting - Launching requested ML instancesProfilerReport-1636338020: InProgress
    ......
    2021-11-08 02:21:40 Starting - Preparing the instances for training.........
    2021-11-08 02:23:21 Downloading - Downloading input data...
    2021-11-08 02:23:41 Training - Downloading the training image..[34mArguments: train[0m
    [34m[2021-11-08:02:23:59:INFO] Running standalone xgboost training.[0m
    [34m[2021-11-08:02:23:59:INFO] Path /opt/ml/input/data/validation does not exist![0m
    [34m[2021-11-08:02:23:59:INFO] File size need to be processed in the node: 3.38mb. Available memory size in the node: 8355.47mb[0m
    [34m[2021-11-08:02:23:59:INFO] Determined delimiter of CSV input is ','[0m
    [34m[02:23:59] S3DistributionType set as FullyReplicated[0m
    [34m[02:24:00] 28831x59 matrix with 1701029 entries loaded from /opt/ml/input/data/train?format=csv&label_column=0&delimiter=,[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 30 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[0]#011train-error:0.100482[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 30 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[1]#011train-error:0.099858[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 28 extra nodes, 22 pruned nodes, max_depth=5[0m
    [34m[2]#011train-error:0.099754[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[3]#011train-error:0.099095[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 28 extra nodes, 12 pruned nodes, max_depth=5[0m
    [34m[4]#011train-error:0.098991[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 32 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[5]#011train-error:0.099303[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 24 extra nodes, 18 pruned nodes, max_depth=5[0m
    [34m[6]#011train-error:0.099684[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 30 extra nodes, 22 pruned nodes, max_depth=5[0m
    [34m[7]#011train-error:0.09906[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 28 extra nodes, 20 pruned nodes, max_depth=5[0m
    [34m[8]#011train-error:0.098852[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 36 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[9]#011train-error:0.098679[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 20 extra nodes, 16 pruned nodes, max_depth=5[0m
    [34m[10]#011train-error:0.098748[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 28 extra nodes, 24 pruned nodes, max_depth=5[0m
    [34m[11]#011train-error:0.098748[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 20 extra nodes, 22 pruned nodes, max_depth=5[0m
    [34m[12]#011train-error:0.098748[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 30 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[13]#011train-error:0.09854[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 20 pruned nodes, max_depth=5[0m
    [34m[14]#011train-error:0.098574[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 24 extra nodes, 20 pruned nodes, max_depth=5[0m
    [34m[15]#011train-error:0.098609[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 32 extra nodes, 16 pruned nodes, max_depth=5[0m
    [34m[16]#011train-error:0.098817[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 18 extra nodes, 16 pruned nodes, max_depth=5[0m
    [34m[17]#011train-error:0.098817[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 18 pruned nodes, max_depth=5[0m
    [34m[18]#011train-error:0.098679[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 30 extra nodes, 16 pruned nodes, max_depth=5[0m
    [34m[19]#011train-error:0.098679[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 20 extra nodes, 12 pruned nodes, max_depth=5[0m
    [34m[20]#011train-error:0.098713[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 30 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[21]#011train-error:0.098505[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 30 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[22]#011train-error:0.098401[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 26 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[23]#011train-error:0.098332[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 34 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[24]#011train-error:0.098332[0m
    [34m[02:24:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 24 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[25]#011train-error:0.09795[0m
    [34m[02:24:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 20 extra nodes, 12 pruned nodes, max_depth=5[0m
    [34m[26]#011train-error:0.098262[0m
    [34m[02:24:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 24 pruned nodes, max_depth=5[0m
    [34m[27]#011train-error:0.098193[0m
    [34m[02:24:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 8 extra nodes, 24 pruned nodes, max_depth=3[0m
    [34m[28]#011train-error:0.097985[0m
    [34m[02:24:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 20 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[29]#011train-error:0.097499[0m
    [34m[02:24:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 28 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[30]#011train-error:0.097638[0m
    [34m[02:24:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 18 extra nodes, 22 pruned nodes, max_depth=5[0m
    [34m[31]#011train-error:0.097395[0m
    [34m[02:24:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 28 extra nodes, 24 pruned nodes, max_depth=5[0m
    [34m[32]#011train-error:0.097222[0m
    [34m[02:24:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 24 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[33]#011train-error:0.097118[0m
    [34m[02:24:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 20 pruned nodes, max_depth=5[0m
    [34m[34]#011train-error:0.097014[0m
    [34m[02:24:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 24 extra nodes, 18 pruned nodes, max_depth=5[0m
    [34m[35]#011train-error:0.09684[0m
    [34m[02:24:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[36]#011train-error:0.096667[0m
    [34m[02:24:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 20 extra nodes, 18 pruned nodes, max_depth=5[0m
    [34m[37]#011train-error:0.096736[0m
    [34m[02:24:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 30 extra nodes, 22 pruned nodes, max_depth=5[0m
    [34m[38]#011train-error:0.096563[0m
    [34m[02:24:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 24 pruned nodes, max_depth=5[0m
    [34m[39]#011train-error:0.096355[0m
    [34m[02:24:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 36 pruned nodes, max_depth=3[0m
    [34m[40]#011train-error:0.096285[0m
    [34m[02:24:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 12 extra nodes, 38 pruned nodes, max_depth=4[0m
    [34m[41]#011train-error:0.096528[0m
    [34m[02:24:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 8 extra nodes, 16 pruned nodes, max_depth=4[0m
    [34m[42]#011train-error:0.096355[0m
    [34m[02:24:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 26 pruned nodes, max_depth=5[0m
    [34m[43]#011train-error:0.096459[0m
    [34m[02:24:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 36 pruned nodes, max_depth=5[0m
    [34m[44]#011train-error:0.096355[0m
    [34m[02:24:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 4 extra nodes, 34 pruned nodes, max_depth=2[0m
    [34m[45]#011train-error:0.096216[0m
    [34m[02:24:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 24 extra nodes, 20 pruned nodes, max_depth=5[0m
    [34m[46]#011train-error:0.096077[0m
    [34m[02:24:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 18 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[47]#011train-error:0.0958[0m
    [34m[02:24:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[48]#011train-error:0.095904[0m
    [34m[02:24:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 16 pruned nodes, max_depth=5[0m
    [34m[49]#011train-error:0.095904[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 34 pruned nodes, max_depth=4[0m
    [34m[50]#011train-error:0.095834[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 16 pruned nodes, max_depth=5[0m
    [34m[51]#011train-error:0.095765[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[52]#011train-error:0.095904[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[53]#011train-error:0.095834[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 18 extra nodes, 22 pruned nodes, max_depth=5[0m
    [34m[54]#011train-error:0.095834[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 22 pruned nodes, max_depth=5[0m
    [34m[55]#011train-error:0.09573[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 30 pruned nodes, max_depth=5[0m
    [34m[56]#011train-error:0.095626[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[57]#011train-error:0.095696[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 28 pruned nodes, max_depth=5[0m
    [34m[58]#011train-error:0.095661[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 24 pruned nodes, max_depth=4[0m
    [34m[59]#011train-error:0.095592[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 24 extra nodes, 16 pruned nodes, max_depth=5[0m
    [34m[60]#011train-error:0.095522[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 16 pruned nodes, max_depth=4[0m
    [34m[61]#011train-error:0.095383[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 8 extra nodes, 20 pruned nodes, max_depth=4[0m
    [34m[62]#011train-error:0.095314[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 26 pruned nodes, max_depth=5[0m
    [34m[63]#011train-error:0.095661[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 8 extra nodes, 32 pruned nodes, max_depth=4[0m
    [34m[64]#011train-error:0.095661[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 32 extra nodes, 18 pruned nodes, max_depth=5[0m
    [34m[65]#011train-error:0.095418[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 36 pruned nodes, max_depth=3[0m
    [34m[66]#011train-error:0.095314[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[67]#011train-error:0.095349[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 6 extra nodes, 28 pruned nodes, max_depth=3[0m
    [34m[68]#011train-error:0.095314[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 0 extra nodes, 28 pruned nodes, max_depth=0[0m
    [34m[69]#011train-error:0.095314[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 32 pruned nodes, max_depth=3[0m
    [34m[70]#011train-error:0.095383[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 12 pruned nodes, max_depth=5[0m
    [34m[71]#011train-error:0.095453[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 12 pruned nodes, max_depth=5[0m
    [34m[72]#011train-error:0.095349[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 8 extra nodes, 24 pruned nodes, max_depth=4[0m
    [34m[73]#011train-error:0.095245[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[74]#011train-error:0.095175[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 8 extra nodes, 18 pruned nodes, max_depth=4[0m
    [34m[75]#011train-error:0.095071[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 22 pruned nodes, max_depth=5[0m
    [34m[76]#011train-error:0.095175[0m
    [34m[02:24:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 18 extra nodes, 30 pruned nodes, max_depth=5[0m
    [34m[77]#011train-error:0.095002[0m
    [34m[02:24:03] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 20 pruned nodes, max_depth=4[0m
    [34m[78]#011train-error:0.095037[0m
    [34m[02:24:03] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 32 pruned nodes, max_depth=5[0m
    [34m[79]#011train-error:0.095037[0m
    [34m[02:24:03] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 24 pruned nodes, max_depth=5[0m
    [34m[80]#011train-error:0.095002[0m
    [34m[02:24:03] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 20 pruned nodes, max_depth=5[0m
    [34m[81]#011train-error:0.094794[0m
    [34m[02:24:03] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 24 pruned nodes, max_depth=5[0m
    [34m[82]#011train-error:0.094759[0m
    [34m[02:24:03] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 20 extra nodes, 20 pruned nodes, max_depth=5[0m
    [34m[83]#011train-error:0.094933[0m
    [34m[02:24:03] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 20 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[84]#011train-error:0.09469[0m
    [34m[02:24:03] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 22 pruned nodes, max_depth=5[0m
    [34m[85]#011train-error:0.094759[0m
    [34m[02:24:03] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 24 pruned nodes, max_depth=5[0m
    [34m[86]#011train-error:0.094482[0m
    [34m[02:24:03] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 16 pruned nodes, max_depth=5[0m
    [34m[87]#011train-error:0.094447[0m
    [34m[02:24:03] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 0 extra nodes, 20 pruned nodes, max_depth=0[0m
    [34m[88]#011train-error:0.094482[0m
    [34m[02:24:03] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 24 pruned nodes, max_depth=5[0m
    [34m[89]#011train-error:0.094378[0m
    [34m[02:24:03] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 0 extra nodes, 28 pruned nodes, max_depth=0[0m
    [34m[90]#011train-error:0.094343[0m
    [34m[02:24:03] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 8 extra nodes, 28 pruned nodes, max_depth=4[0m
    [34m[91]#011train-error:0.094274[0m
    [34m[02:24:03] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 20 pruned nodes, max_depth=5[0m
    [34m[92]#011train-error:0.094239[0m
    [34m[02:24:03] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 32 pruned nodes, max_depth=5[0m
    [34m[93]#011train-error:0.094169[0m
    [34m[02:24:03] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 28 pruned nodes, max_depth=5[0m
    [34m[94]#011train-error:0.094169[0m
    [34m[02:24:03] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[95]#011train-error:0.094204[0m
    [34m[02:24:03] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 0 extra nodes, 28 pruned nodes, max_depth=0[0m
    [34m[96]#011train-error:0.094204[0m
    [34m[02:24:03] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 26 extra nodes, 20 pruned nodes, max_depth=5[0m
    [34m[97]#011train-error:0.093927[0m
    [34m[02:24:03] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 0 extra nodes, 38 pruned nodes, max_depth=0[0m
    [34m[98]#011train-error:0.093927[0m
    [34m[02:24:03] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 0 extra nodes, 32 pruned nodes, max_depth=0[0m
    [34m[99]#011train-error:0.093892[0m
    
    2021-11-08 02:24:21 Uploading - Uploading generated training model
    2021-11-08 02:24:21 Completed - Training job completed
    Training seconds: 61
    Billable seconds: 61



```python
xgb_predictor = xgb.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge')
```

    ------!


```python
from sagemaker.serializers import CSVSerializer

test_data_array = test_data.drop(['y_no', 'y_yes'], axis=1).values #load the data into an array
xgb_predictor.serializer = CSVSerializer() # set the serializer type
predictions = xgb_predictor.predict(test_data_array).decode('utf-8') # predict!
predictions_array = np.fromstring(predictions[1:], sep=',') # and turn the prediction into an array
print(predictions_array.shape)
```

    (12357,)



```python
cm = pd.crosstab(index=test_data['y_yes'], columns=np.round(predictions_array), rownames=['Observed'], colnames=['Predicted'])
tn = cm.iloc[0,0]; fn = cm.iloc[1,0]; tp = cm.iloc[1,1]; fp = cm.iloc[0,1]; p = (tp+tn)/(tp+tn+fp+fn)*100
print("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
print("{0:<15}{1:<15}{2:>8}".format("Predicted", "No Purchase", "Purchase"))
print("Observed")
print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format("No Purchase", tn/(tn+fn)*100,tn, fp/(tp+fp)*100, fp))
print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format("Purchase", fn/(tn+fn)*100,fn, tp/(tp+fp)*100, tp))
```

    
    Overall Classification Rate: 89.5%
    
    Predicted      No Purchase    Purchase
    Observed
    No Purchase    90% (10769)    37% (167)
    Purchase        10% (1133)     63% (288) 
    



```python
xgb_predictor.delete_endpoint(delete_endpoint_config=True)
```


```python
bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
bucket_to_delete.objects.all().delete()
```




    [{'ResponseMetadata': {'RequestId': 'J91JZNDKTXP5XGY7',
       'HostId': 'V7Ff/xhUxQtPmhldpzWbjIjYt6u8uITF0TWCQ5sOQ3StJ06ydwlzjZgKI0hkUoF9GgzkRWgvgZM=',
       'HTTPStatusCode': 200,
       'HTTPHeaders': {'x-amz-id-2': 'V7Ff/xhUxQtPmhldpzWbjIjYt6u8uITF0TWCQ5sOQ3StJ06ydwlzjZgKI0hkUoF9GgzkRWgvgZM=',
        'x-amz-request-id': 'J91JZNDKTXP5XGY7',
        'date': 'Mon, 08 Nov 2021 02:31:57 GMT',
        'content-type': 'application/xml',
        'transfer-encoding': 'chunked',
        'server': 'AmazonS3',
        'connection': 'close'},
       'RetryAttempts': 0},
      'Deleted': [{'Key': 'sagemaker/DEMO-xgboost-dm/output/xgboost-2021-11-08-02-20-20-871/profiler-output/system/training_job_end.ts'},
       {'Key': 'sagemaker/DEMO-xgboost-dm/output/xgboost-2021-11-08-02-20-20-871/rule-output/ProfilerReport-1636338020/profiler-output/profiler-reports/Dataloader.json'},
       {'Key': 'sagemaker/DEMO-xgboost-dm/output/xgboost-2021-11-08-02-20-20-871/rule-output/ProfilerReport-1636338020/profiler-output/profiler-reports/GPUMemoryIncrease.json'},
       {'Key': 'sagemaker/DEMO-xgboost-dm/output/xgboost-2021-11-08-02-20-20-871/output/model.tar.gz'},
       {'Key': 'sagemaker/DEMO-xgboost-dm/output/xgboost-2021-11-08-02-20-20-871/rule-output/ProfilerReport-1636338020/profiler-output/profiler-report.html'},
       {'Key': 'sagemaker/DEMO-xgboost-dm/output/xgboost-2021-11-08-02-20-20-871/rule-output/ProfilerReport-1636338020/profiler-output/profiler-reports/IOBottleneck.json'},
       {'Key': 'sagemaker/DEMO-xgboost-dm/output/xgboost-2021-11-08-02-20-20-871/rule-output/ProfilerReport-1636338020/profiler-output/profiler-reports/MaxInitializationTime.json'},
       {'Key': 'sagemaker/DEMO-xgboost-dm/output/xgboost-2021-11-08-02-20-20-871/rule-output/ProfilerReport-1636338020/profiler-output/profiler-reports/StepOutlier.json'},
       {'Key': 'sagemaker/DEMO-xgboost-dm/output/xgboost-2021-11-08-02-20-20-871/profiler-output/system/incremental/2021110802/1636338240.algo-1.json'},
       {'Key': 'sagemaker/DEMO-xgboost-dm/output/xgboost-2021-11-08-02-20-20-871/rule-output/ProfilerReport-1636338020/profiler-output/profiler-reports/CPUBottleneck.json'},
       {'Key': 'sagemaker/DEMO-xgboost-dm/output/xgboost-2021-11-08-02-20-20-871/rule-output/ProfilerReport-1636338020/profiler-output/profiler-reports/OverallFrameworkMetrics.json'},
       {'Key': 'sagemaker/DEMO-xgboost-dm/train/train.csv'},
       {'Key': 'sagemaker/DEMO-xgboost-dm/output/xgboost-2021-11-08-02-20-20-871/rule-output/ProfilerReport-1636338020/profiler-output/profiler-reports/LoadBalancing.json'},
       {'Key': 'sagemaker/DEMO-xgboost-dm/output/xgboost-2021-11-08-02-20-20-871/profiler-output/framework/training_job_end.ts'},
       {'Key': 'sagemaker/DEMO-xgboost-dm/output/xgboost-2021-11-08-02-20-20-871/rule-output/ProfilerReport-1636338020/profiler-output/profiler-reports/BatchSize.json'},
       {'Key': 'sagemaker/DEMO-xgboost-dm/output/xgboost-2021-11-08-02-20-20-871/rule-output/ProfilerReport-1636338020/profiler-output/profiler-report.ipynb'},
       {'Key': 'sagemaker/DEMO-xgboost-dm/output/xgboost-2021-11-08-02-20-20-871/rule-output/ProfilerReport-1636338020/profiler-output/profiler-reports/LowGPUUtilization.json'},
       {'Key': 'sagemaker/DEMO-xgboost-dm/output/xgboost-2021-11-08-02-20-20-871/rule-output/ProfilerReport-1636338020/profiler-output/profiler-reports/OverallSystemUsage.json'},
       {'Key': 'sagemaker/DEMO-xgboost-dm/output/xgboost-2021-11-08-02-20-20-871/profiler-output/system/incremental/2021110802/1636338180.algo-1.json'}]}]




```python

```
