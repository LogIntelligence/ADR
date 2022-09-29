# ADR

Large-scale software-intensive systems often generate logs for troubleshooting. The system logs are semi-structured text messages that record the internal status of a system at runtime. 

ADR (Anomaly Detection by workflow Relations) can mine numerical relations from logs using linear algebra based techniques and then utilize the discovered relations to detect system anomalies. Firstly the raw log entries are parsed into sequences of log events and transformed to an extended event-count-matrix. The relations among the matrix columns represent the relations among the system events in workflows. Next, ADR evaluates the matrix's nullspace that corresponds to the linearly dependent relations of the columns. Anomalies can be detected by evaluating whether or not the logs violate the mined relations. 

We design two types of ADR: sADR (for semi-supervised learning, need normal logs for training) and uADR (for unsupervised learning).

![Workflow](workflow.png?raw=true "Workflow")

## Demo

The ADR demo is presented in the jupyter notebook: ___demo.ipynb___.

To view and run the ___demo.ipynb___, the followings are required:

- python 3
- jupyter
- notebook
- numpy
- pandas
- scikit-learn
