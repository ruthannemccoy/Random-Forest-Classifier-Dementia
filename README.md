Random Forest Classifier for Dementia Diagnosis Prediction

Description: This code utilizes an opensource dataset (https://www.kaggle.com/datasets/kaggler2412/dementia-patient-health-and-prescriptions-dataset/data) to perform a random forest classification for prediction of dementia diagnosis. Data that directly correlated to dementia, such as being on a prescription for dementia, were removed as this resulted in overfitting of the model. String data was converted to 0/1 for no/yes and Likert scales were converted to values 1-3 in order of increasing risk. Next steps are to set up a data visualization.

The followed dependencies are needed. 
Python 3.12.0
SciKitLearn 1.4.2
Pandas 2.2.0

Must have dementia_data_set_clean.csv downloaded into same directory as main.py
