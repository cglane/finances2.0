from lib import readCsv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import numpy as np
from sklearn import svm
from scipy.sparse import coo_matrix, hstack

# {'date':'Date','value':'Amount Salary','location':'Description II', 'description':'Description'})
financesPath = './csvFiles/finances.csv'
data_dict = readCsv.readCSV(financesPath).valuesLocationsDescriptions({'location':'Description II','value':['Amount Salary','Amount BankSC','13 Drews Ct.','Work Related/Other'],'description':'Description'})

def training_feature(vectorizer,location,value):
    location_matrix = vectorizer.transform([location])
    value_matrix = coo_matrix([value])
    return hstack([location_matrix,value_matrix]).toarray()

def train_data(data_dict):
    vectorizer = CountVectorizer()
    vectorizerArray = vectorizer.fit(data_dict['locations'])

    locations_matrix = vectorizer.fit_transform(data_dict['locations'])
    values_matrix = coo_matrix(data_dict['values'])
    ###Add values to locations matrix, create array
    features = hstack([locations_matrix,values_matrix]).toarray()
    labels = data_dict['descriptions']

    clf = svm.LinearSVC(random_state=0)
    clf.fit(features, labels)
    return clf,vectorizer

clf,vectorizer = train_data(data_dict)
predict_array = training_feature(vectorizer,'shell',-3.21)
print predict_array
print clf.predict(predict_array)
