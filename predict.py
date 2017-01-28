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

# def predictDescription(inputList):
#     for value in inputList:

input_headers = {'date':0,'location':2,'value':7}
inputList = readCsv.readCSV('./csvFiles/creditCard.csv',header = None,source = 'Amex').dateValueLocation(input_headers)
clf,vectorizer = train_data(data_dict)
for item in inputList:
    predict_array = training_feature(vectorizer,(item['location'].split('-')[0]),(item['value']*-1))
    print (item['location'],item['value'],clf.predict(predict_array))
