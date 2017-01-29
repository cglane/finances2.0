from lib import readCsv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import numpy as np
from sklearn import svm
from scipy.sparse import coo_matrix, hstack
import csv
financesPath = './csvFiles/finances.csv'
headers = {'location':'Description II','value':['Amount Salary','Amount BankSC','13 Drews Ct.','Work Related/Other'],'description':'Description'}

"Initialize svm and create bag of words features from finances file"
def trainSVM(financesPath,headers):
    data_dict = readCsv.readCSV(financesPath).valuesLocationsDescriptions(headers)
    clf,vectorizer = train_data(data_dict)

def writeToCSV(outputName,transactions):
    with open(outputName , "wb") as outcsv:
        writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        for item in transactions:
            #Write item to outcsv
            writer.writerow([item[0], item[1],'','','', item[2],item[3]])
"Write for API clients"
def writeToCSVPublic(outputName,transactions):
    with open(outputName,'wb')as outcsv:
        writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerow(['Date','Value','Location','Description'])
        for item in transactions:
            #Write item to outcsv
            writer.writerow([item[0], item[1],item[2],item[3]])
"Adds value parameter to bag of words array"
def training_feature(vectorizer,location,value):
    location_matrix = vectorizer.transform([location])
    value_matrix = coo_matrix([value])
    return hstack([location_matrix,value_matrix]).toarray()
"Training svm with financial spreadsheet using bag of words and value"
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

"Use svm to predict description and write new csv file for all transactions"
def predictDescription(fileName,clf,vectorizer,input_headers,header = None, source = 'Amex', ignoreLocations = ['ONLINE PAYMENT ']):
    transactions = []
    inputList = readCsv.readCSV(fileName=fileName,header = header,source = source,ignoreLocations= ignoreLocations).dateValueLocation(input_headers)
    for item in inputList:
        predict_array = training_feature(vectorizer,(item['location'].split('-')[0]),(item['value']))
        transactions.append((item['date'],item['value'],item['location'],clf.predict(predict_array)[0]))
    writeToCSVPublic(outputName = './output.csv',transactions=transactions)


# input_headers = {'date':0,'location':2,'value':7}
# clf,vectorizer = train_data(data_dict)
#
# predictDescription(fileName = './csvFiles/creditCard.csv',input_headers = input_headers,clf = clf,vectorizer = vectorizer)
# for item in inputList:
#     predict_array = training_feature(vectorizer,(item['location'].split('-')[0]),(item['value']*-1))
#     print (item['location'],item['value'],clf.predict(predict_array))
