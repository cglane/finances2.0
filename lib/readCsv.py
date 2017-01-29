import pandas as pd
from math import isnan
# columnNames = {'date':'date','value':['amount'],'location':'location','description':'description'}
###Values may be strings or NaN
def extendValues(myList):
    myList = [x for x in myList if x != 0]
    if len(myList) > 0:
        try:
            return [float(myList[0])]
        except Exception as e:
            return [0]
    else:
        return [0]
def formatValue(myList,source):
    if source == 'Amex':
        return [round(float(x)*-1,2) for x in myList]
    else:
        return myList
def formatDate(myList,source):
    if source == 'Amex':
        ###Only return month/day/year format
        return [x.split('  ')[0] for x in myList]
    elif source == 'Finances':
        return myList

def formatLocation(myList,source, ignoreLocations):
    if source == 'Amex':
        ###Remove city and state from location description
        return [x.split('-')[0] for x in myList if x.split('-')[0] not in ignoreLocations]
    elif source == 'Finances':
        return myList

class readCSV(object):
    """Read CSV with pandas"""
    def __init__(self, fileName, ignoreLocations = [''],header = 'infer',source ='Finances'):
        print header
        self.file = pd.read_csv(fileName, header = header)
        self.source = source
        self.ignoreLocations = ignoreLocations
    def __str__(self):
        return str(self.file.head(n=3))
    def columns(self):
        return self.file.columns

    def dateValueLocation(self,columns):
        dates = formatDate(self.file[columns['date']].fillna('Empty Date'),self.source)
        values = formatValue(self.file[columns['value']].fillna(0),self.source)
        locations = formatLocation(self.file[columns['location']].fillna('no location'),self.source,self.ignoreLocations)
        return [{'date':x[0],'value':x[1],'location':x[2]} for x in zip(dates,values,locations)]

    def dateValueLocationDescription(self,columns):
        dates = self.file[columns['date']].fillna('Empty Date')
        values = self.file[columns['value']].fillna(0)
        locations = self.file[columns['location']].fillna('no location')
        descriptions = self.file[columns['description']].fillna('no description')
        return [tuple(x) for x in zip(dates,values,locations,descriptions)]

    def valuesLocationsDescriptions(self,columns):
        values =  [extendValues(x) for x in self.file[columns['value']].fillna(0).values]
        descriptions = self.file[columns['description']].fillna('no description')
        locations = self.file[columns['location']].fillna('no location')
        return {'values':values,'locations':locations,'descriptions':descriptions}
