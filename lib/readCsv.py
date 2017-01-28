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

class readCSV(object):
    """Read CSV with pandas"""
    def __init__(self, fileName):
        self.file = pd.read_csv(fileName)
    def __str__(self):
        return str(self.file.head(n=3))
    def columns(self):
        return self.file.columns
    def dateValueLocation(self,columns):
        dates = self.file[columns['date']].fillna('Empty Date')
        values = self.file[columns['value']].fillna(0)
        locations = self.file[columns['location']].fillna('no location')
        return [tuple(x) for x in zip(dates,values,locations)]
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