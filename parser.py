import csv
import code
import datetime

#github test

class data_day:
    attr_name =["date","ASPFWR5","US10YR","EPS","PER","OPEN","HIGH","LOW","CLOSE","BDIY","VIX","PCR","MVOLE","DXY","ASP","ADVDECL","FEDFUNDS","NYSEADV","IC","BAA","NOS","BER","DVY","PTB","AAA","SI","URR","FOMC","PPIR","RV","LOAN","VVIX","NAPMNEWO","NAPMPRIC","NAPMPMI","US3M","DEL","BBY","HTIME","LTIME","TYVIX","PUC","CRP","TERM","UR","INDPRO","HS","VRP","CAPE","CATY","INF","SIM","TOM","RELINF","DTOM","sentiment1","sentiment2","sentiment3","Hulbert.sentiment"]
    def __init__(self,data_lst):
        self.dic = {}
        i = 0
        for name in self.attr_name:
            if data_lst[i] == 'NA':
                self.dic[name] = None
            else:
                self.dic[name] = data_lst[i]
            i += 1
    
    def get(self,name):
        # get the attribute with attribute's name
        # return a number
        return self.dic[name]
    
    def get_date(self):
        # return a datetime object
        date = self.dic['date']
        lst = date.split('-')
        return datetime.datetime(int(lst[0]),int(lst[1]),int(lst[2]))


data = []
with open("dataset.csv",newline = '') as csvfile:
    reader = csv.reader(csvfile,delimiter = ',',quotechar='"')
    next(reader)
    for row in reader:
        today = data_day(row)
        data.append(today)
        

code.interact(local=locals())

