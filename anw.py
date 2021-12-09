import csv
import math
from datetime import datetime
from pprint import pprint
import pandas as pd
from collections import OrderedDict
import statistics
pd.options.display.float_format = "{:,.2f}".format
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', None)


class Record:
    def __init__(self, d):
        self.d=d

class Entity:
    def __init__(self, name):
        self.name = name
        self.rs = []

    def __repr__(self):
        return self.name


entities = {}
counter = 0
#THE FOLOWING IS USED IN results()

files = ["results__multigraph_temperature_small.csv"]


def entity(name):
    if name in entities:
        e = entities[name]
    else:
        e = Entity(name)
        entities[name] = e
    return e





def genlatex(df, topic, caption=None, headers='', row_headers=None, options =''):
    # Expects columns relating to {topic}_avg{header number} and {topic}_stdev{header number}. Combines both

    pm='$\\pm'
    output =''
    global counter
    print()
    print()
    print("\\begin{table}[htbp]")
    print('\\label{tab:'+topic+str(counter)+'}')
    if caption:
        print('\\caption{'+caption.replace('_', ' ')+'}')
    print(options)
    output+='\\begin{tabular}{@{}'
    if row_headers:
        output+='l'
    for h in headers:
        output+='l'
    print(output+"@{}}")

    output =''
    for h in headers:
        if h is headers[(len(headers)-1)]:
            output+='\\textbf{'+topic+str(h)+'}'+'\t \\\\ \\midrule'
        else:
            output+='\\textbf{'+topic+str(h)+'}'+'\t &'
    if row_headers:
        print(row_headers+output)
    for i, j in df.iterrows():
        print(i,end=' & ')
        for k in headers:
            print("{:.2f}".format(j[topic+k+'_avg']),end=' ')
            print(pm,end=' ')
            if k is headers[(len(headers)-1)]:
                print("{:.2f}".format(j[topic+k+'_stdev']),end='$\t')
            else:
                print("{:.2f}".format(j[topic+k+'_stdev']),end='$\t &')
        print('\\\\')
    print('\\bottomrule')
    print('\\end{tabular}')
    diff = options.count('{')-options.count('}')
    for i in range(diff):
        print('}')
    print('\\end{table}')
    print()
    if topic is 'mae':
        counter+=1

def generator(es):
    #creates dataframe of results

    resultsDict = OrderedDict([])

    allds=set()
    for e in es:
        for rc in e.rs:
            allds.add(rc.d)
            for attr, value in rc.__dict__.items():
                if "d" in str(attr):
                    continue
                resultsDict[str(attr)]= ''
    allds=sorted(list(allds))
    print('years:')
    print(allds)
    df_ = pd.DataFrame(index=allds)


    # Creating an empty dictionary

    avg_results = []
    stdv_data = []
    stdv_results = []

    for key  in resultsDict:
        for year in allds:
            tot=0.0
            n=0
            for e in es:
                for rc in e.rs:
                    if rc.d==year:
                        n+=1
                        tot+=getattr(rc, key)
                        stdv_data.append(getattr(rc, key))
            #print(str(d)+'\t'+str(tot))
            avg_results.append(tot/n)
            stdv_results.append(statistics.stdev(stdv_data))
            stdv_data = []
        #resultsDict[str(key)] = results
        df_[str(key)+"_avg"] = avg_results
        df_[str(key)+"_stdev"] = stdv_results
        avg_results=[]
        stdv_results=[]
    return df_



def bybestsel(es):
    output =''
    n=0
    mae=0.0
    maek=0.0
    mae_mostk = 0.0
    maedk=0.0
    maed_mostk = 0.0
    no_of_e = 0
    es2=[]


    avg_maed1=0.0
    for e in es:
        rc=e.rs[0]
        avg_maed1+=rc.maed1
    avg_maed1/=len(es)
    print(avg_maed1)

    ks=[0,0,0,0,0]
    for e in es:
        rc=e.rs[0]
        d=[rc.maed1,rc.maed2,rc.maed3,rc.maed4,rc.maed5]
        ma=max(d)
        if ma<0:continue
        k=d.index(ma)
        ks[k]+=1
    print('ks\t'+str(ks))
    mostk=ks.index(max(ks))
    print('mostk\t'+str(mostk+1))

    for e in es:

        rc=e.rs[0]
        d=[rc.maed1,rc.maed2,rc.maed3,rc.maed4,rc.maed5]
        ma=max(d)
        #print(e.name,str(d.index(ma)))
        if ma<0:continue
        k=d.index(ma)

        rc2=e.rs[1]
        d2=[rc2.maed1,rc2.maed2,rc2.maed3,rc2.maed4,rc2.maed5]

        pos=0
        for dd in d:
            if dd>0:
                pos+=1
        if pos==1 and rc.maed1>avg_maed1:
            continue

        rc=e.rs[1]
        d=[rc.maed1,rc.maed2,rc.maed3,rc.maed4,rc.maed5]
        #if k2<0:k=mostk
        
        es2.append(e)

        #should range start at 1 or 2?
        for i in range(1,len(e.rs)):
            rc=e.rs[i]
            n+=1
            mae+=rc.mae
            d=[rc.mae1,rc.mae2,rc.mae3,rc.mae4,rc.maed5]
            maek+=d[k]
            mae_mostk+=d[mostk]
            d=[rc.maed1,rc.maed2,rc.maed3,rc.maed4,rc.maed5]
            maedk+=d[k]
            maed_mostk+=d[mostk]
    print()

    print('nrecs\t'+"{:.2f}".format(n))
    print('mae\t'+"{:.2f}".format(mae/n))
    print('maek\t'+"{:.2f}".format(maek/n))
    print('maedk\t'+"{:.2f}".format(maedk/n))
    print('mae_mostk\t'+"{:.2f}".format(mae_mostk/n))
    print('maed_mostk\t'+"{:.2f}".format(maed_mostk/n))
    print('*******************************')
    return es2



def results(file):
    entities.clear()
    reader = csv.reader(open(file))
    #dont skip the headings

    n=0
    mind=999999
    maxd=9

    for r in reader:
        if n is 0:
            date = r.index("end_date")
            rmae = r.index("mae")
            rmae1 = r.index("mae1")
            rmaed1 = r.index("maed1")
            rmae2 = r.index("mae2")
            rmaed2 = r.index("maed2")
            rmae3 = r.index("mae3")
            rmaed3 = r.index("maed3")
            rmae4 = r.index("mae4")
            rmaed4 = r.index("maed4")
            rmae5 = r.index("mae5")
            rmaed5 = r.index("maed5")
            n+=1
            continue
        n+=1
        name=r[1]
        e=entity(name)
        if '-' in r[date]:
            dd=datetime.strptime(r[date], "%Y-%m-%d")
            d='%04d-%02d-%02d'%(dd.year,dd.month,dd.day)
            d=dd.year
        elif '/' in r[date]:
            dd=datetime.strptime(r[date], "%d/%m/%Y")
            d=dd.year
        else:
            raise Exception("Incorrect date format")
        mind=min(mind,d)
        maxd=max(maxd,d)
        rc=Record(d)
        rc.mae=float(r[rmae])
        rc.mae1=float(r[rmae1])
        rc.maed1=float(r[rmaed1])
        rc.mae2=float(r[rmae2])
        rc.maed2=float(r[rmaed2])
        rc.mae3=float(r[rmae3])
        rc.maed3=float(r[rmaed3])
        rc.mae4=float(r[rmae4])
        rc.maed4=float(r[rmaed4])
        rc.mae5=float(r[rmae5])
        rc.maed5=float(r[rmaed5])
        e.rs.append(rc)

    es=entities.values()
    #pprint(es)
    bybestsel_es=bybestsel(es)


    
    


    ######## LATEX TABLE GENERATOR

    df_ = generator(es)

    headers = ['','1','2','3','4','5']
    options='\\setlength{\\tabcolsep}{9pt}\\centering \\resizebox{\\textwidth}{!}{%'

    genlatex(df_, 'mae', caption=file, headers=headers, options=options, row_headers='\\textbf{year}\t &')
    df_ = generator(bybestsel_es)
    genlatex(df_, 'mae', caption=file+' by best sel', headers=headers, options=options, row_headers='\\textbf{year}\t &')


for f in files:
    print(f)
    results(f)
