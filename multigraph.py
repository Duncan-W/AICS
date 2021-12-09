import statistics
import fbprophet
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from pandas import Timestamp
import sklearn.metrics
import plotly
import csv
import matplotlib.pylab as plt
import urllib.request

# Variable naming conventions:
# name: the name of an entity
# e: an entity
# d: date, represented as a string 'yyyy-mm-dd'
# y: number measured at a particular date
# t: a timed measurement consisting of (d, y)


class Entity:
    def __init__(self, name):
        self.name = name
        self.ts = []
        self.results = []

    def __repr__(self):
        return self.name


entities = {}


def entities_sorted(key=lambda e: e.name):
    es = []
    for e in entities.values():
        es.append(e)
    es.sort(key=key)
    return es


# read entities from data file
def entity(name):
    if name in entities:
        e = entities[name]
    else:
        e = Entity(name)
        entities[name] = e
    return e



temperature = {
  "filename": "temperature_tiny",
  "online": True,
  "id_name": 4,
  "date": 1,
  "production": 2,
  "limit": 0,
  "period": "monthly"
}

gas = {
  "filename": "State_Data",
  "online": True,
  "id_name": 2,
  "date": 1,
  "production": 4,
  "limit": 0,
  "period": "monthly"
}


employment = {
  "filename": "ststdsadata_teeny",
  "online": True,
  "id_name": 1,
  "date": 0,
  "production": 5,
  "limit": 0,
  "period": "monthly"
}

employment_ns = {
  "filename": "ststdsadata_nonseas_teeny",
  "online": True,
  "id_name": 1,
  "date": 0,
  "production": 5,
  "limit": 0,
  "period": "monthly"
}

farm = {
  "path": "C:/alpha/",
  "filename": "VM_preprocessed",
  "online": False,
  "id_name": 4,
  "date": 5,
  "production": 17,
  "limit": 10,
  "period": "daily"
}

max_nnn=5
cur_data = macedonia

if cur_data["online"]:
    url = "https://raw.githubusercontent.com/Duncan-W/data/main/"+cur_data["filename"] +".csv"
    response = urllib.request.urlopen(url)
    lines = [l.decode('utf-8') for l in response.readlines()]
    reader = csv.reader(lines)
else:
    reader = csv.reader(open(""+cur_data["path"]+cur_data["filename"]+".csv"))


# skip the headings
reader.__next__()
for r in reader:
    # adjust column numbers here according to file format
    e = entity(r[cur_data["id_name"]]) # id_name="State_ID"
    d = r[cur_data["date"]] # date_name="Date"
    y = float(r[cur_data["production"]]) # production="Production"
    # check to make sure date is valid
    dd=datetime.strptime(d, "%Y-%m-%d")
    d='%04d-%02d-%02d'%(dd.year,dd.month,dd.day)
    e.ts.append((d, y))
print("entities: " + str(entities_sorted()))

# Sort each time series by date
# in case they were not already sorted in the data file
for e in entities.values():
    e.ts.sort()


#discard some entities
if cur_data["limit"]>0:
    names=sorted(list(entities.keys()))
    print(f'{len(names)} total entities')
    n=cur_data["limit"]
    print(f'keeping {n}')
    names=names[:n]
    es={}
    for name in names:
        es[name]=entities[name]
    entities=es



# What is the complete list of dates we are dealing with?
all_ds = set()
for e in entities.values():
    for t in e.ts:
        all_ds.add(t[0])
all_ds = sorted(list(all_ds))

# Some entities might have missing entries
# if so, fill them in with None
# holey_ts is used when we are filling in holes with values from neighbors
for e in entities.values():
    tdict = {}
    for d, y in e.ts:
        tdict[d] = y

    y = 0
    for d in all_ds:
        if d not in tdict  or not tdict[d]:
            tdict[d] = None
        y = tdict[d]

    ts = []
    for d in all_ds:
        ts.append((d, tdict[d]))
    e.holey_ts = ts

# Some entities might have missing entries
# if so, fill them in with prev
for e in entities.values():
    tdict = {}
    for d, y in e.ts:
        tdict[d] = y

    y = 0
    for d in all_ds:
        if d not in tdict  or not tdict[d]:
            tdict[d] = y
        y = tdict[d]

    ts = []
    for d in all_ds:
        ts.append((d, tdict[d]))
    e.filled_ts = ts


def avg_tss(tss):
    dicts=[]
    for ts in tss:
        dc={}
        for t in ts:
            dc[t[0]] = t[1]
        dicts.append(dc)

    dset=set()
    for ts in tss:
        for t in ts:
            dset.add(t[0])
    ds=sorted(list(dset))

    rts = []
    for d in ds:
        ys=[]
        for dc in dicts:
            if d in dc and dc[d]is not None:
                ys.append(dc[d])
        if ys:
            y=statistics.mean(ys)
        else:
            y=0.0
        rts.append((d, y))
    return rts

def vals(ts):
    xs=[]
    for t in ts:
        xs.append(t[1])
    return xs

def adjust(xs, y):
    for i in range(len(xs)):
        xs[i]*=y


def euclid_distance(ts, ts2):
    assert len(ts) == len(ts2)
    xs=vals(ts)
    xs2=vals(ts2)
    m=statistics.mean(xs)
    m2=statistics.mean(xs2)
    #adjust(xs,m2/m)
    dist2 = 0
    for i in range(len(xs)):
        dist2 += (xs[i] - xs2[i]) ** 2
    # don't bother with sqrt
    # since we are only ranking distances
    return dist2


def entity_distance(e, e2):
    return euclid_distance(e.filled_ts, e2.filled_ts)


# Calculate the nearest neighbor of every entity
def calc_nns():
    for e in entities.values():
        e.nn = []
        for e2 in entities.values():
            if e == e2:
                continue
            e.nn.append(e2)
        e.nn.sort(key=lambda e2:entity_distance(e, e2))
        e.distance=entity_distance(e, e.nn[0])


def before(ts, start_date):
    return [t for t in ts if t[0] < start_date]


def after(ts, start_date):
    return [t for t in ts if t[0] >= start_date]


# Prophet expects pandas dataframes as input
def mkdf(ts):
    ds = []
    ys = []
    for t in ts:
        ds.append(t[0])
        ys.append(t[1])
    return pd.DataFrame(data={"ds": ds, "y": ys})


# Run an actual forecast
def prophet_forecast(train, test):
    m = fbprophet.Prophet()
    m.fit(train)
    forecast = m.predict(test)
    #fig = fbprophet.plot.plot_plotly(m, forecast)
    #fig.layout.update(title=cur_data["filename"])
    #plotly.offline.iplot(fig)
    #m.plot_components(forecast);
    #exit()
    return forecast






# Split between training and test data
if cur_data["period"] == "monthly":
    period = 12
    j = 4
    start_at=period*4
if cur_data["period"] == "daily":
    period = 360
    j = 4
    start_at=period*4


# Run forecasts
for i in range(start_at, len(all_ds), period):
    start_date = all_ds[i-period]
    print(start_date)
    for e in entities.values():
        r = {}
        r["name"] = e.name
        r["end_date"] = all_ds[i - 1]

        # first do the plain calculations without reference to neighbors
        # just with missing values filled in from the previous day
        e.ts = e.filled_ts[i-period*j:i]
        both = mkdf(e.ts)
        train = mkdf(before(e.ts, start_date))
        test = mkdf(after(e.ts, start_date))
        forecast = prophet_forecast(train, test)
        cmp = forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]].join(
            both.set_index("ds")
        )

        e.mae=[0.0]*(max_nnn+1)
        e.rmse=[0.0]*(max_nnn+1)

        e.mae[0] = sklearn.metrics.mean_absolute_error(
            cmp["y"].values, cmp["yhat"].values
        )
        r["mae"] = e.mae[0]
        if cur_data["online"]:
            e.rmse[0] = sklearn.metrics.mean_squared_error(
                    cmp["y"].values, cmp["yhat"].values, squared = False
                )
        else:
            e.rmse[0] = sklearn.metrics.mean_squared_error(
                    cmp["y"].values, cmp["yhat"].values
                )
        r["rmse"] = e.rmse[0]    
        cmp=cmp[['y','yhat']]

        # now do the stepping sideways in time
        # with missing values filled in from neighbors
        e.ts = e.holey_ts[i-period*j:i]

        calc_nns()
        for nnn in range(1,max_nnn+1):

            tss=[e.ts]+[ee.holey_ts[i-period*j:i] for ee in e.nn[:nnn]]
            e.augmented_ts = avg_tss(tss)
            both = mkdf(e.augmented_ts)
            train = mkdf(before(e.augmented_ts, start_date))
            test = mkdf(after(e.augmented_ts, start_date))
            forecast = prophet_forecast(train, test)
            cmp1 = forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]].join(
                both.set_index("ds")
            )

            #test = mkdf(after(e.ts, start_date))

            # where cmp["ds"] == test["ds"]
            # then cmp["y"] = test["y"]

            e.mae[nnn]= sklearn.metrics.mean_absolute_error(
                cmp["y"].values, cmp1["yhat"].values
            )
            if cur_data["online"]:
                e.rmse[nnn] = sklearn.metrics.mean_squared_error(
                        cmp["y"].values, cmp1["yhat"].values, squared = False
                    )
            else:
                e.rmse[nnn] = sklearn.metrics.mean_squared_error(
                        cmp["y"].values, cmp1["yhat"].values
                    )            
            #r["distance"+str(nnn)] = e.distance
            r["nn"+str(nnn)] = e.nn[nnn].name
            r["mae"+str(nnn)] = e.mae[nnn]
            r["maed"+str(nnn)] = e.mae[0] - e.mae[nnn]
            r["rmse"+str(nnn)] = e.rmse[nnn]
            r["rmsed"+str(nnn)] = e.rmse[0] - e.rmse[nnn]

            del cmp1['y']
            del cmp1['yhat_lower']
            del cmp1['yhat_upper']
            cmp1=cmp1.rename(columns={"yhat":"yhat"+str(nnn)})
            #print(cmp1)
            cmp=cmp.merge(cmp1,on='ds')
            #print(cmp)
            #exit(0)

        plot = cmp.plot(title =e)
        #plot.legend(["actual", "forecast_nn"+str(nnn)]);
        plot.set_title("production")
        fig1 = plot.get_figure()

        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        fig1.savefig(str(e)+str(i)+'.pdf', format="pdf", dpi = 400, transparent=True)
        e.results.append(r)
        plt.close()
        #exit(0)

    # Results
    results = []
    for e in entities_sorted(lambda e: e.results[-1]["mae"]):
        results.extend(e.results)
    results_df = pd.DataFrame(data=results)

    # Print results
    print()
    print("Dates: " + all_ds[0] + " : " + start_date + " : " + all_ds[-1])

    print()
    print(results_df)

    # Output to file
    results_df.to_csv("results__multigraph_"+cur_data["filename"]+".csv")
