
import json
import re
import sklearn
import pandas as pd
import numpy as np

with open('dataset.json') as f:
    data = json.load(f)


def process_date(date_ago):
    date_ago = date_ago.strip()
    if date_ago == None:
        return None
    if 'days' in date_ago or 'day' in date_ago:
        days = int(date_ago.replace('day', '').replace('s','').replace('ago','').strip())
        return days
    elif 'mins ago' in date_ago  or'hour' in date_ago or 'hours' in date_ago:
        return 0
    if 'weeks' in date_ago:
        weeks = int(date_ago.replace('weeks', '').replace('ago','').strip())
        return weeks*7
    if 'months' in date_ago:
        months = int(date_ago.replace('months', '').replace('ago','').strip())
        return months * 30
    elif 'month' in date_ago:
        months = int(date_ago.replace('month', '').replace('ago','').strip())
        return months * 30
    if 'year' in date_ago:
        years = int(date_ago.replace('year', '').replace('s','').replace('ago','').strip())
        return years * 365

def text_to_dict(listing):
    collab = 0
    out = {}
    t1 = listing['text'].split('\n')
    out['category']=listing['category']

    n = len(t1)
    reg = re.compile(r'\(')
    
    dates = t1[0].replace(')','')
    
    if '(' in dates:
        dates = re.split(reg, dates)
        orig_date = dates[1]
        bump_date = dates[0]
        out['bumped']=1
    else:
        orig_date = dates
        bump_date = dates
        out['bumped']=0

        
    out['orig_date'] = process_date(orig_date)
    out['bump_date']=process_date(bump_date)
    
    designer = t1[1]
    if '×' in designer:
        designers = [st.strip() for st in designer.split('×')]
        if designers[0] in designers[1]:
            out['collab'] = 0
            out['designer'] =  designers[1]
        elif designers[1] in designers[0]:
            out['collab'] = 0
            out['designer'] =  designers[0]
        else:
            out['collab'] = 1
            for d in designers:
                out['designer'] = designers
    else:
        out['designer'] = designer
        out['collab']=1
        
    if designer in ['YS (YAMAMOTO)', 'YS FOR MEN' , 'YS FOR MEN / YAMAMOTO']:
        designer = 'YOHJI YAMAMOTO'
        
    out['size'] = t1[2]
    out['name'] = t1[3]
    out['orig_price'] = int(t1[4].replace('$','').replace(',',''))
    out['bump_price'] = int(t1[4].replace('$','').replace(',',''))
    if n == 6:
        out['orig_price'] = int(t1[5].replace('$','').replace(',',''))
    return(out)



unsold_dict = [text_to_dict(listing) for listing in data['unsold']]
sold_dict = [text_to_dict(listing) for listing in data['sold']]
for listing in unsold_dict:
    listing['sold']=0
for listing in sold_dict:
    listing['sold']=1
datavector= unsold_dict + sold_dict
import dill
dill.settings['recurse']=True
dill_file = open("datavector", "wb")
dill_file.write(dill.dumps(datavector))
dill_file.close()



def n_sold_within(days):
    count = 0
    for el in sold_dict:
        if el['orig_date'] <= days:
            count+=1
    return count
def n_unsold_within(days):
    count = 0
    for el in unsold_dict:
        if el['orig_date'] <= days:
            count+=1
    return count



days21 = pd.DataFrame([(n_unsold_within(i), n_sold_within(i)) for i in range(0, 26)])

days21.columns = ['unsold', 'sold']

days21['listed'] = days21['sold']+days21['unsold']
days21['ratio'] = days21['sold']/days21['listed']
days21.index  +=1
fig = days21.plot(y=['listed', 'sold'],
           xlim=[1,21],
           title='Grailed Listings Over 21 Days',
           figsize = (12,8),
           xticks = range(1, 22),
           fontsize = 16)

fig.set_title('Grailed Listings, Last 21 Days', fontsize=30)
fig.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), prop={'size': 20}, markerscale=10)
lgd = fig.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), prop={'size': 20}, markerscale=10)

fig.get_figure().savefig("21days.png", bbox_extra_artists=(lgd,), bbox_inches='tight')



months6 = pd.DataFrame([(n_unsold_within(30*i), n_sold_within(30*i)) for i in range(1,7)])
months6.columns=['unsold', 'sold']
months6['listed'] = months6['unsold'] + months6['sold']
months6.index+=1
fig = months6.plot(y=['listed', 'sold'],
           xlim=[1,6],
           title='Grailed Listings, Last 6 months',
           figsize = (12,8),
           xticks = range(1, 7),
           fontsize = 16)
figure_ = fig.get_figure()
fig.set_title('Grailed Listings Over 6 Months', fontsize=30)
lgd = fig.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), prop={'size': 20}, markerscale=10)


figure_.savefig("6months.png", bbox_extra_artists=(lgd,), bbox_inches='tight')


df_unsold = pd.DataFrame.from_dict(unsold_dict)
df_sold = pd.DataFrame.from_dict(sold_dict)

d_unsold = df_unsold.apply(lambda x: pd.Series(x['designer']),axis=1).stack().reset_index(level=1, drop=True)
d_sold = df_sold.apply(lambda x: pd.Series(x['designer']),axis=1).stack().reset_index(level=1, drop=True)
d_unsold.name = 'designer'
d_sold.name = 'designer'


df_unsold['sold']=0
df_sold['sold']=1


df_all = df_unsold.append(df_sold)


df_all.to_pickle("df-all")


df2 = pd.read_pickle("df-all")


counts = df_all.groupby('designer')['size'].count()



designer_drop = []
for designer, count in counts.items():
    if count < 100:
        designer_drop.append(designer)

df_new = df_all[~df_all.designer.isin(designer_drop)]


df_new.to_pickle("df-new")


def price_dif(data):
    sold = data.loc[data['sold'] ==1]['orig_price'].mean()
    unsold = data.loc[data['sold'] ==0]['orig_price'].mean()
    
    if pd.notnull(unsold/sold):
        return unsold/sold
    return 0


sorted(df_new.groupby(['designer', 'category']).apply(price_dif).items(), key=lambda el:el[1])

ratios = np.array(sorted(df_new.groupby(['designer', 'category']).apply(price_dif).items(), key=lambda el:el[1]))[:,1]

print(ratios.mean())


print(np.median(ratios))



df_dummy = pd.concat([df_new, pd.get_dummies(df_new['category'])], axis=1)

df_dummy = pd.concat([df_new2, pd.get_dummies(df_dummy['size'])], axis=1)

df_dummy = pd.concat([df_new3, pd.get_dummies(df_dummy['designer'])], axis=1)

df_dummy.to_pickle("df-dummy")

