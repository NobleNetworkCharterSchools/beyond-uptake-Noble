
# Building a model to predict student attrition from High School

*Author*: Matt Niksch

*Synopsis*:
- Noble is a non-profit operator of 17 high schools in Chicago and serves more than 10% of public high school students in the city
- Student outcomes, including high school graduation, test score growth, GPA, college matriculation & completion are very strong
- However, although better than nearby options, too many students leave Noble and transfer to other schools. We would like to predict which students are likely to leave in order to provide them with more supports.

### Summary of potential features we might use to predict labels related to student depature:
![Features](./PotentialFeatures.png)

## Final crime database construction

In this file, we'll be working with the concepts from the prior file to create a condensed database of crimes that has a row per block per year, where year is chosen to start on a specific month and then include the following 11 months. (This will allow for us to look at a full year prior to the start of the school year


```python
# First some library imports

import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from pygeocoder import Geocoder
```


```python
crime_file = 'Crimes_-_2016.csv'
cdf = pd.read_csv(crime_file, index_col=0)
```


```python
c_ll_df = cdf[~pd.isnull(cdf['Latitude'])]
print('{} of {} records in 2016 had lat/longs'.format(len(c_ll_df),len(cdf)))
```

    251014 of 268073 records in 2016 had lat/longs
    


```python
# Now combine this with the 2017 database to create as full a block database as possible
a = c_ll_df[['Block', 'Latitude', 'Longitude']]
```


```python
crime_file = 'Crimes_-_2017.csv'
cdf = pd.read_csv(crime_file, index_col=0)
b = cdf[['Block', 'Latitude', 'Longitude']]
```


```python
block_lat_long_df = pd.concat([a,b])
block_lat_long_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 462959 entries, 10398641 to 23648
    Data columns (total 3 columns):
    Block        462959 non-null object
    Latitude     454881 non-null float64
    Longitude    454881 non-null float64
    dtypes: float64(2), object(1)
    memory usage: 12.4+ MB
    


```python
rounded_lat_lon = block_lat_long_df[['Block', 'Latitude', 'Longitude']].groupby(['Block']).mean().round(3)
print(rounded_lat_lon.head(10))
```

                      Latitude  Longitude
    Block                                
    0000X E 100TH PL    41.712    -87.622
    0000X E 100TH ST    41.713    -87.622
    0000X E 101ST PL    41.710    -87.622
    0000X E 101ST ST    41.711    -87.622
    0000X E 102ND PL    41.708    -87.622
    0000X E 102ND ST    41.709    -87.622
    0000X E 103RD PL    41.706    -87.622
    0000X E 103RD ST    41.707    -87.622
    0000X E 104TH ST    41.705    -87.622
    0000X E 105TH ST    41.703    -87.623
    


```python
# This is all of the geocoding currently in the CPD database. Next, we'll look at all of the other blocks in the database
# and see if there are any that we need to manually create geocodes for each
crime_file = 'Crimes_-_2001.csv'
cdf = pd.read_csv(crime_file, index_col=0)
block_count = cdf[['Block','Beat']].groupby(['Block']).count()
block_db = {i:row['Beat'] for i,row in block_count.iterrows()}
len(block_db)
```




    32726




```python
# Figure this out and then get it for other years
blocks_in_db = list(block_db.keys())
for block in blocks_in_db[:5]:
    print('{} has {} entries'.format(block,block_db[block]))
```

    0000X E 100 PL has 30 entries
    0000X E 100 ST has 38 entries
    0000X E 101 PL has 7 entries
    0000X E 101 ST has 24 entries
    0000X E 102 PL has 14 entries
    


```python
for year in range(2002,2018):
    crimes_file = 'Crimes_-_{}.csv'.format(year)
    cdf = pd.read_csv(crimes_file, index_col=0)
    block_count = cdf[['Block','Beat']].groupby(['Block']).count()
    for i,row in block_count.iterrows():
        if i in block_db:
            block_db[i] = block_db[i] + row['Beat']
        else:
            block_db[i] = row['Beat']
    print('{} db entries and {} crimes after {}'.format(len(block_db),sum(block_db.values()),year))
```

    52779 db entries and 972477 crimes after 2002
    54847 db entries and 1448395 crimes after 2003
    55671 db entries and 1917755 crimes after 2004
    56205 db entries and 2371442 crimes after 2005
    56563 db entries and 2819505 crimes after 2006
    56819 db entries and 3256467 crimes after 2007
    57071 db entries and 3683465 crimes after 2008
    57249 db entries and 4076063 crimes after 2009
    57491 db entries and 4446277 crimes after 2010
    57658 db entries and 4797923 crimes after 2011
    57871 db entries and 5133712 crimes after 2012
    58010 db entries and 5440543 crimes after 2013
    58161 db entries and 5715358 crimes after 2014
    58380 db entries and 5978822 crimes after 2015
    58782 db entries and 6246895 crimes after 2016
    59163 db entries and 6458840 crimes after 2017
    


```python
crime_count_s = pd.Series(block_db)
crime_count_s.head()
```




    0000X E 100 PL       44
    0000X E 100 ST       44
    0000X E 100TH PL    495
    0000X E 100TH ST    392
    0000X E 101 PL       10
    dtype: int64




```python
missing_blocks = set(crime_count_s.index)-set(rounded_lat_lon.index)
len(missing_blocks)
```




    29045




```python
len(rounded_lat_lon)
```




    30118




```python
# Even though almost half of the crime blocks are missing lat/longs, let's check to see our crime coverage
full_lat_lon = pd.concat([rounded_lat_lon, crime_count_s], axis=1)
full_lat_lon = full_lat_lon.rename(columns={0:'N'})
```


```python
full_lat_lon.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>N</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0000X E 100 PL</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>44</td>
    </tr>
    <tr>
      <th>0000X E 100 ST</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>44</td>
    </tr>
    <tr>
      <th>0000X E 100TH PL</th>
      <td>41.712</td>
      <td>-87.622</td>
      <td>495</td>
    </tr>
    <tr>
      <th>0000X E 100TH ST</th>
      <td>41.713</td>
      <td>-87.622</td>
      <td>392</td>
    </tr>
    <tr>
      <th>0000X E 101 PL</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
    </tr>
    <tr>
      <th>0000X E 101 ST</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>30</td>
    </tr>
    <tr>
      <th>0000X E 101ST PL</th>
      <td>41.710</td>
      <td>-87.622</td>
      <td>289</td>
    </tr>
    <tr>
      <th>0000X E 101ST ST</th>
      <td>41.711</td>
      <td>-87.622</td>
      <td>354</td>
    </tr>
    <tr>
      <th>0000X E 102 PL</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>19</td>
    </tr>
    <tr>
      <th>0000X E 102 ST</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
# It seems the issue is lack of standardization of addresses. Ideally we can find a way to match
full_lat_lon['ll_present']=~np.isnan(full_lat_lon['Latitude'])
full_lat_lon.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>N</th>
      <th>ll_present</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0000X E 100 PL</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>44</td>
      <td>False</td>
    </tr>
    <tr>
      <th>0000X E 100 ST</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>44</td>
      <td>False</td>
    </tr>
    <tr>
      <th>0000X E 100TH PL</th>
      <td>41.712</td>
      <td>-87.622</td>
      <td>495</td>
      <td>True</td>
    </tr>
    <tr>
      <th>0000X E 100TH ST</th>
      <td>41.713</td>
      <td>-87.622</td>
      <td>392</td>
      <td>True</td>
    </tr>
    <tr>
      <th>0000X E 101 PL</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>False</td>
    </tr>
    <tr>
      <th>0000X E 101 ST</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>30</td>
      <td>False</td>
    </tr>
    <tr>
      <th>0000X E 101ST PL</th>
      <td>41.710</td>
      <td>-87.622</td>
      <td>289</td>
      <td>True</td>
    </tr>
    <tr>
      <th>0000X E 101ST ST</th>
      <td>41.711</td>
      <td>-87.622</td>
      <td>354</td>
      <td>True</td>
    </tr>
    <tr>
      <th>0000X E 102 PL</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>19</td>
      <td>False</td>
    </tr>
    <tr>
      <th>0000X E 102 ST</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>16</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# It seems like the fuller entries tend to have "RD" or "TH" or "ST" or "ND" appended to the street names
# We can create a record of correct entries, strip the indices of those phrases and then use this as a lookup to correct
# the blocks missing those terms
df_full = full_lat_lon[full_lat_lon['ll_present_now']]
print('Original:')
print(df_full.head(10))
old_index = list(df_full.index)
print(old_index[:5])
new_index = [x.replace('RD ',' ').replace('TH ',' ').replace('ST ',' ').replace('ND ',' ') for x in old_index]
print(new_index[:5])
df_full = df_full.rename(index={old:new for old, new in zip(list(df_full.index),new_index)})
print('After:')
print(df_full.head(10))
```

    Original:
                      Latitude  Longitude    N  ll_present  ll_present_now
    0000X E 100 PL      41.712    -87.622   44       False            True
    0000X E 100 ST      41.713    -87.622   44       False            True
    0000X E 100TH PL    41.712    -87.622  495        True            True
    0000X E 100TH ST    41.713    -87.622  392        True            True
    0000X E 101 PL      41.710    -87.622   10       False            True
    0000X E 101 ST      41.711    -87.622   30       False            True
    0000X E 101ST PL    41.710    -87.622  289        True            True
    0000X E 101ST ST    41.711    -87.622  354        True            True
    0000X E 102 PL      41.708    -87.622   19       False            True
    0000X E 102 ST      41.709    -87.622   16       False            True
    ['0000X E 100 PL', '0000X E 100 ST', '0000X E 100TH PL', '0000X E 100TH ST', '0000X E 101 PL']
    ['0000X E 100 PL', '0000X E 100 ST', '0000X E 100 PL', '0000X E 100 ST', '0000X E 101 PL']
    After:
                    Latitude  Longitude    N  ll_present  ll_present_now
    0000X E 100 PL    41.712    -87.622   44       False            True
    0000X E 100 ST    41.713    -87.622   44       False            True
    0000X E 100 PL    41.712    -87.622  495        True            True
    0000X E 100 ST    41.713    -87.622  392        True            True
    0000X E 101 PL    41.710    -87.622   10       False            True
    0000X E 101 ST    41.711    -87.622   30       False            True
    0000X E 101 PL    41.710    -87.622  289        True            True
    0000X E 101 ST    41.711    -87.622  354        True            True
    0000X E 102 PL    41.708    -87.622   19       False            True
    0000X E 102 ST    41.709    -87.622   16       False            True
    


```python
'0000X E 104 ST' in df_full.index
entry = df_full.loc['0000X E 104 ST','Latitude']
entry
```




    0000X E 104 ST    41.705
    0000X E 104 ST    41.705
    Name: Latitude, dtype: float64




```python
for block in full_lat_lon.index:
    if not full_lat_lon.loc[block,'ll_present_now']:
        if block in df_full.index:
            full_lat_lon.loc[block,'Latitude'] = df_full.loc[block,'Latitude']
            full_lat_lon.loc[block,'Longitude'] = df_full.loc[block,'Longitude']
```


```python
# This import allows access to the Google API for geocoding
# With a free Server API key, you can run 2,500 requests per day
import googlemaps
gmaps = googlemaps.Client(key='insert_your_server_key_here')
```


```python
# RUN_FROM_HERE (We'll repeat these steps after running the ones above)
# Let's see now how much that helped
full_lat_lon['ll_present_now']=~np.isnan(full_lat_lon['Latitude'])
still_missing = full_lat_lon[~full_lat_lon['ll_present_now']]
len(still_missing)
```




    7136




```python
# we went from 29045 missing blocks down to 24305
# Now let's see what % of crime that is:
counts = full_lat_lon[['ll_present_now','N']].groupby(['ll_present_now']).sum()
print('{} missing crimes or {:2.0%} of total'.format(counts.loc[False,'N'],counts.loc[False,'N']/counts.loc[True,'N']))
```

    27218 missing crimes or 0% of total
    


```python
# So we've got 93% of all crimes geocoded. We can probably do better by focusing on the highest volume crimes
still_missing = still_missing.sort_values(by='N',ascending=False)
still_missing.head(40)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>N</th>
      <th>ll_present</th>
      <th>ll_present_now</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0000X W CHECKPOINT 2 ST</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>97</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>0000X W CHECKPOINT 1 ST</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>84</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>007XX E STEPHEN A DOUGLAS DR</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>50</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>0000X W CHECKPOINT 3 ST</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>45</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>032XX S LAKE SHORE DR SB</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>25</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>048XX S LAKE SHORE DR SB</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>23</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>031XX W 107TH ST</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>018XX N LAMON AVE</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>122XX S ASHLAND AVE</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>109XX S BELL AVE</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>033XX W REDFIELD DR</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>046XX N KIONA AVE</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>043XX S OAKLEY AVE</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>056XX S NORMAL AV</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>046XX S COTTAGE GROVE</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>018XX W 13TH ST</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>056XX W BERTEAU AVE</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>017XX E 57TH DR</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>057XX N KINGSDALE AVE</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>064XX S NORMANDY AVE</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>114XX S KEDZIE AVE</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>004XX W DIVERSEY AV</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>108XX S SACRAMENTO AVE</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>026XX E 127TH ST</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>118XX S STONY ISLAND AVE</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>017XX W AUGUSTA BV</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>066XX W MELROSE ST</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>027XX E 100TH ST</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>117XX S BELL AVE</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>047XX N LAWRENCE WILSON DR</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>005XX E MORGAN DR</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>054XX S SHERMAN PARK SD W</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>017XX W 14 PL</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>015XX W SUMMERDALE AVE</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>035XX N PANAMA AVE</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>053XX N PULASKI RD</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>066XX S NORMAL AV</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>046XX W 53RD ST</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>108XX S KING DR</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>024XX E 89TH ST</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
still_missing.iloc[:2430,:].groupby(['ll_present']).sum().loc[False,'N']
```




    19479




```python
sf = (6,1000) # This part can be modified for running in chunks (we're worried about api limits)
_old_blocks = list(still_missing.iloc[sf[0]:sf[1],:].index)
_new_blocks = [x.lstrip('0').replace('XX','50').replace('0000X','5') + ', CHICAGO, IL' for x in _old_blocks]
for old_block, new_block in zip(_old_blocks, _new_blocks):
    results = gmaps.geocode(new_block)
    if results:
        if 'geometry' in results[0]:
            this_ll = results[0]['geometry']['location']
            this_ll = [this_ll['lat'], this_ll['lng']]
            this_ll = [round(x,3) for x in this_ll]
            still_missing.loc[old_block,'Latitude'] = this_ll[0]
            still_missing.loc[old_block,'Longitude'] = this_ll[1]
print(len(still_missing[~np.isnan(still_missing['Latitude'])]))
```

    985
    


```python
# In the event of an error for API limits, run this to see if worth merging
print(len(still_missing[~np.isnan(still_missing['Latitude'])]))
```

    985
    


```python
# Only run the below of there are positive results above
these_results = still_missing[~np.isnan(still_missing['Latitude'])]
for i,row in these_results.iterrows():
    full_lat_lon.loc[i,'Latitude'] = row['Latitude']
    full_lat_lon.loc[i,'Longitude'] = row['Longitude']
```


```python
# Diagnostic here, but also start running again at the #RUN FROM HERE tag
full_lat_lon['ll_present_now']=~np.isnan(full_lat_lon['Latitude'])
counts = full_lat_lon[['ll_present_now','N']].groupby(['ll_present_now']).sum()
print('{} missing crimes or {:2.0%} of total'.format(counts.loc[False,'N'],counts.loc[False,'N']/counts.loc[True,'N']))
```

    16145 missing crimes or 0% of total
    


```python
full_lat_lon.to_csv('full_lat_lon_11_6_2017_more.csv')
```


```python
still_missing.head(20)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>N</th>
      <th>ll_present</th>
      <th>ll_present_now</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0000X W CHECKPOINT 2 ST</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>97</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>0000X W CHECKPOINT 1 ST</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>84</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>007XX E STEPHEN A DOUGLAS DR</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>50</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>0000X W CHECKPOINT 3 ST</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>45</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>032XX S LAKE SHORE DR SB</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>25</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>048XX S LAKE SHORE DR SB</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>23</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>031XX W 107TH ST</th>
      <td>41.699</td>
      <td>-87.701</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>018XX N LAMON AVE</th>
      <td>41.915</td>
      <td>-87.749</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>122XX S ASHLAND AVE</th>
      <td>41.801</td>
      <td>-87.665</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>109XX S BELL AVE</th>
      <td>41.694</td>
      <td>-87.676</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>033XX W REDFIELD DR</th>
      <td>41.765</td>
      <td>-87.703</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>046XX N KIONA AVE</th>
      <td>41.965</td>
      <td>-87.734</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>043XX S OAKLEY AVE</th>
      <td>41.815</td>
      <td>-87.683</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>056XX S NORMAL AV</th>
      <td>41.776</td>
      <td>-87.637</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>046XX S COTTAGE GROVE</th>
      <td>41.810</td>
      <td>-87.607</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>018XX W 13TH ST</th>
      <td>41.865</td>
      <td>-87.673</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>056XX W BERTEAU AVE</th>
      <td>41.957</td>
      <td>-87.769</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>017XX E 57TH DR</th>
      <td>41.792</td>
      <td>-87.587</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>057XX N KINGSDALE AVE</th>
      <td>41.985</td>
      <td>-87.746</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>064XX S NORMANDY AVE</th>
      <td>41.775</td>
      <td>-87.788</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# It appears by inspection that many of the remaining missed cases are based on improper abbreviations for BLVD (BL)
# AVE (AV) and PKWY (PW)
# We can use the same process used above with missing "ND" and "TH" phrases to try to determine the proper geocodes
```


```python
df_full = full_lat_lon[full_lat_lon['ll_present_now']]
print('Original:')
print(df_full.head(10))
old_index = list(df_full.index)
print(old_index[:5])
new_index = [x.replace(' BLVD',' BL').replace(' AVE',' AV').replace(' PKWY',' PW') for x in old_index]
print(new_index[:5])
df_full = df_full.rename(index={old:new for old, new in zip(list(df_full.index),new_index)})
print('After:')
print(df_full.head(10))
```

    Original:
                      Latitude  Longitude    N  ll_present  ll_present_now
    0000X E 100 PL      41.712    -87.622   44       False            True
    0000X E 100 ST      41.713    -87.622   44       False            True
    0000X E 100TH PL    41.712    -87.622  495        True            True
    0000X E 100TH ST    41.713    -87.622  392        True            True
    0000X E 101 PL      41.710    -87.622   10       False            True
    0000X E 101 ST      41.711    -87.622   30       False            True
    0000X E 101ST PL    41.710    -87.622  289        True            True
    0000X E 101ST ST    41.711    -87.622  354        True            True
    0000X E 102 PL      41.708    -87.622   19       False            True
    0000X E 102 ST      41.709    -87.622   16       False            True
    ['0000X E 100 PL', '0000X E 100 ST', '0000X E 100TH PL', '0000X E 100TH ST', '0000X E 101 PL']
    ['0000X E 100 PL', '0000X E 100 ST', '0000X E 100TH PL', '0000X E 100TH ST', '0000X E 101 PL']
    After:
                      Latitude  Longitude    N  ll_present  ll_present_now
    0000X E 100 PL      41.712    -87.622   44       False            True
    0000X E 100 ST      41.713    -87.622   44       False            True
    0000X E 100TH PL    41.712    -87.622  495        True            True
    0000X E 100TH ST    41.713    -87.622  392        True            True
    0000X E 101 PL      41.710    -87.622   10       False            True
    0000X E 101 ST      41.711    -87.622   30       False            True
    0000X E 101ST PL    41.710    -87.622  289        True            True
    0000X E 101ST ST    41.711    -87.622  354        True            True
    0000X E 102 PL      41.708    -87.622   19       False            True
    0000X E 102 ST      41.709    -87.622   16       False            True
    


```python
for block in full_lat_lon.index:
    if not full_lat_lon.loc[block,'ll_present_now']:
        if block in df_full.index:
            full_lat_lon.loc[block,'Latitude'] = df_full.loc[block,'Latitude']
            full_lat_lon.loc[block,'Longitude'] = df_full.loc[block,'Longitude']
```


```python
# Let's see how well that worked:
full_lat_lon['ll_present_now']=~np.isnan(full_lat_lon['Latitude'])
counts = full_lat_lon[['ll_present_now','N']].groupby(['ll_present_now']).sum()
print('{} missing crimes or {:3.1%} of total'.format(counts.loc[False,'N'],counts.loc[False,'N']/counts.loc[True,'N']))
```

    13997 missing crimes or 0.2% of total
    


```python
# We'll claim success!
# Note that the above cells were repeated multiple times over approximately 4,500 geocode calls in order to get to this result
```
