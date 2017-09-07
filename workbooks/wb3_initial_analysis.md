
### Building a model to predict the college persistence of high school alumni

## File 3: Initial analysis
<i>Author:</i> Matt Niksch
<p><i>Synopsis</i>: Prior to the start of this project, an initial model has been built to predict which alumni of a high school network will persist in college. The purpose of this project is to reinvestigate that model and potentially augment it with free text responses from students using natural language processing.
<p>While the ultimate goal for the network is to have alumni persist through 4 years of college and earn bachelor's degrees, we estimate over 2/3 of students who leave college do so prior to the start of sophomore year. As such, the focus of this project is on predicting persistence to sophomore year. The College Success field uses two standard definitions to discuss these results:
* <b>Retention</b>: defined as being enrolled one year later at the initial college you began at as a first-time, full-time freshman; technically, a student could have retention if they skip the second semester of freshman year, but we'll use a stricter definition requiring continuous enrollment to the 3rd semester or 4th quarter
* <b>Persistence</b>: similar to retention but allows for transfers; students persist as long as they stay in a college, at any college

# This workbook restates the initial analysis performed against predicted retention
Generally, the approach uses a logistic regression model with a variety of features to predict the "Retention3" label (making it to sophomore year of college


```python
import pandas as pd
import numpy as np

from modules.predictions import Prediction
from create_coefficients import process_survey_file
%matplotlib inline
import matplotlib.pyplot as plt

# First we need to load the data (see prior workbooks for details)
survey_data_file = 'inputs/Senior_Survey_Data.csv'
survey_key_file = 'inputs/Senior_Survey_Key.csv'
persistence_file = 'inputs/Persistence_Data.csv'

survey_df = process_survey_file(survey_data_file, survey_key_file)
main_df = pd.read_csv(persistence_file, encoding='cp1252', index_col=0)
main_df = pd.concat([main_df, survey_df], axis=1)
main_df.info()

#This combines the two main datafiles into a single frame
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 8345 entries, 20141948 to 50495984
    Data columns (total 59 columns):
    Id                                 8345 non-null object
    IsBlack                            8345 non-null int64
    IsLatino                           8345 non-null int64
    IsWhite                            8345 non-null int64
    IsAsian                            8345 non-null int64
    IsMinority                         8345 non-null int64
    IsLowIncome                        8184 non-null float64
    IsMale                             8345 non-null int64
    ACT                                8306 non-null float64
    GPA                                8345 non-null float64
    WGPA                               8345 non-null float64
    GPA_diff                           8345 non-null float64
    Class                              8345 non-null int64
    RandomSplit                        8345 non-null int64
    IsSpEd                             7995 non-null float64
    Campus                             8345 non-null object
    College attainment                 8345 non-null object
    Current Status                     8345 non-null object
    Initial PGR                        8345 non-null float64
    Initial IRR                        7197 non-null float64
    Initial NCES                       7267 non-null object
    IsMCPlus                           7267 non-null float64
    IsMC                               7267 non-null float64
    IsHC                               7267 non-null float64
    IsVC                               7267 non-null float64
    IsC                                7267 non-null float64
    IsNC                               7267 non-null float64
    Is2yr                              7267 non-null float64
    IsInitialHBCU                      7267 non-null float64
    Retention3                         5648 non-null float64
    Persistence3                       5648 non-null float64
    Retention5                         4282 non-null float64
    BA_in6                             699 non-null float64
    Retention semesters                8345 non-null int64
    BA completed in X Semesters        8345 non-null int64
    Initial College                    7295 non-null object
    Sem 2 College                      6915 non-null object
    Sem 3 College                      4723 non-null object
    Sem 4 College                      4496 non-null object
    Sem 5 College                      3098 non-null object
    Sem 6 College                      2976 non-null object
    Sem 7 College                      1921 non-null object
    Sem 8 College                      1878 non-null object
    Sem 9 College                      764 non-null object
    Sem 10 College                     692 non-null object
    Sem 11 College                     332 non-null object
    Sem 12 College                     296 non-null object
    HS_Class                           2776 non-null float64
    Performance_Avoidance              2733 non-null float64
    Self_Concept                       2774 non-null float64
    Academic_Identity                  2753 non-null float64
    Growth_Mindset_Self_Efficacy       2749 non-null float64
    Academic_Delay_of_Gratification    2723 non-null float64
    Intrinsic_Motivation               2754 non-null float64
    HS_Preparation                     2690 non-null float64
    Self_Regulation                    2734 non-null float64
    Support_Networks_School            2767 non-null float64
    Support_Networks_Family            2772 non-null float64
    Organization_Time_Management       2729 non-null float64
    dtypes: float64(32), int64(10), object(17)
    memory usage: 3.8+ MB
    


```python
# For some of the cases, I removed high frequency colleges under the theory (a) they would skew the data and (b) they
# had enough frequency we could conduct separate analyses specifically for those colleges
special_colleges = (
        (145600, 'University of Illinois at Chicago'),
        (145637, 'University of Illinois at Urbana-Champaign'),
        (149772, 'Western Illinois University'),
        (144209, 'City Colleges of Chicago-Harold Washington College'),
        (145813, 'Illinois State University'),
        (147776, 'Northeastern Illinois University'),
        (144218, 'City Colleges of Chicago-Wilbur Wright College'),
        (170301, 'Hope College'),
        (149222, 'Southern Illinois University Carbondale'),
        (148654, 'University of Illinois at Springfield'),
        (147341, 'Monmouth College'),
        (144892, 'Eastern Illinois University'),
        (144740, 'DePaul University'),
        (148496, 'Dominican University'),
        (147703, 'Northern Illinois University'),
        )
special_exclude = [str(x[0]) for x in special_colleges] # a list of college indices to exclude from some trials
```


```python
# This next block of code runs the prediction function (creates an object) for a basic case as an example

Prediction(main_df, #data source
                  [2013, 2014, 2015],     # years to use either for training or testing
                  ['GPA', 'Initial PGR',],  # features for this trial
                  'Persistence3',         # label
                  "GPA/GR for '13-14 ('15 test)", # description of the analysis
                  require=None,           # this input could "Require" we only focus on certain slices of data
                  remove=None,            # this input could "Remove" certain slices of data
            train=[2013, 2014])           # train on these years (other years will be for testing)

# After this code is run we can observe the main outcome stats:
```




    Case: GPA/GR for '13-14 ('15 test)
    [Coefficients]: ["[('GPA', 1.1048020313042533), ('Initial PGR', 1.3075388663569016), ('intercept', -2.6275486226494276)]"]
    [Train stats]: 75% positive, AOC: 56%, Score: 76%, N: 2447
    [Test stats ]: 76% positive, AOC: 53%, Score: 76%, N: 1366




```python
# Now we'll try the same analysis minus the high frequency colleges
Prediction(main_df, #data source
                  [2013, 2014, 2015],     # years to use either for training or testing
                  ['GPA', 'Initial PGR',],  # features for this trial
                  'Persistence3',         # label
                  "GPA/GR for '13-14 ('15 test) no big c", # description of the analysis
                  require=None,           # this input could "Require" we only focus on certain slices of data
                  remove=[('Initial NCES', special_exclude)], # DIFFERENCE from the prior run
            train=[2013, 2014])           # train on these years (other years will be for testing)

#Scores go up slightly with the high frequency colleges removed
```




    Case: GPA/GR for '13-14 ('15 test) no big c
    [Coefficients]: ["[('GPA', 0.9490192248401379), ('Initial PGR', 1.3226136003652624), ('intercept', -2.1059517128009615)]"]
    [Train stats]: 78% positive, AOC: 53%, Score: 78%, N: 1158
    [Test stats ]: 80% positive, AOC: 51%, Score: 79%, N: 656




```python
# Adding demographics helped, but we can still hopefully do better
# We'll attempt to split the colleges by their Barrons (selectivity) classification
# In each case, we'll try with 2013-2014 as training data and then also including 2015 (no test)
barrons_cases = [
        ('IsMCPlus', 'Most Competitive+'),
        ('IsMC', 'Most Competitive'),
        ('IsHC', 'Highly Competitive'),
        ('IsVC', 'Very Competitive'),
        ('IsC', 'Competitive'),
        ('IsNC', 'Noncompetitive'),
        ('Is2yr', '2 year'),
        ]
for field, label in barrons_cases:
    newP = Prediction(main_df, [2013, 2014, 2015], ['GPA', 'Initial PGR', 'IsMale', 'IsBlack', 'IsLatino'],
            'Persistence3', label+' (plus demographics no big colleges)',
            require=[(field, 1)], remove=[('Initial NCES', special_exclude)], train=[2013, 2014])
    print('{}\n'.format(newP))
    newP = Prediction(main_df, [2013, 2014, 2015], ['GPA', 'Initial PGR', 'IsMale', 'IsBlack', 'IsLatino'],
            'Persistence3', label+' (plus demographics no big colleges PLUS 2015)',
            require=[(field, 1)], remove=[('Initial NCES', special_exclude)], train=None)
    print('{}\n--------------------------------------------------------------------'.format(newP))
```

    Case: Most Competitive+ (plus demographics no big colleges)
    [Coefficients]: ["[('GPA', -1.8074806390059495), ('Initial PGR', -17.717251678314721), ('IsMale', -0.041837795126276035), ('IsBlack', -8.9108228192364454), ('IsLatino', -8.6519520793462181), ('intercept', 34.57765071607089)]"]
    [Train stats]: 94% positive, AOC: 50%, Score: 94%, N: 88
    [Test stats ]: 96% positive, AOC: 50%, Score: 96%, N: 46
    
    Case: Most Competitive+ (plus demographics no big colleges PLUS 2015)
    [Coefficients]: ["[('GPA', 1.0213342783589892), ('Initial PGR', -19.084549285897218), ('IsMale', 0.75490222170827104), ('IsBlack', -8.7291205468955102), ('IsLatino', -8.006504761514428), ('intercept', 24.37159805765259)]"]
    [Train stats]: 95% positive, AOC: 50%, Score: 95%, N: 134
    [Test stats ]: 95% positive, AOC: 50%, Score: 95%, N: 134
    --------------------------------------------------------------------
    Case: Most Competitive (plus demographics no big colleges)
    [Coefficients]: ["[('GPA', 1.0894211560206692), ('Initial PGR', 4.0737325429357298), ('IsMale', -0.4711986239583662), ('IsBlack', -6.0134437456778578), ('IsLatino', -6.7876459625352261), ('intercept', 1.5540496808505475)]"]
    [Train stats]: 86% positive, AOC: 49%, Score: 84%, N: 69
    [Test stats ]: 96% positive, AOC: 50%, Score: 96%, N: 26
    
    Case: Most Competitive (plus demographics no big colleges PLUS 2015)
    [Coefficients]: ["[('GPA', 1.557642740246679), ('Initial PGR', 4.2493906929019358), ('IsMale', -0.62820923568498555), ('IsBlack', 1.5753400879100141), ('IsLatino', 0.63387065951513577), ('intercept', -7.279787989712741)]"]
    [Train stats]: 88% positive, AOC: 54%, Score: 88%, N: 95
    [Test stats ]: 88% positive, AOC: 54%, Score: 88%, N: 95
    --------------------------------------------------------------------
    Case: Highly Competitive (plus demographics no big colleges)
    [Coefficients]: ["[('GPA', 1.5502363334044142), ('Initial PGR', 1.5687857421272382), ('IsMale', 0.21080119739048989), ('IsBlack', -0.088882542757638822), ('IsLatino', -0.34937012411313001), ('intercept', -4.109348673457653)]"]
    [Train stats]: 88% positive, AOC: 50%, Score: 88%, N: 199
    [Test stats ]: 94% positive, AOC: 50%, Score: 94%, N: 127
    
    Case: Highly Competitive (plus demographics no big colleges PLUS 2015)
    [Coefficients]: ["[('GPA', 1.7025557812269543), ('Initial PGR', 3.047535973398392), ('IsMale', 0.078171266695564262), ('IsBlack', -0.068733380450837253), ('IsLatino', -0.67478795900090882), ('intercept', -5.214482117665658)]"]
    [Train stats]: 91% positive, AOC: 52%, Score: 91%, N: 326
    [Test stats ]: 91% positive, AOC: 52%, Score: 91%, N: 326
    --------------------------------------------------------------------
    Case: Very Competitive (plus demographics no big colleges)
    [Coefficients]: ["[('GPA', 1.2265885975755839), ('Initial PGR', 1.2005506747428292), ('IsMale', -0.53404255823138869), ('IsBlack', -7.3608667721411702), ('IsLatino', -7.4623936908577759), ('intercept', 4.830166580579058)]"]
    [Train stats]: 84% positive, AOC: 51%, Score: 84%, N: 302
    [Test stats ]: 84% positive, AOC: 52%, Score: 85%, N: 146
    
    Case: Very Competitive (plus demographics no big colleges PLUS 2015)
    [Coefficients]: ["[('GPA', 1.1009705832284031), ('Initial PGR', 1.2753780700375101), ('IsMale', -0.56937953476598435), ('IsBlack', -7.6395765438298335), ('IsLatino', -7.9601492003296208), ('intercept', 5.579988661828363)]"]
    [Train stats]: 84% positive, AOC: 51%, Score: 84%, N: 448
    [Test stats ]: 84% positive, AOC: 51%, Score: 84%, N: 448
    --------------------------------------------------------------------
    Case: Competitive (plus demographics no big colleges)
    [Coefficients]: ["[('GPA', 0.78256024308698913), ('Initial PGR', 2.4163603878946311), ('IsMale', -0.51283603614449891), ('IsBlack', 0.80576983318816653), ('IsLatino', 0.57485799463263942), ('intercept', -2.5078029760142253)]"]
    [Train stats]: 71% positive, AOC: 50%, Score: 70%, N: 254
    [Test stats ]: 72% positive, AOC: 51%, Score: 72%, N: 121
    
    Case: Competitive (plus demographics no big colleges PLUS 2015)
    [Coefficients]: ["[('GPA', 0.89952660367600634), ('Initial PGR', 1.6475990003944356), ('IsMale', -0.33971656829526031), ('IsBlack', 0.45180849025673525), ('IsLatino', 0.29713609105202848), ('intercept', -2.342694929909492)]"]
    [Train stats]: 71% positive, AOC: 50%, Score: 70%, N: 375
    [Test stats ]: 71% positive, AOC: 50%, Score: 70%, N: 375
    --------------------------------------------------------------------
    Case: Noncompetitive (plus demographics no big colleges)
    [Coefficients]: ["[('GPA', 0.44479284452247453), ('Initial PGR', 1.5942739992856703), ('IsMale', -0.12349558474456011), ('IsBlack', 0.8393905320554379), ('IsLatino', 0.49825475548528209), ('intercept', -1.313087414217298)]"]
    [Train stats]: 66% positive, AOC: 53%, Score: 68%, N: 121
    [Test stats ]: 69% positive, AOC: 49%, Score: 68%, N: 114
    
    Case: Noncompetitive (plus demographics no big colleges PLUS 2015)
    [Coefficients]: ["[('GPA', 0.59000982841769167), ('Initial PGR', 0.62930747087595873), ('IsMale', -0.068510867000537864), ('IsBlack', 0.28255576266262283), ('IsLatino', -0.020027913465177995), ('intercept', -0.9406796777188825)]"]
    [Train stats]: 68% positive, AOC: 51%, Score: 68%, N: 235
    [Test stats ]: 68% positive, AOC: 51%, Score: 68%, N: 235
    --------------------------------------------------------------------
    Case: 2 year (plus demographics no big colleges)
    [Coefficients]: ["[('GPA', 0.98037566635310447), ('Initial PGR', 3.7418941050312711), ('IsMale', -0.68104437906833026), ('IsBlack', 1.4337699576211926), ('IsLatino', 2.0085622466670285), ('intercept', -4.558301940322747)]"]
    [Train stats]: 56% positive, AOC: 63%, Score: 64%, N: 119
    [Test stats ]: 55% positive, AOC: 60%, Score: 62%, N: 71
    
    Case: 2 year (plus demographics no big colleges PLUS 2015)
    [Coefficients]: ["[('GPA', 0.7143650875346983), ('Initial PGR', 1.5758733940508152), ('IsMale', -0.75065261521982585), ('IsBlack', 0.084232882338917028), ('IsLatino', 0.77052032884176824), ('intercept', -2.1029521924718937)]"]
    [Train stats]: 56% positive, AOC: 61%, Score: 63%, N: 190
    [Test stats ]: 56% positive, AOC: 61%, Score: 63%, N: 190
    --------------------------------------------------------------------
    

# The section above is focusing on generic colleges. Below, we'll focus on specific high frequency colleges


```python
# Now let's try for GPA only at the high frequency colleges
# For each college, we'll try (a) with just GPA, (b) with that plus demographics, and (c) throwing 2015 into the training set
for nces, school_name in special_colleges: # loop through each college
        requireA = ('Initial NCES', str(nces))
        text = "GPA for '13-14 ('15 test) at "+school_name
        newP = Prediction(main_df, [2013, 2014, 2015], ['GPA'],
            'Persistence3', text, require=[requireA], train=[2013, 2014])
        print('{}\n'.format(newP)) # the base case (a)
        newP = Prediction(main_df, [2013, 2014, 2015], ['GPA', 'IsMale', 'IsBlack', 'IsLatino'],
            'Persistence3', text+' (plus demographics)', require=[requireA], train=[2013, 2014])
        print('{}\n'.format(newP)) # plus demographics (b)
        newP = Prediction(main_df, [2013, 2014, 2015], ['GPA', 'IsMale', 'IsBlack', 'IsLatino'],
            'Persistence3', "GPA for '13-15 (no test) " +school_name+' (plus demographics)', require=[requireA], train=None)
        print(newP) # plus demographics and 2015 for training (c)
        print('-------------------------------------------------------\n')
```

    Case: GPA for '13-14 ('15 test) at University of Illinois at Chicago
    [Coefficients]: ["[('GPA', 0.75845800718996226), ('intercept', -1.1682238655236687)]"]
    [Train stats]: 76% positive, AOC: 50%, Score: 76%, N: 196
    [Test stats ]: 77% positive, AOC: 50%, Score: 77%, N: 159
    
    Case: GPA for '13-14 ('15 test) at University of Illinois at Chicago (plus demographics)
    [Coefficients]: ["[('GPA', 0.77945406635944825), ('IsMale', 0.038339367718936368), ('IsBlack', 1.6590986586036129), ('IsLatino', 0.16738134501962004), ('intercept', -1.4746376673693846)]"]
    [Train stats]: 76% positive, AOC: 50%, Score: 76%, N: 196
    [Test stats ]: 77% positive, AOC: 50%, Score: 77%, N: 159
    
    Case: GPA for '13-15 (no test)University of Illinois at Chicago (plus demographics)
    [Coefficients]: ["[('GPA', 0.92118788807809704), ('IsMale', -0.36152840190326524), ('IsBlack', 1.7256218703055601), ('IsLatino', 0.071744061641422163), ('intercept', -1.6340547035444424)]"]
    [Train stats]: 76% positive, AOC: 50%, Score: 76%, N: 355
    [Test stats ]: 76% positive, AOC: 50%, Score: 76%, N: 355
    -------------------------------------------------------
    
    Case: GPA for '13-14 ('15 test) at University of Illinois at Urbana-Champaign
    [Coefficients]: ["[('GPA', 4.2552410501297571), ('intercept', -11.544406321648347)]"]
    [Train stats]: 92% positive, AOC: 50%, Score: 92%, N: 174
    [Test stats ]: 87% positive, AOC: 54%, Score: 88%, N: 103
    
    Case: GPA for '13-14 ('15 test) at University of Illinois at Urbana-Champaign (plus demographics)
    [Coefficients]: ["[('GPA', 3.7110428235349011), ('IsMale', -0.42694993229129141), ('IsBlack', -8.2544509365992571), ('IsLatino', -6.81554945399325), ('intercept', -2.229809875938283)]"]
    [Train stats]: 92% positive, AOC: 49%, Score: 91%, N: 174
    [Test stats ]: 87% positive, AOC: 50%, Score: 87%, N: 103
    
    Case: GPA for '13-15 (no test)University of Illinois at Urbana-Champaign (plus demographics)
    [Coefficients]: ["[('GPA', 4.073118046010153), ('IsMale', -0.93452499468860883), ('IsBlack', -0.92631608510592744), ('IsLatino', -0.39529244705782535), ('intercept', -10.098018521024683)]"]
    [Train stats]: 90% positive, AOC: 54%, Score: 91%, N: 277
    [Test stats ]: 90% positive, AOC: 54%, Score: 91%, N: 277
    -------------------------------------------------------
    
    Case: GPA for '13-14 ('15 test) at Western Illinois University
    [Coefficients]: ["[('GPA', 0.03863200058730145), ('intercept', 0.8392531870625297)]"]
    [Train stats]: 72% positive, AOC: 50%, Score: 72%, N: 121
    [Test stats ]: 65% positive, AOC: 50%, Score: 65%, N: 52
    
    Case: GPA for '13-14 ('15 test) at Western Illinois University (plus demographics)
    [Coefficients]: ["[('GPA', -0.27926084755151248), ('IsMale', -0.53832589434969202), ('IsBlack', 0.22311918188155525), ('IsLatino', -0.073065194841457598), ('intercept', 1.8230157428907559)]"]
    [Train stats]: 72% positive, AOC: 50%, Score: 72%, N: 121
    [Test stats ]: 65% positive, AOC: 50%, Score: 65%, N: 52
    
    Case: GPA for '13-15 (no test)Western Illinois University (plus demographics)
    [Coefficients]: ["[('GPA', 0.2239520961518606), ('IsMale', -0.60306888230862099), ('IsBlack', 0.93132715229211371), ('IsLatino', 0.78748472641799849), ('intercept', -0.3014454819881238)]"]
    [Train stats]: 70% positive, AOC: 50%, Score: 69%, N: 173
    [Test stats ]: 70% positive, AOC: 50%, Score: 69%, N: 173
    -------------------------------------------------------
    
    Case: GPA for '13-14 ('15 test) at City Colleges of Chicago-Harold Washington College
    [Coefficients]: ["[('GPA', 1.0951184254159105), ('intercept', -2.3472451423634473)]"]
    [Train stats]: 59% positive, AOC: 58%, Score: 61%, N: 111
    [Test stats ]: 58% positive, AOC: 56%, Score: 56%, N: 43
    
    Case: GPA for '13-14 ('15 test) at City Colleges of Chicago-Harold Washington College (plus demographics)
    [Coefficients]: ["[('GPA', 1.0013785982606644), ('IsMale', -0.23953082200525455), ('IsBlack', 0.23221192282972367), ('IsLatino', 0.47113023597504028), ('intercept', -2.417235971202911)]"]
    [Train stats]: 59% positive, AOC: 55%, Score: 59%, N: 111
    [Test stats ]: 58% positive, AOC: 65%, Score: 65%, N: 43
    
    Case: GPA for '13-15 (no test)City Colleges of Chicago-Harold Washington College (plus demographics)
    [Coefficients]: ["[('GPA', 0.91273496171505408), ('IsMale', -0.43026485449959123), ('IsBlack', -0.57056700203448596), ('IsLatino', -0.18708008740619911), ('intercept', -1.3734392932616872)]"]
    [Train stats]: 59% positive, AOC: 57%, Score: 61%, N: 154
    [Test stats ]: 59% positive, AOC: 57%, Score: 61%, N: 154
    -------------------------------------------------------
    
    Case: GPA for '13-14 ('15 test) at Illinois State University
    [Coefficients]: ["[('GPA', 3.8388578128400477), ('intercept', -9.73302091335591)]"]
    [Train stats]: 75% positive, AOC: 59%, Score: 76%, N: 101
    [Test stats ]: 73% positive, AOC: 49%, Score: 67%, N: 96
    
    Case: GPA for '13-14 ('15 test) at Illinois State University (plus demographics)
    [Coefficients]: ["[('GPA', 4.2797477918237643), ('IsMale', 0.2304373414729681), ('IsBlack', 1.5446701978571844), ('IsLatino', 1.2136181920478155), ('intercept', -12.397354862631174)]"]
    [Train stats]: 75% positive, AOC: 63%, Score: 79%, N: 101
    [Test stats ]: 73% positive, AOC: 50%, Score: 66%, N: 96
    
    Case: GPA for '13-15 (no test)Illinois State University (plus demographics)
    [Coefficients]: ["[('GPA', 2.2107198348422981), ('IsMale', 0.16893737979136289), ('IsBlack', -0.42844927196415894), ('IsLatino', -0.77488288423807228), ('intercept', -4.764506594823099)]"]
    [Train stats]: 74% positive, AOC: 51%, Score: 73%, N: 197
    [Test stats ]: 74% positive, AOC: 51%, Score: 73%, N: 197
    -------------------------------------------------------
    
    Case: GPA for '13-14 ('15 test) at Northeastern Illinois University
    [Coefficients]: ["[('GPA', 1.2748808441361545), ('intercept', -2.9413398700608866)]"]
    [Train stats]: 52% positive, AOC: 61%, Score: 61%, N: 98
    [Test stats ]: 53% positive, AOC: 65%, Score: 65%, N: 43
    
    Case: GPA for '13-14 ('15 test) at Northeastern Illinois University (plus demographics)
    [Coefficients]: ["[('GPA', 1.2956362899190037), ('IsMale', -0.35628297558895905), ('IsBlack', -0.85121805693109243), ('IsLatino', -1.577992359183354), ('intercept', -1.3439742619983623)]"]
    [Train stats]: 52% positive, AOC: 63%, Score: 63%, N: 98
    [Test stats ]: 53% positive, AOC: 70%, Score: 70%, N: 43
    
    Case: GPA for '13-15 (no test)Northeastern Illinois University (plus demographics)
    [Coefficients]: ["[('GPA', 1.4733052984377526), ('IsMale', -0.3930851540952941), ('IsBlack', -0.86283352977006944), ('IsLatino', -1.5941696456080861), ('intercept', -1.7113741922295862)]"]
    [Train stats]: 52% positive, AOC: 64%, Score: 64%, N: 141
    [Test stats ]: 52% positive, AOC: 64%, Score: 64%, N: 141
    -------------------------------------------------------
    
    Case: GPA for '13-14 ('15 test) at City Colleges of Chicago-Wilbur Wright College
    [Coefficients]: ["[('GPA', 0.7597349398579627), ('intercept', -1.9568340850427968)]"]
    [Train stats]: 47% positive, AOC: 56%, Score: 57%, N: 70
    [Test stats ]: 74% positive, AOC: 68%, Score: 61%, N: 23
    
    Case: GPA for '13-14 ('15 test) at City Colleges of Chicago-Wilbur Wright College (plus demographics)
    [Coefficients]: ["[('GPA', 0.45569645009926674), ('IsMale', -0.44645663860329926), ('IsBlack', 9.1917868148937956), ('IsLatino', 9.7886483528931656), ('intercept', -10.705963502514894)]"]
    [Train stats]: 47% positive, AOC: 60%, Score: 60%, N: 70
    [Test stats ]: 74% positive, AOC: 60%, Score: 57%, N: 23
    
    Case: GPA for '13-15 (no test)City Colleges of Chicago-Wilbur Wright College (plus demographics)
    [Coefficients]: ["[('GPA', 1.0228178119139264), ('IsMale', -0.1603056826710878), ('IsBlack', 0.87555757836454029), ('IsLatino', 0.58334678563562581), ('intercept', -2.865562322856006)]"]
    [Train stats]: 54% positive, AOC: 61%, Score: 61%, N: 93
    [Test stats ]: 54% positive, AOC: 61%, Score: 61%, N: 93
    -------------------------------------------------------
    
    Case: GPA for '13-14 ('15 test) at Hope College
    [Coefficients]: ["[('GPA', 3.0720623587563045), ('intercept', -7.590291918939713)]"]
    [Train stats]: 84% positive, AOC: 50%, Score: 84%, N: 67
    [Test stats ]: 89% positive, AOC: 50%, Score: 89%, N: 19
    
    Case: GPA for '13-14 ('15 test) at Hope College (plus demographics)
    [Coefficients]: ["[('GPA', 3.2693492285846419), ('IsMale', -0.014026850286721453), ('IsBlack', -2.5593000970828137), ('IsLatino', -2.8431218040525228), ('intercept', -5.402421901135423)]"]
    [Train stats]: 84% positive, AOC: 50%, Score: 84%, N: 67
    [Test stats ]: 89% positive, AOC: 50%, Score: 89%, N: 19
    
    Case: GPA for '13-15 (no test)Hope College (plus demographics)
    [Coefficients]: ["[('GPA', 2.9396257885570973), ('IsMale', 0.10142250873286898), ('IsBlack', -2.2679347608153866), ('IsLatino', -2.4899435992424834), ('intercept', -4.757878360057743)]"]
    [Train stats]: 85% positive, AOC: 50%, Score: 85%, N: 86
    [Test stats ]: 85% positive, AOC: 50%, Score: 85%, N: 86
    -------------------------------------------------------
    
    Case: GPA for '13-14 ('15 test) at Southern Illinois University Carbondale
    [Coefficients]: ["[('GPA', 1.3476361665480228), ('intercept', -3.360650604093952)]"]
    [Train stats]: 48% positive, AOC: 61%, Score: 62%, N: 65
    [Test stats ]: 48% positive, AOC: 66%, Score: 67%, N: 27
    
    Case: GPA for '13-14 ('15 test) at Southern Illinois University Carbondale (plus demographics)
    [Coefficients]: ["[('GPA', 1.3353374566564802), ('IsMale', -0.21153899498995701), ('IsBlack', -1.2014272570770435), ('IsLatino', -0.9364512575684536), ('intercept', -2.137878514645488)]"]
    [Train stats]: 48% positive, AOC: 61%, Score: 62%, N: 65
    [Test stats ]: 48% positive, AOC: 63%, Score: 63%, N: 27
    
    Case: GPA for '13-15 (no test)Southern Illinois University Carbondale (plus demographics)
    [Coefficients]: ["[('GPA', 1.4522530587486366), ('IsMale', 0.040661019602590845), ('IsBlack', -1.3951855000587248), ('IsLatino', -0.99681780945701126), ('intercept', -2.3920033095157334)]"]
    [Train stats]: 48% positive, AOC: 59%, Score: 60%, N: 92
    [Test stats ]: 48% positive, AOC: 59%, Score: 60%, N: 92
    -------------------------------------------------------
    
    Case: GPA for '13-14 ('15 test) at University of Illinois at Springfield
    [Coefficients]: ["[('GPA', 1.494203484102675), ('intercept', -2.3833582431464495)]"]
    [Train stats]: 85% positive, AOC: 50%, Score: 85%, N: 53
    [Test stats ]: 64% positive, AOC: 50%, Score: 64%, N: 28
    
    Case: GPA for '13-14 ('15 test) at University of Illinois at Springfield (plus demographics)
    [Coefficients]: ["[('GPA', 1.7334855262553477), ('IsMale', 1.1186990023661869), ('IsBlack', -6.3291491059739498), ('IsLatino', -7.051526472186727), ('intercept', 3.2746258762222276)]"]
    [Train stats]: 85% positive, AOC: 50%, Score: 85%, N: 53
    [Test stats ]: 64% positive, AOC: 50%, Score: 64%, N: 28
    
    Case: GPA for '13-15 (no test)University of Illinois at Springfield (plus demographics)
    [Coefficients]: ["[('GPA', 2.6338538043519151), ('IsMale', 0.32866038496531574), ('IsBlack', -8.1735226718133536), ('IsLatino', -8.2333470010394194), ('intercept', 2.074439033303843)]"]
    [Train stats]: 78% positive, AOC: 51%, Score: 77%, N: 81
    [Test stats ]: 78% positive, AOC: 51%, Score: 77%, N: 81
    -------------------------------------------------------
    
    Case: GPA for '13-14 ('15 test) at Monmouth College
    [Coefficients]: ["[('GPA', 2.0026863645738624), ('intercept', -4.050103262706133)]"]
    [Train stats]: 83% positive, AOC: 50%, Score: 83%, N: 48
    [Test stats ]: 75% positive, AOC: 50%, Score: 75%, N: 12
    
    Case: GPA for '13-14 ('15 test) at Monmouth College (plus demographics)
    [Coefficients]: ["[('GPA', 1.9471019895902439), ('IsMale', -0.13105454972957176), ('IsBlack', -9.302736697063839), ('IsLatino', -7.9533930187410817), ('intercept', 4.5147021962525855)]"]
    [Train stats]: 83% positive, AOC: 56%, Score: 85%, N: 48
    [Test stats ]: 75% positive, AOC: 50%, Score: 75%, N: 12
    
    Case: GPA for '13-15 (no test)Monmouth College (plus demographics)
    [Coefficients]: ["[('GPA', 2.4541625354703736), ('IsMale', -0.30299238729384054), ('IsBlack', -8.7911108913480192), ('IsLatino', -7.6232337372887029), ('intercept', 2.7514679561298108)]"]
    [Train stats]: 82% positive, AOC: 58%, Score: 83%, N: 60
    [Test stats ]: 82% positive, AOC: 58%, Score: 83%, N: 60
    -------------------------------------------------------
    
    Case: GPA for '13-14 ('15 test) at Eastern Illinois University
    [Coefficients]: ["[('GPA', -1.273840237051356), ('intercept', 5.197586987032425)]"]
    [Train stats]: 83% positive, AOC: 50%, Score: 83%, N: 47
    [Test stats ]: 72% positive, AOC: 50%, Score: 72%, N: 40
    
    Case: GPA for '13-14 ('15 test) at Eastern Illinois University (plus demographics)
    [Coefficients]: ["[('GPA', -1.2986280580930687), ('IsMale', -0.3325090487677036), ('IsBlack', 1.9332196116849452), ('IsLatino', 1.5918393978899079), ('intercept', 3.525059009574857)]"]
    [Train stats]: 83% positive, AOC: 50%, Score: 83%, N: 47
    [Test stats ]: 72% positive, AOC: 50%, Score: 72%, N: 40
    
    Case: GPA for '13-15 (no test)Eastern Illinois University (plus demographics)
    [Coefficients]: ["[('GPA', -0.35262000894129525), ('IsMale', -0.094834788546906051), ('IsBlack', 0.70709586923044065), ('IsLatino', 0.8474591344444935), ('intercept', 1.5545550036749438)]"]
    [Train stats]: 78% positive, AOC: 50%, Score: 78%, N: 87
    [Test stats ]: 78% positive, AOC: 50%, Score: 78%, N: 87
    -------------------------------------------------------
    
    Case: GPA for '13-14 ('15 test) at DePaul University
    [Coefficients]: ["[('GPA', 2.6083606116733429), ('intercept', -6.6215414421708285)]"]
    [Train stats]: 79% positive, AOC: 52%, Score: 77%, N: 47
    [Test stats ]: 100% positive, AOC: -100%, Score: 92%, N: 13
    
    Case: GPA for '13-14 ('15 test) at DePaul University (plus demographics)
    [Coefficients]: ["[('GPA', 7.6949451659583321), ('IsMale', 4.7420340564444032), ('IsBlack', 2.8168323901379426), ('IsLatino', 1.5707772077673756), ('intercept', -25.07999667192226)]"]
    [Train stats]: 79% positive, AOC: 84%, Score: 91%, N: 47
    [Test stats ]: 100% positive, AOC: -100%, Score: 77%, N: 13
    
    Case: GPA for '13-15 (no test)DePaul University (plus demographics)
    [Coefficients]: ["[('GPA', 5.9378374255684712), ('IsMale', 3.7266194979451215), ('IsBlack', 2.1974408000768832), ('IsLatino', 1.6462058843218814), ('intercept', -19.24509398836435)]"]
    [Train stats]: 83% positive, AOC: 69%, Score: 88%, N: 60
    [Test stats ]: 83% positive, AOC: 69%, Score: 88%, N: 60
    -------------------------------------------------------
    
    Case: GPA for '13-14 ('15 test) at Dominican University
    [Coefficients]: ["[('GPA', 1.1934423386734774), ('intercept', -2.624279263242289)]"]
    [Train stats]: 70% positive, AOC: 52%, Score: 70%, N: 46
    [Test stats ]: 67% positive, AOC: 50%, Score: 67%, N: 33
    
    Case: GPA for '13-14 ('15 test) at Dominican University (plus demographics)
    [Coefficients]: ["[('GPA', 1.2770461762742453), ('IsMale', -0.58671847252855458), ('IsBlack', -1.5216804475884198), ('IsLatino', -0.83993748547461244), ('intercept', -1.7609315907251748)]"]
    [Train stats]: 70% positive, AOC: 54%, Score: 70%, N: 46
    [Test stats ]: 67% positive, AOC: 55%, Score: 70%, N: 33
    
    Case: GPA for '13-15 (no test)Dominican University (plus demographics)
    [Coefficients]: ["[('GPA', 0.92006090631443238), ('IsMale', -0.84273331800660556), ('IsBlack', -1.7108090705457131), ('IsLatino', -0.6425172904198706), ('intercept', -0.8388454035833608)]"]
    [Train stats]: 68% positive, AOC: 54%, Score: 68%, N: 79
    [Test stats ]: 68% positive, AOC: 54%, Score: 68%, N: 79
    -------------------------------------------------------
    
    Case: GPA for '13-14 ('15 test) at Northern Illinois University
    [Coefficients]: ["[('GPA', 0.65636303741022828), ('intercept', -1.1095904991002024)]"]
    [Train stats]: 67% positive, AOC: 50%, Score: 67%, N: 45
    [Test stats ]: 68% positive, AOC: 50%, Score: 68%, N: 19
    
    Case: GPA for '13-14 ('15 test) at Northern Illinois University (plus demographics)
    [Coefficients]: ["[('GPA', 0.62851865285401054), ('IsMale', 0.019427230843116143), ('IsBlack', -0.16221801016288234), ('IsLatino', -0.53725586653697255), ('intercept', -0.69947387669986)]"]
    [Train stats]: 67% positive, AOC: 50%, Score: 67%, N: 45
    [Test stats ]: 68% positive, AOC: 50%, Score: 68%, N: 19
    
    Case: GPA for '13-15 (no test)Northern Illinois University (plus demographics)
    [Coefficients]: ["[('GPA', 1.6064717519105065), ('IsMale', 0.020875246808274916), ('IsBlack', -7.8942460043842217), ('IsLatino', -8.1515388416198089), ('intercept', 4.31052799212182)]"]
    [Train stats]: 67% positive, AOC: 62%, Score: 72%, N: 64
    [Test stats ]: 67% positive, AOC: 62%, Score: 72%, N: 64
    -------------------------------------------------------
    
    

## Questions? I've left off a final set of analyses on only the 2015 data where I tried to estimate the impact of the various survey questions


```python

```
