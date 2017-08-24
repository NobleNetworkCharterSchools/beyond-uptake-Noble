
# Building a model to predict the college persistence of high school alumni
<i>Author:</i> Matt Niksch
<p><i>Synopsis</i>: Prior to the start of this project, an initial model has been built to predict which alumni of a high school network will persist in college. The purpose of this project is to reinvestigate that model and potentially augment it with free text responses from students using natural language processing.


```python
import pandas as pd
import numpy as np
```

## Exploration of survey data
One source of potential features for this analysis is a set of survey questions given to students who have graduated from Noble over the last 3 years. This is potentially very useful, but this particular data suffers from a number of limitations:
* Only the graduates from the Classes of 2015, 2016, and 2017 took the survey
* In 2015 and 2016 in particular, a handful of campuses did not administer the survey
* The questions have remained the same every year for consistency, but a handful of the Likert style questions have different directions for agreement (i.e. within the questions for one construct, some questions measure highest agreement with a 5 and others with a 1 on a 1-5 scale.)


```python
survey_data_file = 'inputs/Senior_Survey_Data.csv' # Raw survey responses
survey_key_file = 'inputs/Senior_Survey_Key.csv' # Metadata about the survey

sdf = pd.read_csv(survey_data_file, encoding='cp1252', index_col=0)
skf = pd.read_csv(survey_key_file, encoding='cp1252')
```


```python
sdf.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HS_Class</th>
      <th>N</th>
      <th>Self_Concept0</th>
      <th>Self_Concept1</th>
      <th>Self_Concept2</th>
      <th>Self_Concept3</th>
      <th>Self_Concept4</th>
      <th>Self_Concept5</th>
      <th>Self_Concept6</th>
      <th>Self_Concept7</th>
      <th>...</th>
      <th>Performance_Avoidance1</th>
      <th>Performance_Avoidance2</th>
      <th>Performance_Avoidance3</th>
      <th>Performance_Avoidance4</th>
      <th>Performance_Avoidance5</th>
      <th>Self_Regulation1</th>
      <th>Self_Regulation2</th>
      <th>Self_Regulation3</th>
      <th>Self_Regulation4</th>
      <th>HS_Preparation</th>
    </tr>
    <tr>
      <th>SID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34600104</th>
      <td>2016</td>
      <td>50</td>
      <td>Average</td>
      <td>Average</td>
      <td>Average</td>
      <td>Above average</td>
      <td>Average</td>
      <td>Average</td>
      <td>Average</td>
      <td>Average</td>
      <td>...</td>
      <td>A little true</td>
      <td>Somewhat true</td>
      <td>Not at all true</td>
      <td>Not at all true</td>
      <td>Not at all true</td>
      <td>A little like me</td>
      <td>Somewhat like me</td>
      <td>Somewhat like me</td>
      <td>A little like me</td>
      <td>Somewhat well</td>
    </tr>
    <tr>
      <th>38994980</th>
      <td>2016</td>
      <td>51</td>
      <td>Average</td>
      <td>Above average</td>
      <td>Above average</td>
      <td>Above average</td>
      <td>Above average</td>
      <td>Above average</td>
      <td>Above average</td>
      <td>Above average</td>
      <td>...</td>
      <td>Not at all true</td>
      <td>Not at all true</td>
      <td>Not at all true</td>
      <td>Not at all true</td>
      <td>Not at all true</td>
      <td>Not at all like me</td>
      <td>Not at all like me</td>
      <td>Not at all like me</td>
      <td>Not at all like me</td>
      <td>Somewhat well</td>
    </tr>
    <tr>
      <th>39650495</th>
      <td>2016</td>
      <td>49</td>
      <td>Average</td>
      <td>Above average</td>
      <td>Above average</td>
      <td>Above average</td>
      <td>Average</td>
      <td>Above average</td>
      <td>Average</td>
      <td>Average</td>
      <td>...</td>
      <td>A little true</td>
      <td>Not at all true</td>
      <td>A little true</td>
      <td>Not at all true</td>
      <td>Mostly true</td>
      <td>A little like me</td>
      <td>Not at all like me</td>
      <td>Not at all like me</td>
      <td>A little like me</td>
      <td>Somewhat well</td>
    </tr>
    <tr>
      <th>39663376</th>
      <td>2016</td>
      <td>49</td>
      <td>Average</td>
      <td>Above average</td>
      <td>Average</td>
      <td>Average</td>
      <td>Average</td>
      <td>Average</td>
      <td>Average</td>
      <td>Below average</td>
      <td>...</td>
      <td>Not at all true</td>
      <td>Not at all true</td>
      <td>Not at all true</td>
      <td>A little true</td>
      <td>A little true</td>
      <td>A little like me</td>
      <td>A little like me</td>
      <td>A little like me</td>
      <td>Not at all like me</td>
      <td>Somewhat well</td>
    </tr>
    <tr>
      <th>39665123</th>
      <td>2016</td>
      <td>50</td>
      <td>Above average</td>
      <td>Average</td>
      <td>Above average</td>
      <td>Average</td>
      <td>Average</td>
      <td>Average</td>
      <td>Average</td>
      <td>Below average</td>
      <td>...</td>
      <td>A little true</td>
      <td>Somewhat true</td>
      <td>Not at all true</td>
      <td>Not at all true</td>
      <td>A little true</td>
      <td>Not at all like me</td>
      <td>Not at all like me</td>
      <td>Not at all like me</td>
      <td>Not at all like me</td>
      <td>Very well</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 52 columns</p>
</div>




```python
skf.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Key</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Self_Concept0</td>
      <td>Below average</td>
      <td>Not Used</td>
      <td>Average</td>
      <td>Not Used</td>
      <td>Above average</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Self_Concept1</td>
      <td>Below average</td>
      <td>Not Used</td>
      <td>Average</td>
      <td>Not Used</td>
      <td>Above average</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Self_Concept2</td>
      <td>Below average</td>
      <td>Not Used</td>
      <td>Average</td>
      <td>Not Used</td>
      <td>Above average</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Self_Concept3</td>
      <td>Below average</td>
      <td>Not Used</td>
      <td>Average</td>
      <td>Not Used</td>
      <td>Above average</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Self_Concept4</td>
      <td>Below average</td>
      <td>Not Used</td>
      <td>Average</td>
      <td>Not Used</td>
      <td>Above average</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
