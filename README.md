![Exoplanets](Images/exoplanets.jpg)

In this project, machine learning models were created to classify candidate exoplanets from the exoplanet dataset.  
The machine models used for analysis were SVC and Logistic Regression.  After training the machine models and using hyperparameter tuning, 
the Logistic Regression model produced a slightly better score (0.8850259225373589) than the SVC model (0.8690149435803599).  Thus, the 
Logistic Regression model should be used for classifying candidate exoplanets.


# Jupyter Notebook: Machine Learning Code

```python
import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
```

# Read the CSV and Perform Basic Data Cleaning


```python
df = pd.read_csv("cumulative.csv")
df = df.drop(columns=["rowid", "kepid", "kepoi_name", "kepler_name", "koi_pdisposition", "koi_score", "koi_tce_delivname"])
# Drop the null columns where all values are null
df = df.dropna(axis='columns', how='all')
# Drop the null rows
df = df.dropna()
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>koi_disposition</th>
      <th>koi_fpflag_nt</th>
      <th>koi_fpflag_ss</th>
      <th>koi_fpflag_co</th>
      <th>koi_fpflag_ec</th>
      <th>koi_period</th>
      <th>koi_period_err1</th>
      <th>koi_period_err2</th>
      <th>koi_time0bk</th>
      <th>koi_time0bk_err1</th>
      <th>...</th>
      <th>koi_steff_err2</th>
      <th>koi_slogg</th>
      <th>koi_slogg_err1</th>
      <th>koi_slogg_err2</th>
      <th>koi_srad</th>
      <th>koi_srad_err1</th>
      <th>koi_srad_err2</th>
      <th>ra</th>
      <th>dec</th>
      <th>koi_kepmag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CONFIRMED</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9.488036</td>
      <td>2.775000e-05</td>
      <td>-2.775000e-05</td>
      <td>170.538750</td>
      <td>0.002160</td>
      <td>...</td>
      <td>-81.0</td>
      <td>4.467</td>
      <td>0.064</td>
      <td>-0.096</td>
      <td>0.927</td>
      <td>0.105</td>
      <td>-0.061</td>
      <td>291.93423</td>
      <td>48.141651</td>
      <td>15.347</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CONFIRMED</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>54.418383</td>
      <td>2.479000e-04</td>
      <td>-2.479000e-04</td>
      <td>162.513840</td>
      <td>0.003520</td>
      <td>...</td>
      <td>-81.0</td>
      <td>4.467</td>
      <td>0.064</td>
      <td>-0.096</td>
      <td>0.927</td>
      <td>0.105</td>
      <td>-0.061</td>
      <td>291.93423</td>
      <td>48.141651</td>
      <td>15.347</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FALSE POSITIVE</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>19.899140</td>
      <td>1.494000e-05</td>
      <td>-1.494000e-05</td>
      <td>175.850252</td>
      <td>0.000581</td>
      <td>...</td>
      <td>-176.0</td>
      <td>4.544</td>
      <td>0.044</td>
      <td>-0.176</td>
      <td>0.868</td>
      <td>0.233</td>
      <td>-0.078</td>
      <td>297.00482</td>
      <td>48.134129</td>
      <td>15.436</td>
    </tr>
    <tr>
      <th>3</th>
      <td>FALSE POSITIVE</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1.736952</td>
      <td>2.630000e-07</td>
      <td>-2.630000e-07</td>
      <td>170.307565</td>
      <td>0.000115</td>
      <td>...</td>
      <td>-174.0</td>
      <td>4.564</td>
      <td>0.053</td>
      <td>-0.168</td>
      <td>0.791</td>
      <td>0.201</td>
      <td>-0.067</td>
      <td>285.53461</td>
      <td>48.285210</td>
      <td>15.597</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CONFIRMED</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2.525592</td>
      <td>3.761000e-06</td>
      <td>-3.761000e-06</td>
      <td>171.595550</td>
      <td>0.001130</td>
      <td>...</td>
      <td>-211.0</td>
      <td>4.438</td>
      <td>0.070</td>
      <td>-0.210</td>
      <td>1.046</td>
      <td>0.334</td>
      <td>-0.133</td>
      <td>288.75488</td>
      <td>48.226200</td>
      <td>15.509</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 41 columns</p>
</div>



# Create a Train Test Split

Use `koi_disposition` for the y values


```python
from sklearn.model_selection import train_test_split

target = df["koi_disposition"]
data = df.drop(columns=['koi_disposition'])
```


```python
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=1, stratify=target)
```


```python
X_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>koi_fpflag_nt</th>
      <th>koi_fpflag_ss</th>
      <th>koi_fpflag_co</th>
      <th>koi_fpflag_ec</th>
      <th>koi_period</th>
      <th>koi_period_err1</th>
      <th>koi_period_err2</th>
      <th>koi_time0bk</th>
      <th>koi_time0bk_err1</th>
      <th>koi_time0bk_err2</th>
      <th>...</th>
      <th>koi_steff_err2</th>
      <th>koi_slogg</th>
      <th>koi_slogg_err1</th>
      <th>koi_slogg_err2</th>
      <th>koi_srad</th>
      <th>koi_srad_err1</th>
      <th>koi_srad_err2</th>
      <th>ra</th>
      <th>dec</th>
      <th>koi_kepmag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5964</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>252.044440</td>
      <td>0.027490</td>
      <td>-0.027490</td>
      <td>265.2010</td>
      <td>0.0494</td>
      <td>-0.0494</td>
      <td>...</td>
      <td>-136.0</td>
      <td>4.621</td>
      <td>0.041</td>
      <td>-0.035</td>
      <td>0.664</td>
      <td>0.057</td>
      <td>-0.059</td>
      <td>292.79022</td>
      <td>41.948639</td>
      <td>15.884</td>
    </tr>
    <tr>
      <th>9410</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>371.518520</td>
      <td>0.015790</td>
      <td>-0.015790</td>
      <td>317.6836</td>
      <td>0.0339</td>
      <td>-0.0339</td>
      <td>...</td>
      <td>-206.0</td>
      <td>4.377</td>
      <td>0.101</td>
      <td>-0.203</td>
      <td>1.089</td>
      <td>0.364</td>
      <td>-0.145</td>
      <td>293.06400</td>
      <td>45.034210</td>
      <td>13.731</td>
    </tr>
    <tr>
      <th>4204</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>8.038670</td>
      <td>0.000114</td>
      <td>-0.000114</td>
      <td>135.3098</td>
      <td>0.0123</td>
      <td>-0.0123</td>
      <td>...</td>
      <td>-181.0</td>
      <td>4.485</td>
      <td>0.050</td>
      <td>-0.200</td>
      <td>0.975</td>
      <td>0.282</td>
      <td>-0.101</td>
      <td>290.51785</td>
      <td>41.238762</td>
      <td>14.999</td>
    </tr>
    <tr>
      <th>5933</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>18.782160</td>
      <td>0.000406</td>
      <td>-0.000406</td>
      <td>147.8508</td>
      <td>0.0148</td>
      <td>-0.0148</td>
      <td>...</td>
      <td>-167.0</td>
      <td>4.488</td>
      <td>0.048</td>
      <td>-0.290</td>
      <td>0.940</td>
      <td>0.386</td>
      <td>-0.087</td>
      <td>291.76413</td>
      <td>41.860130</td>
      <td>14.043</td>
    </tr>
    <tr>
      <th>6996</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>12.429716</td>
      <td>0.000472</td>
      <td>-0.000472</td>
      <td>141.2846</td>
      <td>0.0420</td>
      <td>-0.0420</td>
      <td>...</td>
      <td>-200.0</td>
      <td>4.534</td>
      <td>0.037</td>
      <td>-0.213</td>
      <td>0.905</td>
      <td>0.281</td>
      <td>-0.088</td>
      <td>297.52072</td>
      <td>40.585419</td>
      <td>15.842</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 40 columns</p>
</div>



# Pre-processing

Scale the data using the MinMaxScaler


```python
from sklearn.preprocessing import MinMaxScaler
X_scaler = MinMaxScaler().fit(X_train)

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
```

# Train the Support Vector Machine


```python
# Create SVC model
from sklearn.svm import SVC
SVCmodel = SVC(kernel='linear')

# Train the model
SVCmodel.fit(X_train_scaled, y_train)

# Print scores
print(f"Training Data Score: {SVCmodel.score(X_train_scaled, y_train)}")
print(f"Testing Data Score: {SVCmodel.score(X_test_scaled, y_test)}")
```

    Training Data Score: 0.8479719426654467
    Testing Data Score: 0.8462946020128088
    


```python
# Create Logistic Regression model
from sklearn.linear_model import LogisticRegression
model_log = LogisticRegression()

# Train the model
model_log.fit(X_train_scaled, y_train)

# Print scores
print(f"Training Data Score: {model_log.score(X_train_scaled, y_train)}")
print(f"Testing Data Score: {model_log.score(X_test_scaled, y_test)}")
```

    Training Data Score: 0.8443122903324184
    Testing Data Score: 0.8394327538883806
    

# Hyperparameter Tuning

Use `GridSearchCV` to tune the `C` and `gamma` parameters


```python
# Create the GridSearch estimator and parameters for SVC model
from sklearn.model_selection import GridSearchCV
svc_param_grid = {'C': [1, 5, 10],
              'gamma': [0.0001, 0.001, 0.01],
              'kernel': ['linear']}
svc_grid = GridSearchCV(SVCmodel, svc_param_grid, verbose=3)
```


```python
# Fit the model using the grid search estimator
# This will take the SVC model and try each combination of parameters
svc_grid.fit(X_train_scaled, y_train)
```

    Fitting 3 folds for each of 9 candidates, totalling 27 fits
    [CV] C=1, gamma=0.0001, kernel=linear ................................
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    

    [CV]  C=1, gamma=0.0001, kernel=linear, score=0.8399634202103338, total=   0.3s
    [CV] C=1, gamma=0.0001, kernel=linear ................................
    

    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.5s remaining:    0.0s
    

    [CV]  C=1, gamma=0.0001, kernel=linear, score=0.8508691674290942, total=   0.3s
    [CV] C=1, gamma=0.0001, kernel=linear ................................
    

    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    1.1s remaining:    0.0s
    

    [CV]  C=1, gamma=0.0001, kernel=linear, score=0.8361556064073227, total=   0.3s
    [CV] C=1, gamma=0.001, kernel=linear .................................
    [CV]  C=1, gamma=0.001, kernel=linear, score=0.8399634202103338, total=   0.3s
    [CV] C=1, gamma=0.001, kernel=linear .................................
    [CV]  C=1, gamma=0.001, kernel=linear, score=0.8508691674290942, total=   0.3s
    [CV] C=1, gamma=0.001, kernel=linear .................................
    [CV]  C=1, gamma=0.001, kernel=linear, score=0.8361556064073227, total=   0.3s
    [CV] C=1, gamma=0.01, kernel=linear ..................................
    [CV]  C=1, gamma=0.01, kernel=linear, score=0.8399634202103338, total=   0.3s
    [CV] C=1, gamma=0.01, kernel=linear ..................................
    [CV]  C=1, gamma=0.01, kernel=linear, score=0.8508691674290942, total=   0.3s
    [CV] C=1, gamma=0.01, kernel=linear ..................................
    [CV]  C=1, gamma=0.01, kernel=linear, score=0.8361556064073227, total=   0.3s
    [CV] C=5, gamma=0.0001, kernel=linear ................................
    [CV]  C=5, gamma=0.0001, kernel=linear, score=0.8582533150434385, total=   0.3s
    [CV] C=5, gamma=0.0001, kernel=linear ................................
    [CV]  C=5, gamma=0.0001, kernel=linear, score=0.869167429094236, total=   0.2s
    [CV] C=5, gamma=0.0001, kernel=linear ................................
    [CV]  C=5, gamma=0.0001, kernel=linear, score=0.8617848970251716, total=   0.2s
    [CV] C=5, gamma=0.001, kernel=linear .................................
    [CV]  C=5, gamma=0.001, kernel=linear, score=0.8582533150434385, total=   0.3s
    [CV] C=5, gamma=0.001, kernel=linear .................................
    [CV]  C=5, gamma=0.001, kernel=linear, score=0.869167429094236, total=   0.2s
    [CV] C=5, gamma=0.001, kernel=linear .................................
    [CV]  C=5, gamma=0.001, kernel=linear, score=0.8617848970251716, total=   0.2s
    [CV] C=5, gamma=0.01, kernel=linear ..................................
    [CV]  C=5, gamma=0.01, kernel=linear, score=0.8582533150434385, total=   0.3s
    [CV] C=5, gamma=0.01, kernel=linear ..................................
    [CV]  C=5, gamma=0.01, kernel=linear, score=0.869167429094236, total=   0.2s
    [CV] C=5, gamma=0.01, kernel=linear ..................................
    [CV]  C=5, gamma=0.01, kernel=linear, score=0.8617848970251716, total=   0.2s
    [CV] C=10, gamma=0.0001, kernel=linear ...............................
    [CV]  C=10, gamma=0.0001, kernel=linear, score=0.8651120256058528, total=   0.3s
    [CV] C=10, gamma=0.0001, kernel=linear ...............................
    [CV]  C=10, gamma=0.0001, kernel=linear, score=0.8746569075937786, total=   0.3s
    [CV] C=10, gamma=0.0001, kernel=linear ...............................
    [CV]  C=10, gamma=0.0001, kernel=linear, score=0.8672768878718535, total=   0.3s
    [CV] C=10, gamma=0.001, kernel=linear ................................
    [CV]  C=10, gamma=0.001, kernel=linear, score=0.8651120256058528, total=   0.3s
    [CV] C=10, gamma=0.001, kernel=linear ................................
    [CV]  C=10, gamma=0.001, kernel=linear, score=0.8746569075937786, total=   0.2s
    [CV] C=10, gamma=0.001, kernel=linear ................................
    [CV]  C=10, gamma=0.001, kernel=linear, score=0.8672768878718535, total=   0.2s
    [CV] C=10, gamma=0.01, kernel=linear .................................
    [CV]  C=10, gamma=0.01, kernel=linear, score=0.8651120256058528, total=   0.3s
    [CV] C=10, gamma=0.01, kernel=linear .................................
    [CV]  C=10, gamma=0.01, kernel=linear, score=0.8746569075937786, total=   0.2s
    [CV] C=10, gamma=0.01, kernel=linear .................................
    [CV]  C=10, gamma=0.01, kernel=linear, score=0.8672768878718535, total=   0.3s
    

    [Parallel(n_jobs=1)]: Done  27 out of  27 | elapsed:   15.8s finished
    




    GridSearchCV(cv='warn', error_score='raise-deprecating',
           estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='linear', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False),
           fit_params=None, iid='warn', n_jobs=None,
           param_grid={'C': [1, 5, 10], 'gamma': [0.0001, 0.001, 0.01], 'kernel': ['linear']},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=3)




```python
# Print best parameters and best scores for SVC model
print(svc_grid.best_params_)
print(svc_grid.best_score_)
```

    {'C': 10, 'gamma': 0.0001, 'kernel': 'linear'}
    0.8690149435803599
    


```python
# Create the GridSearch estimator and parameters for Logistic Regression model
from sklearn.model_selection import GridSearchCV

logistic_param_grid = {"penalty": ['l1', 'l2'],
              "C": np.logspace(0, 4, 10)}
logistic_grid = GridSearchCV(model_log, logistic_param_grid, cv=5, verbose=3)
```


```python
# Fit the model using the grid search estimator
# This will take the Logistic Regression model and try each combination of parameters
logistic_grid.fit(X_train, y_train)
```

    Fitting 5 folds for each of 20 candidates, totalling 100 fits
    [CV] C=1.0, penalty=l1 ...............................................
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    

    [CV] ...... C=1.0, penalty=l1, score=0.8796648895658796, total=   0.7s
    [CV] C=1.0, penalty=l1 ...............................................
    

    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.7s remaining:    0.0s
    

    [CV] ....... C=1.0, penalty=l1, score=0.881859756097561, total=   3.4s
    [CV] C=1.0, penalty=l1 ...............................................
    

    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    4.3s remaining:    0.0s
    

    [CV] ...... C=1.0, penalty=l1, score=0.8887195121951219, total=   1.5s
    [CV] C=1.0, penalty=l1 ...............................................
    [CV] ...... C=1.0, penalty=l1, score=0.8924485125858124, total=   2.1s
    [CV] C=1.0, penalty=l1 ...............................................
    [CV] ....... C=1.0, penalty=l1, score=0.867175572519084, total=   1.1s
    [CV] C=1.0, penalty=l2 ...............................................
    [CV] ....... C=1.0, penalty=l2, score=0.674028941355674, total=   0.6s
    [CV] C=1.0, penalty=l2 ...............................................
    [CV] ...... C=1.0, penalty=l2, score=0.6829268292682927, total=   0.6s
    [CV] C=1.0, penalty=l2 ...............................................
    [CV] ...... C=1.0, penalty=l2, score=0.6509146341463414, total=   0.5s
    [CV] C=1.0, penalty=l2 ...............................................
    [CV] ...... C=1.0, penalty=l2, score=0.6422578184591915, total=   0.5s
    [CV] C=1.0, penalty=l2 ...............................................
    [CV] ....... C=1.0, penalty=l2, score=0.666412213740458, total=   0.7s
    [CV] C=2.7825594022071245, penalty=l1 ................................
    [CV]  C=2.7825594022071245, penalty=l1, score=0.8796648895658796, total=   0.9s
    [CV] C=2.7825594022071245, penalty=l1 ................................
    [CV]  C=2.7825594022071245, penalty=l1, score=0.8833841463414634, total=   2.9s
    [CV] C=2.7825594022071245, penalty=l1 ................................
    [CV]  C=2.7825594022071245, penalty=l1, score=0.8887195121951219, total=   1.6s
    [CV] C=2.7825594022071245, penalty=l1 ................................
    [CV]  C=2.7825594022071245, penalty=l1, score=0.8924485125858124, total=   2.5s
    [CV] C=2.7825594022071245, penalty=l1 ................................
    [CV]  C=2.7825594022071245, penalty=l1, score=0.8694656488549618, total=   1.3s
    [CV] C=2.7825594022071245, penalty=l2 ................................
    [CV]  C=2.7825594022071245, penalty=l2, score=0.6884996191926885, total=   0.6s
    [CV] C=2.7825594022071245, penalty=l2 ................................
    [CV]  C=2.7825594022071245, penalty=l2, score=0.6417682926829268, total=   0.5s
    [CV] C=2.7825594022071245, penalty=l2 ................................
    [CV]  C=2.7825594022071245, penalty=l2, score=0.6547256097560976, total=   0.6s
    [CV] C=2.7825594022071245, penalty=l2 ................................
    [CV]  C=2.7825594022071245, penalty=l2, score=0.6651411136536994, total=   0.6s
    [CV] C=2.7825594022071245, penalty=l2 ................................
    [CV]  C=2.7825594022071245, penalty=l2, score=0.6511450381679389, total=   0.6s
    [CV] C=7.742636826811269, penalty=l1 .................................
    [CV]  C=7.742636826811269, penalty=l1, score=0.8819497334348819, total=   0.5s
    [CV] C=7.742636826811269, penalty=l1 .................................
    [CV]  C=7.742636826811269, penalty=l1, score=0.8810975609756098, total=   2.1s
    [CV] C=7.742636826811269, penalty=l1 .................................
    [CV]  C=7.742636826811269, penalty=l1, score=0.8902439024390244, total=   2.4s
    [CV] C=7.742636826811269, penalty=l1 .................................
    [CV]  C=7.742636826811269, penalty=l1, score=0.8939740655987796, total=   2.8s
    [CV] C=7.742636826811269, penalty=l1 .................................
    [CV]  C=7.742636826811269, penalty=l1, score=0.8709923664122138, total=   1.4s
    [CV] C=7.742636826811269, penalty=l2 .................................
    [CV]  C=7.742636826811269, penalty=l2, score=0.6763137852246763, total=   0.7s
    [CV] C=7.742636826811269, penalty=l2 .................................
    [CV]  C=7.742636826811269, penalty=l2, score=0.6791158536585366, total=   0.7s
    [CV] C=7.742636826811269, penalty=l2 .................................
    [CV]  C=7.742636826811269, penalty=l2, score=0.6547256097560976, total=   0.7s
    [CV] C=7.742636826811269, penalty=l2 .................................
    [CV]  C=7.742636826811269, penalty=l2, score=0.6666666666666666, total=   0.6s
    [CV] C=7.742636826811269, penalty=l2 .................................
    [CV]  C=7.742636826811269, penalty=l2, score=0.6305343511450382, total=   0.6s
    [CV] C=21.544346900318832, penalty=l1 ................................
    [CV]  C=21.544346900318832, penalty=l1, score=0.8819497334348819, total=   0.4s
    [CV] C=21.544346900318832, penalty=l1 ................................
    [CV]  C=21.544346900318832, penalty=l1, score=0.8795731707317073, total=   2.9s
    [CV] C=21.544346900318832, penalty=l1 ................................
    [CV]  C=21.544346900318832, penalty=l1, score=0.8932926829268293, total=   3.6s
    [CV] C=21.544346900318832, penalty=l1 ................................
    [CV]  C=21.544346900318832, penalty=l1, score=0.8947368421052632, total=   2.5s
    [CV] C=21.544346900318832, penalty=l1 ................................
    [CV]  C=21.544346900318832, penalty=l1, score=0.8717557251908397, total=   1.6s
    [CV] C=21.544346900318832, penalty=l2 ................................
    [CV]  C=21.544346900318832, penalty=l2, score=0.6725057121096725, total=   0.6s
    [CV] C=21.544346900318832, penalty=l2 ................................
    [CV]  C=21.544346900318832, penalty=l2, score=0.6737804878048781, total=   0.6s
    [CV] C=21.544346900318832, penalty=l2 ................................
    [CV]  C=21.544346900318832, penalty=l2, score=0.6463414634146342, total=   0.5s
    [CV] C=21.544346900318832, penalty=l2 ................................
    [CV]  C=21.544346900318832, penalty=l2, score=0.6598016781083142, total=   0.5s
    [CV] C=21.544346900318832, penalty=l2 ................................
    [CV]  C=21.544346900318832, penalty=l2, score=0.6389312977099236, total=   0.6s
    [CV] C=59.94842503189409, penalty=l1 .................................
    [CV]  C=59.94842503189409, penalty=l1, score=0.8796648895658796, total=   0.9s
    [CV] C=59.94842503189409, penalty=l1 .................................
    [CV]  C=59.94842503189409, penalty=l1, score=0.8810975609756098, total=   2.4s
    [CV] C=59.94842503189409, penalty=l1 .................................
    [CV]  C=59.94842503189409, penalty=l1, score=0.8910060975609756, total=   2.5s
    [CV] C=59.94842503189409, penalty=l1 .................................
    [CV]  C=59.94842503189409, penalty=l1, score=0.8954996186117468, total=  13.5s
    [CV] C=59.94842503189409, penalty=l1 .................................
    [CV]  C=59.94842503189409, penalty=l1, score=0.8725190839694656, total=  18.9s
    [CV] C=59.94842503189409, penalty=l2 .................................
    [CV]  C=59.94842503189409, penalty=l2, score=0.670982482863671, total=   0.5s
    [CV] C=59.94842503189409, penalty=l2 .................................
    [CV]  C=59.94842503189409, penalty=l2, score=0.6592987804878049, total=   0.7s
    [CV] C=59.94842503189409, penalty=l2 .................................
    [CV]  C=59.94842503189409, penalty=l2, score=0.6737804878048781, total=   0.6s
    [CV] C=59.94842503189409, penalty=l2 .................................
    [CV]  C=59.94842503189409, penalty=l2, score=0.6552250190694127, total=   0.7s
    [CV] C=59.94842503189409, penalty=l2 .................................
    [CV]  C=59.94842503189409, penalty=l2, score=0.6618320610687023, total=   0.7s
    [CV] C=166.81005372000593, penalty=l1 ................................
    [CV]  C=166.81005372000593, penalty=l1, score=0.8804265041888805, total=   0.6s
    [CV] C=166.81005372000593, penalty=l1 ................................
    [CV]  C=166.81005372000593, penalty=l1, score=0.8810975609756098, total=   2.5s
    [CV] C=166.81005372000593, penalty=l1 ................................
    [CV]  C=166.81005372000593, penalty=l1, score=0.8910060975609756, total=   2.7s
    [CV] C=166.81005372000593, penalty=l1 ................................
    [CV]  C=166.81005372000593, penalty=l1, score=0.8954996186117468, total=  15.1s
    [CV] C=166.81005372000593, penalty=l1 ................................
    [CV]  C=166.81005372000593, penalty=l1, score=0.8725190839694656, total=  18.6s
    [CV] C=166.81005372000593, penalty=l2 ................................
    [CV]  C=166.81005372000593, penalty=l2, score=0.6770753998476771, total=   0.7s
    [CV] C=166.81005372000593, penalty=l2 ................................
    [CV]  C=166.81005372000593, penalty=l2, score=0.6432926829268293, total=   0.8s
    [CV] C=166.81005372000593, penalty=l2 ................................
    [CV]  C=166.81005372000593, penalty=l2, score=0.6676829268292683, total=   0.6s
    [CV] C=166.81005372000593, penalty=l2 ................................
    [CV]  C=166.81005372000593, penalty=l2, score=0.6575133485888635, total=   0.7s
    [CV] C=166.81005372000593, penalty=l2 ................................
    [CV]  C=166.81005372000593, penalty=l2, score=0.6534351145038167, total=   0.7s
    [CV] C=464.15888336127773, penalty=l1 ................................
    [CV]  C=464.15888336127773, penalty=l1, score=0.8804265041888805, total=   1.4s
    [CV] C=464.15888336127773, penalty=l1 ................................
    [CV]  C=464.15888336127773, penalty=l1, score=0.8833841463414634, total=   2.4s
    [CV] C=464.15888336127773, penalty=l1 ................................
    [CV]  C=464.15888336127773, penalty=l1, score=0.8910060975609756, total=   2.9s
    [CV] C=464.15888336127773, penalty=l1 ................................
    [CV]  C=464.15888336127773, penalty=l1, score=0.8954996186117468, total=  14.8s
    [CV] C=464.15888336127773, penalty=l1 ................................
    [CV]  C=464.15888336127773, penalty=l1, score=0.8732824427480916, total=  19.1s
    [CV] C=464.15888336127773, penalty=l2 ................................
    [CV]  C=464.15888336127773, penalty=l2, score=0.6831683168316832, total=   0.7s
    [CV] C=464.15888336127773, penalty=l2 ................................
    [CV] .. C=464.15888336127773, penalty=l2, score=0.65625, total=   0.6s
    [CV] C=464.15888336127773, penalty=l2 ................................
    [CV]  C=464.15888336127773, penalty=l2, score=0.6570121951219512, total=   0.6s
    [CV] C=464.15888336127773, penalty=l2 ................................
    [CV]  C=464.15888336127773, penalty=l2, score=0.6636155606407322, total=   0.6s
    [CV] C=464.15888336127773, penalty=l2 ................................
    [CV]  C=464.15888336127773, penalty=l2, score=0.6312977099236641, total=   0.5s
    [CV] C=1291.5496650148827, penalty=l1 ................................
    [CV]  C=1291.5496650148827, penalty=l1, score=0.8811881188118812, total=   0.9s
    [CV] C=1291.5496650148827, penalty=l1 ................................
    [CV]  C=1291.5496650148827, penalty=l1, score=0.8826219512195121, total=   2.5s
    [CV] C=1291.5496650148827, penalty=l1 ................................
    [CV]  C=1291.5496650148827, penalty=l1, score=0.8932926829268293, total=   3.4s
    [CV] C=1291.5496650148827, penalty=l1 ................................
    [CV]  C=1291.5496650148827, penalty=l1, score=0.8947368421052632, total=  13.5s
    [CV] C=1291.5496650148827, penalty=l1 ................................
    [CV]  C=1291.5496650148827, penalty=l1, score=0.8732824427480916, total=  18.5s
    [CV] C=1291.5496650148827, penalty=l2 ................................
    [CV]  C=1291.5496650148827, penalty=l2, score=0.6808834729626809, total=   0.6s
    [CV] C=1291.5496650148827, penalty=l2 ................................
    [CV]  C=1291.5496650148827, penalty=l2, score=0.6730182926829268, total=   0.6s
    [CV] C=1291.5496650148827, penalty=l2 ................................
    [CV]  C=1291.5496650148827, penalty=l2, score=0.6600609756097561, total=   0.6s
    [CV] C=1291.5496650148827, penalty=l2 ................................
    [CV]  C=1291.5496650148827, penalty=l2, score=0.6796338672768879, total=   0.6s
    [CV] C=1291.5496650148827, penalty=l2 ................................
    [CV]  C=1291.5496650148827, penalty=l2, score=0.649618320610687, total=   0.5s
    [CV] C=3593.813663804626, penalty=l1 .................................
    [CV]  C=3593.813663804626, penalty=l1, score=0.8804265041888805, total=   0.6s
    [CV] C=3593.813663804626, penalty=l1 .................................
    [CV]  C=3593.813663804626, penalty=l1, score=0.8826219512195121, total=   2.3s
    [CV] C=3593.813663804626, penalty=l1 .................................
    [CV]  C=3593.813663804626, penalty=l1, score=0.8910060975609756, total=   2.6s
    [CV] C=3593.813663804626, penalty=l1 .................................
    [CV]  C=3593.813663804626, penalty=l1, score=0.8947368421052632, total=  16.8s
    [CV] C=3593.813663804626, penalty=l1 .................................
    [CV]  C=3593.813663804626, penalty=l1, score=0.8732824427480916, total=  19.6s
    [CV] C=3593.813663804626, penalty=l2 .................................
    [CV]  C=3593.813663804626, penalty=l2, score=0.6816450875856817, total=   0.5s
    [CV] C=3593.813663804626, penalty=l2 .................................
    [CV]  C=3593.813663804626, penalty=l2, score=0.6692073170731707, total=   0.6s
    [CV] C=3593.813663804626, penalty=l2 .................................
    [CV]  C=3593.813663804626, penalty=l2, score=0.6516768292682927, total=   0.6s
    [CV] C=3593.813663804626, penalty=l2 .................................
    [CV]  C=3593.813663804626, penalty=l2, score=0.635392829900839, total=   0.6s
    [CV] C=3593.813663804626, penalty=l2 .................................
    [CV]  C=3593.813663804626, penalty=l2, score=0.6572519083969466, total=   0.7s
    [CV] C=10000.0, penalty=l1 ...........................................
    [CV] .. C=10000.0, penalty=l1, score=0.8819497334348819, total=   0.6s
    [CV] C=10000.0, penalty=l1 ...........................................
    [CV] .. C=10000.0, penalty=l1, score=0.8826219512195121, total=   2.8s
    [CV] C=10000.0, penalty=l1 ...........................................
    [CV] .. C=10000.0, penalty=l1, score=0.8910060975609756, total=   2.5s
    [CV] C=10000.0, penalty=l1 ...........................................
    [CV] .. C=10000.0, penalty=l1, score=0.8954996186117468, total=  17.3s
    [CV] C=10000.0, penalty=l1 ...........................................
    [CV] .. C=10000.0, penalty=l1, score=0.8732824427480916, total=  18.3s
    [CV] C=10000.0, penalty=l2 ...........................................
    [CV] .. C=10000.0, penalty=l2, score=0.6801218583396801, total=   0.5s
    [CV] C=10000.0, penalty=l2 ...........................................
    [CV] .. C=10000.0, penalty=l2, score=0.6821646341463414, total=   0.6s
    [CV] C=10000.0, penalty=l2 ...........................................
    [CV] .. C=10000.0, penalty=l2, score=0.6463414634146342, total=   0.6s
    [CV] C=10000.0, penalty=l2 ...........................................
    [CV] .. C=10000.0, penalty=l2, score=0.6598016781083142, total=   0.6s
    [CV] C=10000.0, penalty=l2 ...........................................
    [CV] .. C=10000.0, penalty=l2, score=0.6183206106870229, total=   0.5s
    

    [Parallel(n_jobs=1)]: Done 100 out of 100 | elapsed:  5.3min finished
    




    GridSearchCV(cv=5, error_score='raise-deprecating',
           estimator=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='warn',
              n_jobs=None, penalty='l2', random_state=None, solver='warn',
              tol=0.0001, verbose=0, warm_start=False),
           fit_params=None, iid='warn', n_jobs=None,
           param_grid={'penalty': ['l1', 'l2'], 'C': array([1.00000e+00, 2.78256e+00, 7.74264e+00, 2.15443e+01, 5.99484e+01,
           1.66810e+02, 4.64159e+02, 1.29155e+03, 3.59381e+03, 1.00000e+04])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=3)




```python
# Print best paramaters and best scores for Logistic Regression model
print(logistic_grid.best_params_)
print(logistic_grid.best_score_)
```

    {'C': 1291.5496650148827, 'penalty': 'l1'}
    0.8850259225373589
    


```python
# Compare SVC model and Logistic Regression model scores

print("SVC model")
print(svc_grid.best_params_)
print(svc_grid.best_score_)
print("----------------")
print("Logistic Regression model")
print(logistic_grid.best_params_)
print(logistic_grid.best_score_)
```

    SVC model
    {'C': 10, 'gamma': 0.0001, 'kernel': 'linear'}
    0.8690149435803599
    ----------------
    Logistic Regression model
    {'C': 1291.5496650148827, 'penalty': 'l1'}
    0.8850259225373589
    
