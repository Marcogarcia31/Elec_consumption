
## Project overview<p>&nbsp;</p>

* In order to meet the objective of null CO2 emissions by 2050, this project consists of building the model that would predict both the CO2 emissions and the energy consumption of non-residential buildings in Seattle<p>&nbsp;</p>

* Technically the project addresses regression problems with tabular data



* The overall performance of the fitted model is robust and the few features it needs as input allow a clear interpretation  <p>&nbsp;</p><p>&nbsp;</p>


### Workflow summary



* cleaned the data

* explored the data, in particular the relationship betweenn features and targets

* implemented feature engineering : selection, transformation and creation of features

* compared linear vs non-linear estimators on metrics using CV

* optimized XGBoost, RandomForest & SVR estimators using both GridSearchCV and RandomizedSearchCV

* predicted test samples target values with optimized XGBoost & interpreted both r2 score and feature importances<p>&nbsp;</p>

![](Images/energy_preds.png)<p>&nbsp;</p>


## Project installation

* use command pip install -r requirements.txt to install the dependencies

* the data is directly available [here](https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv).  The csv file should be stored in a folder named Data to reproduce code in the cleaning notebook

* the cleaned data ready for exploration and modelling is available in the project repository under the name exploration_data.csv and may be read directly by the modelling notebook
<p>&nbsp;</p>


## Detailed workflow<p>&nbsp;</p>


### Importing data<p>&nbsp;</p>

![](Images/Importing.png)<p>&nbsp;</p>