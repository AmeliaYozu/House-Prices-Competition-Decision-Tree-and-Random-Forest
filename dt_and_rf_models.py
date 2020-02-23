import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

data_y = data["SalePrice"]
data_X = data.drop(columns = "SalePrice")

## Data Clean:
#fillna "object" missing value columns with 'Unknown'
test_categorical = test.select_dtypes("object")
test_categorical_cols = test_categorical.columns
test[test_categorical_cols] = test_categorical.fillna("Unkown")

data_X_categorical = data_X.select_dtypes("object")
data_X_categorical_cols = data_X_categorical.columns
data_X[data_X_categorical_cols] = data_X_categorical.fillna("Unkown")

#fillna "float64" missing value with imputer
my_imputer = SimpleImputer()

test_numerical = test.select_dtypes(exclude="object")
test_numerical_cols = test_numerical.columns
test[test_numerical_cols] = my_imputer.fit_transform(test_numerical)

data_X_numerical = data_X.select_dtypes(exclude="object")
data_X_numerical_cols = data_X_numerical.columns
data_X[data_X_numerical_cols] = my_imputer.fit_transform(data_X_numerical)

#Categorical Label Encode:
le = LabelEncoder()
test[test_categorical_cols] = test[test_categorical_cols].apply(lambda col: le.fit_transform(col), axis=0)
data_X[data_X_categorical_cols] = data_X[data_X_categorical_cols].apply(lambda col: le.fit_transform(col), axis=0)

(train_X, val_X, train_y, val_y) = train_test_split(data_X, data_y)

#Decison tree model training
# nodes = [10,20,50,100,200,500,1000]
# for n in nodes:
#     dt_model = DecisionTreeRegressor(max_leaf_nodes=n,criterion='mae',random_state=1)
#     dt_model.fit(train_X, train_y)
#     preds = dt_model.predict(val_X)
#     mae = mean_absolute_error(preds, val_y)
#     print("leaf nodes {} : MAE {}".format(n, mae))

# max_leaf_nodes = 100 ~ minimum
# leaf nodes 10 : MAE 25525.227397260274
# leaf nodes 20 : MAE 24756.156164383563
# leaf nodes 50 : MAE 23642.038356164383
# leaf nodes 100 : MAE 23048.54794520548
# leaf nodes 200 : MAE 23078.01506849315
# leaf nodes 500 : MAE 23529.8602739726
# leaf nodes 1000 : MAE 23645.746575342466

#predict test
dt_model = DecisionTreeRegressor(max_leaf_nodes=100,criterion='mae',random_state=1)
dt_model.fit(data_X, data_y)
preds = dt_model.predict(test)

output = pd.DataFrame({"Id":test.Id, "SalePrice":preds})
output.to_csv("dt_with_categorical_submission.csv", index=False)

#random forest model validation
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
preds = dt_model.predict(val_X)
mae = mean_absolute_error(preds, val_y)
print("MAE: "+str(mae))
# MAE: 12704.689041095891

#predict
rf_model.fit(data_X,data_y)
preds = rf_model.predict(test)
rf_output = pd.DataFrame({"Id":test.Id, "SalePrice":preds})
rf_output.to_csv("rf_with_all_categorical_submission.csv", index=False)
