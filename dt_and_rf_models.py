import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
test_id = test.Id

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
nodes = [5,10,50,100,500,1000]
for n in nodes:
    dt_model = DecisionTreeRegressor(max_leaf_nodes=n,criterion = 'mse', random_state=1)
    dt_model.fit(train_X, train_y)
    preds = dt_model.predict(val_X)
    log_rmse = mean_squared_error(np.log(preds), np.log(val_y))**0.5
    print("leaf nodes {} : log RMSE {}".format(n, log_rmse))

# max_leaf_nodes = 100 ~ minimum

#predict test
dt_model = DecisionTreeRegressor(max_leaf_nodes=100,random_state=1)
dt_model.fit(data_X, data_y)
preds = dt_model.predict(test)

output = pd.DataFrame({"Id":test_id, "SalePrice":preds})
output.to_csv("dt_with_categorical_2_submission.csv", index=False)

#random forest model validation
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(train_X, train_y)
preds = dt_model.predict(val_X)
log_rmse = mean_squared_error(np.log(preds), np.log(val_y))**0.5
print("Random Forest log rmse: "+str(log_rmse))

#predict
rf_model.fit(data_X,data_y)
preds = rf_model.predict(test)
rf_output = pd.DataFrame({"Id":test_id, "SalePrice":preds})
rf_output.to_csv("rf_with_all_categorical_2_submission.csv", index=False)
