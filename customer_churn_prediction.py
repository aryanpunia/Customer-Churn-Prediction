import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('Churn_Modelling.csv')

X = data.iloc[:,[3,6,7,8,9,10,11,12]].values
y = data.iloc[:, -1].values

# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
# imputer.fit(X)
# X = imputer.transform(X)

# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# X[:,[4,5]] = np.array(ct.fit_transform(X[:,[4,5]]))

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=1)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

# import pandas as pd
# import numpy as np
# from sklearn.metrics import confusion_matrix, accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier

# # Load the dataset

# data = pd.read_csv('Churn_Modelling.csv')

# # Define features and target variable
# X = data.iloc[:, 3:-1].values
# y = data.iloc[:, -1].values

# # Define preprocessing steps for numeric and categorical features
# numeric_features = [0, 2, 3, 4, 5, 6, 7, 9]
# categorical_features = [1, 8]

# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])

# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(drop='first'))
# ])

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# # Append classifier to preprocessing pipeline
# clf = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0))
# ])

# # Fit the model
# clf.fit(X_train, y_train)

# # Predict on test set
# y_pred = clf.predict(X_test)

# # Evaluate the model
# cm = confusion_matrix(y_test, y_pred)
# accuracy = accuracy_score(y_test, y_pred)

# cm, accuracy

