import sklearn 
from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

  
# https://archive.ics.uci.edu/dataset/222/bank+marketing  
# fetch dataset 
bank_marketing = fetch_ucirepo(id=222) 
  
# data (as pandas dataframes) 
X = bank_marketing.data.features 
y = bank_marketing.data.targets 
  
# Drop the column duration because in the page it says it isn't necessary for a predictive model
X = X.drop(columns=['duration'])

X.job = X.job.fillna('unknown')
X.education = X.education.fillna('unknown')
X.poutcome = X.poutcome.fillna('nonexistent')
X.contact = X.contact.fillna('unknown')

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = encoder.fit_transform(X.select_dtypes(include=['object']))
X_encoded_df = pd.DataFrame(
    X_encoded.toarray(), 
    columns=encoder.get_feature_names_out(X.select_dtypes(include=['object']).columns)
)


X_numerico = X.select_dtypes(exclude=['object'])
X_final = pd.concat([X_numerico.reset_index(drop=True), X_encoded_df.reset_index(drop=True)], axis=1)

## AQUI EMPIEZA EL CLASSIFIER

cat_col = X.select_dtypes(include='object').columns.tolist()
num_col = X.select_dtypes(exclude='object').columns.tolist()


categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, cat_col),
        ('num', numerical_transformer, num_col)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Separating training and testing data
y = y['y'].map({'yes': 1, 'no': 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Hyperparameter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV

param = {
    'classifier__C': [0.01, 0.1, 1, 10],
    'classifier__penalty': ['l2'],
    'classifier__solver': ['lbfgs']
}

grid_search = GridSearchCV(pipeline, param, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

best_logisticreg_model = grid_search.best_estimator_

# Evaluating the model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred = best_logisticreg_model.predict(X_test)
y_proba = best_logisticreg_model.predict_proba(X_test)[:, 1]

print("Mejores parámetros:", grid_search.best_params_)
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
