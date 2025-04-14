# %% [markdown]
# # Building PAM-Ai ML Model

# %%
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
# import psycopg2 as sqlC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
import joblib 

import mysql.connector
from mysql.connector import Error
from datetime import datetime
import pandas as pd

# %% [markdown]
# # 1.0 New Technique involving connnecting to PAM-Ai Database for predictions
# ## 1.1 Connecting to Database

# %%
db_host = "pam-ai-db.czgs8si2qttr.us-east-1.rds.amazonaws.com"
db_user = 'root'
db_password = "PG0XQ`,#?C*\\42_"
database = "pamai_db"

# Set up the MySQL connection
connection = mysql.connector.connect(
    host=db_host,
    user=db_user,
    password=db_password,
    database=database
)
db_info = connection.get_server_info()
print(f"Connected to MySQL database... MySQL Server version: {db_info}")
cursor = connection.cursor()
cursor.execute("SELECT DATABASE();")
record = cursor.fetchone()
print(f"You're connected to database: {record}")

# Create a cursor object
cursor = connection.cursor()

# Execute the query to show all tables
cursor.execute("SHOW TABLES")

# Fetch all the tables
tables = cursor.fetchall()

# Convert the result to a DataFrame to display nicely in Jupyter
df = pd.DataFrame(tables, columns=["Tables"])

# Display the DataFrame
df

# # Close the connection
# cursor.close()


# %% [markdown]
# **Activity 1.0.0** Wranglinng Function

# %%
# Wrangle Function to execute the query for each table
def wrangle(query,energy,battery):
    cursor.execute(query)

    # Fetch all the rows
    rows = cursor.fetchall()
    
    # Get the column names
    column_names = [i[0] for i in cursor.description]
    
    # Convert the result to a DataFrame to display nicely in Jupyter
    df = pd.DataFrame(rows, columns=column_names)

    # Display a few rows with full precision
    pd.set_option('display.float_format', lambda x: f'{x:.8f}')

    #Drop irrelevant columns
    # df.drop(columns=["id","alias","time","watt_conversion"], inplace=True) 
    if (energy):
        df.drop(columns=["watt_hours","id","alias","time","watt_conversion"], inplace=True)
    elif (battery):
        df.drop(columns=["battery_power","id","alias","time","watt_conversion"], inplace=True)
    else:
        df.drop(columns="id", inplace=True)
    
    # Convert each column to floating value
    df = df.astype(float)
    
    return df


# %% [markdown]
# **Activity 1.1.0** Importing Energy Consumption from PAM-AI Database

# %%
df = wrangle("SELECT * FROM EnergyUsage",True,False)
df.head()

# %%
df.info()

# %%
df["amount"].plot()

# %%
df["amount"].plot(kind="box",vert=False)

# %% [markdown]
# **Activity 1.1.1** Train-Test and Split

# %%
from sklearn.preprocessing import StandardScaler

# %%
target="amount"
features=["rate","watt","max","min"]

cutoff = int(len(df["amount"]) * 0.8)

X_train, y_train = df[features].iloc[:cutoff], df[target].iloc[:cutoff]

X_test, y_test = df[features].iloc[cutoff:], df[target].iloc[cutoff:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# %%
X_train_scaled

# %%
X_train

# %% [markdown]
# **Activity 1.1.2** Baseline Model (Ho)

# %%
y_mean = y_train.mean()
y_baseline_pred = [y_mean] * len(y_train)
mae_baseline = mean_absolute_error(y_train,y_baseline_pred)

print("Baseline mean: ",y_mean.round(2))
print("Baseline MAE: ",mae_baseline.round(2))

# %% [markdown]
# **Activity 1.1.3a** Linear Regression Model

# %%
linear_model1 = LinearRegression().fit(X_train_scaled,y_train)

# logistic_model1 = LogisticRegression().fit(X_train_scaled,y_train)

# naive_bayes_model1 = GaussianNB().fit(X_train_scaled,y_train)

# svm_model1 = SVC(kernel='rbf').fit(X_train_scaled,y_train)
svm_model1 = SVR(kernel='rbf').fit(X_train_scaled,y_train)

# %% [markdown]
# **Activity 1.1.3b** Evaluation of Linear Regression Model with training data 

# %%
y_train_pred_linear_model1 = linear_model1.predict(X_train_scaled)
mae_y_train_linear_model1 = mean_absolute_error(y_train,y_train_pred_linear_model1)

# y_train_pred_logistic_model1 = logistic_model1.predict(X_train)
# mae_y_train_logistic_model1 = mean_absolute_error(y_train,y_train_pred_logistic_model1)

# y_train_pred_naive_bayes_model1 = naive_bayes_model1.predict(X_train)
# mae_y_train_naive_bayes_model1 = mean_absolute_error(y_train,y_train_pred_naive_bayes_model1)

y_train_pred_svm_model1 = svm_model1.predict(X_train_scaled)
mae_y_train_svm_model1 = mean_absolute_error(y_train,y_train_pred_svm_model1)

print("Training mean Linear Model: ",y_train_pred_linear_model1.mean().round(2))
print("Training MAE Linear Model: ",mae_y_train_linear_model1.round(2))

# print("Training mean logistic Model: ",y_train_pred_logistic_model1.mean().round(2))
# print("Training MAE logistic Model: ",mae_y_train_logistic_model1.round(2))

# print("Training mean naive_bayes Model: ",y_train_pred_naive_bayes_model1.mean().round(2))
# print("Training MAE naive_bayes Model: ",mae_y_train_naive_bayes_model1.round(2))

print("Training mean SVM Model: ",y_train_pred_svm_model1.mean().round(2))
print("Training MAE SVM Model: ",mae_y_train_svm_model1.round(2))

# %%
df1_pred_train = pd.DataFrame(
    {
        "y_test": y_train.iloc[:399],
        "y_pred_wfv": y_train_pred_linear_model1[:399]
    },
    index = y_train.index[:399]
)

fig = px.line(df1_pred_train, labels={"value":"Amount [NGN]"})
fig.show()

# %%
df1_pred_train = pd.DataFrame(
    {
        "y_test": y_train.iloc[:399],
        "y_pred_wfv": y_train_pred_svm_model1[:399]
    },
    index = y_train.index[:399]
)

fig = px.line(df1_pred_train, labels={"value":"Amount [NGN]"})
fig.show()

# %% [markdown]
# **Activity 1.1.4c** Evaluation of Linear Regression Model with test data 

# %%
X_test_pred_linear_model1 = linear_model1.predict(X_test_scaled)
mae_X_test_linear_model1 = mean_absolute_error(y_test,X_test_pred_linear_model1)

# X_test_pred_logistic_model1 = logistic_model1.predict(X_test)
# mae_X_test_logistic_model1 = mean_absolute_error(y_test,X_test_pred_logistic_model1)

# X_test_pred_naive_bayes_model1 = naive_bayes_model1.predict(X_test)
# mae_X_test_naive_bayes_model1 = mean_absolute_error(y_test,X_test_pred_naive_bayes_model1)

X_test_pred_svm_model1 = svm_model1.predict(X_test_scaled)
mae_X_test_svm_model1 = mean_absolute_error(y_test,X_test_pred_svm_model1)

print("Test mean Linear Model: ",X_test_pred_linear_model1.mean().round(2))
print("Test MAE Linear Model: ",mae_X_test_linear_model1.round(2))

# print("Test mean logistic Model: ",X_test_pred_logistic_model1.mean().round(2))
# print("Test MAE logistic Model: ",mae_X_test_logistic_model1.round(2))

# print("Test mean naive_bayes Model: ",X_test_pred_naive_bayes_model1.mean().round(2))
# print("Test MAE naive_bayes Model: ",mae_X_test_naive_bayes_model1.round(2))

print("Test mean SVM Model: ",X_test_pred_svm_model1.mean().round(2))
print("Test MAE SVM Model: ",mae_X_test_svm_model1.round(2))

# %%
df1_pred_train = pd.DataFrame(
    {
        "y_test": y_test.iloc[:399],
        "y_pred_wfv": X_test_pred_linear_model1[:399]
    },
    index = y_test.index[:399]
)

fig = px.line(df1_pred_train, labels={"value":"Amount [NGN]"})
fig.show()

# %%
df1_pred_train = pd.DataFrame(
    {
        "y_test": y_test.iloc[:399],
        "y_pred_wfv": X_test_pred_svm_model1[:399]
    },
    index = y_test.index[:399]
)

fig = px.line(df1_pred_train, labels={"value":"Amount [NGN]"})
fig.show()

# %%
# Mean Squared Error
mse = mean_squared_error(y_test, X_test_pred_linear_model1)
print(f'MSE Linear Model: {mse}')
# Root Mean Squared Error
rmse = mse ** 0.5
print(f'RMSE: {rmse}')

# mse = mean_squared_error(y_test, X_test_pred_logistic_model1)
# print(f'MSE logistic Model: {mse}')
# # Root Mean Squared Error
# rmse = mse ** 0.5
# print(f'RMSE: {rmse}')

# mse = mean_squared_error(y_test, X_test_pred_naive_bayes_model1)
# print(f'MSE naive_bayes Model: {mse}')
# # Root Mean Squared Error
# rmse = mse ** 0.5
# print(f'RMSE: {rmse}')

mse = mean_squared_error(y_test, X_test_pred_svm_model1)
print(f'MSE Linear Model: {mse}')
# Root Mean Squared Error
rmse = mse ** 0.5
print(f'RMSE: {rmse}')

# # Root Mean Squared Error
# rmse = mse ** 0.5
# print(f'RMSE: {rmse}')

# # Mean Absolute Error
# mae = mean_squared_error(y_test, X_test_pred_linear_model1)
# print(f'MAE: {mae}')

# R-squared
r2 = r2_score(y_test, X_test_pred_linear_model1)
print(f'R²: {r2}')

# # R-squared
# r2 = r2_score(y_test, X_test_pred_logistic_model1)
# print(f'R²: {r2}')
# # R-squared
# r2 = r2_score(y_test, X_test_pred_naive_bayes_model1)
# print(f'R²: {r2}')

# R-squared
r2 = r2_score(y_test, X_test_pred_svm_model1)
print(f'R²: {r2}')

# %%
linear_model1.predict([[1000,0.045,0.9,0.00000128]])[0]

# %%
linear_model1.predict([[3000,0.86,0.9,0.00000128]])[0]

# %% [markdown]
# **Activity 1.2.0** Importing Fuel Consumption from PAM-AI Database

# %%
df = wrangle("SELECT * FROM FuelConsumption",False,True)
df.head()

# %%
df.info()

# %%
df["amount"].plot()

# %%
df["amount"].plot(kind="box",vert=False)

# %% [markdown]
# **Activity 1.2.1** Train-Test and Split

# %%
target="amount"
features=["rate","watt","max","min"]

cutoff = int(len(df["amount"]) * 0.8)

X_train, y_train = df[features].iloc[:cutoff], df[target].iloc[:cutoff]

X_test, y_test = df[features].iloc[cutoff:], df[target].iloc[cutoff:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# %% [markdown]
# **Activity 1.2.2** Baseline Model (Ho)

# %%
y_mean = y_train.mean()
y_baseline_pred = [y_mean] * len(y_train)
mae_baseline = mean_absolute_error(y_train,y_baseline_pred)

print("Baseline mean: ",y_mean.round(2))
print("Baseline MAE: ",mae_baseline.round(2))

# %% [markdown]
# **Activity 1.2.3a** Linear Regression Model

# %%
linear_model2 = LinearRegression().fit(X_train_scaled,y_train)

# logistic_model2 = LogisticRegression().fit(X_train,y_train)

# naive_bayes_model2 = GaussianNB().fit(X_train,y_train)

# svm_model2 = SVC(kernel='linear', C=0.05, gamma='scale').fit(X_train,y_train)

svm_model2 = SVR(kernel='rbf').fit(X_train_scaled,y_train)

# %% [markdown]
# **Activity 1.2.3b** Evaluation of Linear Regression Model with training data 

# %%
y_train_pred_linear_model2 = linear_model2.predict(X_train_scaled)
mae_y_train_linear_model2 = mean_absolute_error(y_train,y_train_pred_linear_model2)

# y_train_pred_logistic_model2 = logistic_model2.predict(X_train)
# mae_y_train_logistic_model2 = mean_absolute_error(y_train,y_train_pred_logistic_model2)

# y_train_pred_naive_bayes_model2 = naive_bayes_model2.predict(X_train)
# mae_y_train_naive_bayes_model2 = mean_absolute_error(y_train,y_train_pred_naive_bayes_model2)

y_train_pred_svm_model2 = svm_model2.predict(X_train_scaled)
mae_y_train_svm_model2 = mean_absolute_error(y_train,y_train_pred_svm_model2)

print("Training mean Linear Model: ",y_train_pred_linear_model2.mean().round(2))
print("Training MAE Linear Model: ",mae_y_train_linear_model2.round(2))

# print("Training mean logistic Model: ",y_train_pred_logistic_model2.mean().round(2))
# print("Training MAE logistic Model: ",mae_y_train_logistic_model2.round(2))

# print("Training mean naive_bayes Model: ",y_train_pred_naive_bayes_model2.mean().round(2))
# print("Training MAE naive_bayes Model: ",mae_y_train_naive_bayes_model2.round(2))

print("Training mean Linear Model: ",y_train_pred_linear_model2.mean().round(2))
print("Training MAE Linear Model: ",mae_y_train_linear_model2.round(2))

# %%
print("Training mean Linear Model: ",y_train_pred_svm_model2.mean().round(2))
print("Training MAE Linear Model: ",mae_y_train_svm_model2.round(2))

# %%
df2_pred_train = pd.DataFrame(
    {
        "y_test": y_train.iloc[:399],
        "y_pred_wfv": y_train_pred_linear_model2[:399]
    },
    index = y_train.index[:399]
)

fig = px.line(df2_pred_train, labels={"value":"Amount [NGN]"})
fig.show()

# %%
df2_pred_train = pd.DataFrame(
    {
        "y_test": y_train.iloc[:399],
        "y_pred_wfv": y_train_pred_svm_model2[:399]
    },
    index = y_train.index[:399]
)

fig = px.line(df2_pred_train, labels={"value":"Amount [NGN]"})
fig.show()

# %% [markdown]
# **Activity 1.1.4c** Evaluation of Linear Regression Model with test data 

# %%
X_test_pred_linear_model2 = linear_model2.predict(X_test_scaled)
mae_X_test_linear_model2 = mean_absolute_error(y_test,X_test_pred_linear_model2)

# X_test_pred_logistic_model2 = logistic_model2.predict(X_test_scaled)
# mae_X_test_logistic_model2 = mean_absolute_error(y_test,X_test_pred_logistic_model2)

# X_test_pred_naive_bayes_model2 = naive_bayes_model2.predict(X_test_scaled)
# mae_X_test_naive_bayes_model2 = mean_absolute_error(y_test,X_test_pred_naive_bayes_model2)

X_test_pred_svm_model2 = svm_model2.predict(X_test_scaled)
mae_X_test_svm_model2 = mean_absolute_error(y_test,X_test_pred_svm_model2)

print("Test mean Linear Model: ",X_test_pred_linear_model2.mean().round(2))
print("Test MAE Linear Model: ",mae_X_test_linear_model2.round(2))

# print("Test mean logistic Model: ",X_test_pred_logistic_model2.mean().round(2))
# print("Test MAE logistic Model: ",mae_X_test_logistic_model2.round(2))

# print("Test mean naive_bayes Model: ",X_test_pred_naive_bayes_model2.mean().round(2))
# print("Test MAE naive_bayes Model: ",mae_X_test_naive_bayes_model2.round(2))

print("Test mean Linear Model: ",X_test_pred_svm_model2.mean().round(2))
print("Test MAE Linear Model: ",mae_X_test_svm_model2.round(2))

# %%
df2_pred_test = pd.DataFrame(
    {
        "y_test": y_train.iloc[:399],
        "y_pred_wfv": X_test_pred_linear_model2[:399]
    },
    index = y_train.index[:399]
)

fig = px.line(df2_pred_test, labels={"value":"Amount [NGN]"})
fig.show()

# %%
df2_pred_test = pd.DataFrame(
    {
        "y_test": y_train.iloc[:399],
        "y_pred_wfv": X_test_pred_svm_model2[:399]
    },
    index = y_train.index[:399]
)

fig = px.line(df2_pred_test, labels={"value":"Amount [NGN]"})
fig.show()

# %%
# Mean Squared Error
mse = mean_squared_error(y_test, X_test_pred_linear_model2)
print(f'MSE Linear Model: {mse}')
# Root Mean Squared Error
rmse = mse ** 0.5
print(f'RMSE: {rmse}')

# mse = mean_squared_error(y_test, X_test_pred_logistic_model2)
# print(f'MSE logistic Model: {mse}')
# # Root Mean Squared Error
# rmse = mse ** 0.5
# print(f'RMSE: {rmse}')

# mse = mean_squared_error(y_test, X_test_pred_naive_bayes_model2)
# print(f'MSE naive_bayes Model: {mse}')
# # Root Mean Squared Error
# rmse = mse ** 0.5
# print(f'RMSE: {rmse}')

mse = mean_squared_error(y_test, X_test_pred_svm_model2)
print(f'MSE Linear Model: {mse}')
# Root Mean Squared Error
rmse = mse ** 0.5
print(f'RMSE: {rmse}')

# Root Mean Squared Error
rmse = mse ** 0.5
print(f'RMSE: {rmse}')

# # Mean Absolute Error
# mae = mean_squared_error(y_test, X_test_pred_linear_model1)
# print(f'MAE: {mae}')

# R-squared
r2 = r2_score(y_test, X_test_pred_linear_model2)
print(f'R²: {r2}')
# # R-squared
# r2 = r2_score(y_test, X_test_pred_logistic_model2)
# print(f'R²: {r2}')
# # R-squared
# r2 = r2_score(y_test, X_test_pred_naive_bayes_model2)
# print(f'R²: {r2}')
# R-squared
r2 = r2_score(y_test, X_test_pred_svm_model2)
print(f'R²: {r2}')

# %%
linear_model2.predict([[1000,0.045,0.9,0.00000128]])[0]

# %%
linear_model2.predict([[3000,0.86,0.9,0.00000128]])[0]

# %% [markdown]
# **Activity 1.3.0** Importing Solar Production from PAM-AI Database

# %%
df = wrangle("SELECT * FROM SolarProduction",False,False)
df.head()

# %%
df.info()

# %%
df["amount"].plot()

# %%
df["amount"].plot(kind="box",vert=False)

# %% [markdown]
# **Activity 1.3.1** Train-Test and Split

# %%
target="amount"
features=["rate","watt","max","min"]

cutoff = int(len(df["amount"]) * 0.8)

X_train, y_train = df[features].iloc[:cutoff], df[target].iloc[:cutoff]

X_test, y_test = df[features].iloc[cutoff:], df[target].iloc[cutoff:]


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# %% [markdown]
# **Activity 1.3.2** Baseline Model (Ho)

# %%
y_mean = y_train.mean()
y_baseline_pred = [y_mean] * len(y_train)
mae_baseline = mean_absolute_error(y_train,y_baseline_pred)

print("Baseline mean: ",y_mean.round(2))
print("Baseline MAE: ",mae_baseline.round(2))

# %% [markdown]
# **Activity 1.3.3a** Linear Regression Model

# %%
linear_model3 = LinearRegression().fit(X_train_scaled,y_train)

# logistic_model2 = LogisticRegression().fit(X_train,y_train)

# naive_bayes_model2 = GaussianNB().fit(X_train,y_train)

# svm_model2 = SVC(kernel='linear', C=0.05, gamma='scale').fit(X_train,y_train)

svm_model3 = SVR(kernel='rbf').fit(X_train_scaled,y_train)

# %% [markdown]
# **Activity 1.3.3b** Evaluation of Linear Regression Model with training data

# %%
y_train_pred_linear_model2 = linear_model2.predict(X_train_scaled)
mae_y_train_linear_model2 = mean_absolute_error(y_train,y_train_pred_linear_model2)

# y_train_pred_logistic_model2 = logistic_model2.predict(X_train)
# mae_y_train_logistic_model2 = mean_absolute_error(y_train,y_train_pred_logistic_model2)

# y_train_pred_naive_bayes_model2 = naive_bayes_model2.predict(X_train)
# mae_y_train_naive_bayes_model2 = mean_absolute_error(y_train,y_train_pred_naive_bayes_model2)

y_train_pred_svm_model2 = svm_model2.predict(X_train_scaled)
mae_y_train_svm_model2 = mean_absolute_error(y_train,y_train_pred_svm_model2)

print("Training mean Linear Model: ",y_train_pred_linear_model2.mean().round(2))
print("Training MAE Linear Model: ",mae_y_train_linear_model2.round(2))

# print("Training mean logistic Model: ",y_train_pred_logistic_model2.mean().round(2))
# print("Training MAE logistic Model: ",mae_y_train_logistic_model2.round(2))

# print("Training mean naive_bayes Model: ",y_train_pred_naive_bayes_model2.mean().round(2))
# print("Training MAE naive_bayes Model: ",mae_y_train_naive_bayes_model2.round(2))

print("Training mean Linear Model: ",y_train_pred_linear_model2.mean().round(2))
print("Training MAE Linear Model: ",mae_y_train_linear_model2.round(2))

# %%
df3_pred_train = pd.DataFrame(
    {
        "y_test": y_train.iloc[:399],
        "y_pred_wfv": y_train_pred_linear_model2[:399]
    },
    index = y_train.index[:399]
)

fig = px.line(df3_pred_train, labels={"value":"Amount [NGN]"})
fig.show()

# %%
df3_pred_train = pd.DataFrame(
    {
        "y_test": y_train.iloc[:399],
        "y_pred_wfv": y_train_pred_svm_model2[:399]
    },
    index = y_train.index[:399]
)

fig = px.line(df3_pred_train, labels={"value":"Amount [NGN]"})
fig.show()

# %% [markdown]
# **Activity 1.3.4c** Evaluation of Linear Regression Model with test data

# %%
X_test_pred_linear_model3 = linear_model3.predict(X_test_scaled)
mae_X_test_linear_model3 = mean_absolute_error(y_test,X_test_pred_linear_model3)

# X_test_pred_logistic_model3 = logistic_model3.predict(X_test)
# mae_X_test_logistic_model3 = mean_absolute_error(y_test,X_test_pred_logistic_model3)

# X_test_pred_naive_bayes_model3 = naive_bayes_model3.predict(X_test)
# mae_X_test_naive_bayes_model3 = mean_absolute_error(y_test,X_test_pred_naive_bayes_model3)

X_test_pred_svm_model3 = svm_model3.predict(X_test)
mae_X_test_svm_model3 = mean_absolute_error(y_test,X_test_pred_svm_model3)

print("Test mean Linear Model: ",X_test_pred_linear_model3.mean().round(2))
print("Test MAE Linear Model: ",mae_X_test_linear_model3.round(2))

# print("Test mean logistic Model: ",X_test_pred_logistic_model3.mean().round(2))
# print("Test MAE logistic Model: ",mae_X_test_logistic_model3.round(2))

# print("Test mean naive_bayes Model: ",X_test_pred_naive_bayes_model3.mean().round(2))
# print("Test MAE naive_bayes Model: ",mae_X_test_naive_bayes_model3.round(2))

print("Test mean Linear Model: ",X_test_pred_svm_model3.mean().round(2))
print("Test MAE Linear Model: ",mae_X_test_svm_model3.round(2))

# %%
df3_pred_test = pd.DataFrame(
    {
        "y_test": y_train.iloc[:399],
        "y_pred_wfv": X_test_pred_linear_model3[:399]
    },
    index = y_train.index[:399]
)

fig = px.line(df3_pred_test, labels={"value":"Amount [NGN]"})
fig.show()

# %%
df3_pred_test = pd.DataFrame(
    {
        "y_test": y_train.iloc[:399],
        "y_pred_wfv": X_test_pred_svm_model3[:399]
    },
    index = y_train.index[:399]
)

fig = px.line(df3_pred_test, labels={"value":"Amount [NGN]"})
fig.show()

# %%
# Mean Squared Error
mse = mean_squared_error(y_test, X_test_pred_linear_model3)
print(f'MSE Linear Model: {mse}')
# Root Mean Squared Error
rmse = mse ** 0.5
print(f'RMSE: {rmse}')

# mse = mean_squared_error(y_test, X_test_pred_logistic_model3)
# print(f'MSE logistic Model: {mse}')
# # Root Mean Squared Error
# rmse = mse ** 0.5
# print(f'RMSE: {rmse}')

# mse = mean_squared_error(y_test, X_test_pred_naive_bayes_model3)
# print(f'MSE naive_bayes Model: {mse}')
# # Root Mean Squared Error
# rmse = mse ** 0.5
# print(f'RMSE: {rmse}')

mse = mean_squared_error(y_test, X_test_pred_svm_model3)
print(f'MSE Linear Model: {mse}')
# Root Mean Squared Error
rmse = mse ** 0.5
print(f'RMSE: {rmse}')

# # Root Mean Squared Error
# rmse = mse ** 0.5
# print(f'RMSE: {rmse}')

# # Mean Absolute Error
# mae = mean_squared_error(y_test, X_test_pred_linear_model1)
# print(f'MAE: {mae}')

# R-squared
r2 = r2_score(y_test, X_test_pred_linear_model3)
print(f'R²: {r2}')
# R-squared
# r2 = r2_score(y_test, X_test_pred_logistic_model3)
# print(f'R²: {r2}')
# # R-squared
# r2 = r2_score(y_test, X_test_pred_naive_bayes_model3)
# print(f'R²: {r2}')
# R-squared
# r2 = r2_score(y_test, X_test_pred_svm_model1)
# print(f'R²: {r2}')


joblib.dump(svm_model1, 'energy_SVM_model.pkl')
joblib.dump(svm_model2, 'fuel_SVM_model.pkl')
joblib.dump(svm_model3, 'solar_SVM_model.pkl')

