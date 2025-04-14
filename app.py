"""
 @author: Mbaka JohnPaul

 """

from http.client import HTTPException
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from dict import PAM
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import mysql.connector
from mysql.connector import Error
from sklearn.metrics import mean_absolute_error
import joblib 


app = FastAPI()

class EnergyRateRequest(BaseModel):
    energyRate: float


# Load Database
db_host = "pam-ai-db.czgs8si2qttr.us-east-1.rds.amazonaws.com"
db_user = 'root'
db_password = "PG0XQ`,#?C*\\42_"
database = "pamai_db"

@app.get('/')
def index():
    return {'message': 'Hello, welcome to PAM-ML'}

@app.post('/seasonal-ml')
def predict_amount_seasonal(data:PAM):
    try:
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
        # df
        
        # Wrangle Function to execute the query for each table
        def wrangle_base(query,energy,battery):
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
                df.drop(columns=["watt_hours","id","alias","time","watt_conversion"], inplace=True) #"ProfileId"
            elif (battery):
                df.drop(columns=["battery_power","id","alias","time","watt_conversion","ProfileId"], inplace=True)
            else:
                df.drop(columns=["id","ProfileId","UpdatedAt"], inplace=True)
            print(df.head())
            
            # Convert each column to floating value
            df = df.astype(float)
            
            return df

        # load all Models
        energy_model = joblib.load('energy_SVM_model.pkl')
        
        # load database
        df1e = wrangle_base("SELECT * FROM EnergyUsage",True,False)

        data = data.dict()
        rate = data['rate']
        kw = data['kw']
        max = np.round(df1e["watt"],2).max()
        min = np.round(df1e["watt"],2)[np.round(df1e["watt"],2) != 0].min()
        print("I am max:",max)
        print("I am min:",min)
        
        prediction = energy_model.predict([[rate,kw,max,min]])
        
        return{
            'amount': str(prediction[0])
        }
    except database.DatabaseError as db_err:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {db_err}")
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=f"Type conversion error: {val_err}")
    
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
    
@app.post('/fuel-ml')
def predict_amount_fuel(data:PAM):
    try:
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
        # df
        
        # Wrangle Function to execute the query for each table
        def wrangle_base(query,energy,battery):
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
                df.drop(columns=["watt_hours","id","alias","time","watt_conversion"], inplace=True) #"ProfileId"
            elif (battery):
                df.drop(columns=["battery_power","id","alias","time","watt_conversion","ProfileId"], inplace=True)
            else:
                df.drop(columns=["id","ProfileId","UpdatedAt"], inplace=True)
            print(df.head())
            
            # Convert each column to floating value
            df = df.astype(float)
            
            return df

        # load all Models
        fuel_model = joblib.load('fuel_SVM_model.pkl')  
        
        # load database
        df2e = wrangle_base("SELECT * FROM FuelConsumption",False,True) 

        data = data.dict()
        rate = data['rate']
        kw = data['kw']
        max = np.round(df2e["watt"],2).max()
        min = np.round(df2e["watt"],2)[np.round(df2e["watt"],2) != 0].min()
        print("I am max:",max)
        print("I am min:",min)
        
        prediction = fuel_model.predict([[rate,kw,max,min]])
        
        return{
            'amount': str(prediction[0])
        }
    except database.DatabaseError as db_err:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {db_err}")
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=f"Type conversion error: {val_err}")
    
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
    
    
@app.post('/solar-ml')
def predict_amount_solar(data:PAM):
  try:
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
    # df
    
    # Wrangle Function to execute the query for each table
    def wrangle_base(query,energy,battery):
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
                df.drop(columns=["watt_hours","id","alias","time","watt_conversion"], inplace=True) #"ProfileId"
            elif (battery):
                df.drop(columns=["battery_power","id","alias","time","watt_conversion","ProfileId"], inplace=True)
            else:
                df.drop(columns=["id","ProfileId","UpdatedAt"], inplace=True)
            print(df.head())
            
            # Convert each column to floating value
            df = df.astype(float)
            
            return df

    # load all Models
    solar_model = joblib.load('solar_SVM_model.pkl') 
        
    # load database
    df3e = wrangle_base("SELECT * FROM SolarProduction",False,False)

    data = data.dict()
    rate = data['rate']
    kw = data['kw']
    max = np.round(df3e["watt"],2).max()
    min = np.round(df3e["watt"],2)[np.round(df3e["watt"],2) != 0].min()
    print("I am max:",max)
    print("I am min:",min)
    
    prediction = solar_model.predict([[rate,kw,max,min]])
    
    return{
        'amount': str(prediction[0])
    }
  except database.DatabaseError as db_err:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {db_err}")
  except ValueError as val_err:
        raise HTTPException(status_code=400, detail=f"Type conversion error: {val_err}")
    
  except Error as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")
  finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
      
@app.post('/forecast')
async def forecast_24h(request: EnergyRateRequest):
    energyRate = request.energyRate
    
    # Run your long computations in a separate thread
    result = await asyncio.to_thread(long_running_computation, energyRate)
    return result

def long_running_computation(energyRate: float):
    try:
        # Set up the MySQL connection
        connection = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=database
        )
        
        # Place your looping logic and computation-intensive code here
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
        # df
        
        # Wrangle Function to execute the query for each table
        def wrangle_base(query,energy,battery):
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
                df.drop(columns=["watt_hours","id","alias","time","watt_conversion"], inplace=True) #"ProfileId"
            elif (battery):
                df.drop(columns=["battery_power","id","alias","time","watt_conversion","ProfileId"], inplace=True)
            else:
                df.drop(columns=["id","ProfileId","UpdatedAt"], inplace=True)
            print(df.head())
            
            # Convert each column to floating value
            df = df.astype(float)
            
            return df
        
        # load database
        df1 = wrangle_base("SELECT * FROM EnergyUsage",True,False)
        df2 = wrangle_base("SELECT * FROM FuelConsumption",False,True) 
        df3 = wrangle_base("SELECT * FROM SolarProduction",False,False)
        
        # load all Models
        energy_model = joblib.load('energy_SVM_model.pkl')
            
        def wrangle(df,energy=False,fuel=False,solar=False):
        # Reassignment of Watt variable with the dataset of the specific Watt
            if(energy):
                df["energy"] = df["watt"]
                df.drop(columns="watt",inplace=True)
            elif(fuel):
                df["fuel"] = df["watt"]
                df.drop(columns="watt",inplace=True)
            elif(solar):
                df["solar"] = df["watt"]
                df.drop(columns="watt",inplace=True)
            return df

        df1 = wrangle(df1,energy=True)
        df2 = wrangle(df2,fuel=True)
        df3 = wrangle(df3,solar=True)

        # df2.isnull.sum(), df2.isnull.sum(), df3.isnull.sum()
        df3.drop(columns=["rate","amount","min","max"],inplace=True, errors="ignore")
        df2.drop(columns=["rate","amount","min","max"],inplace=True, errors="ignore")
        df1.drop(columns=["rate","amount","min","max"],inplace=True, errors="ignore")

        #df = pd.concat([df1,df2], ignore_index=True)
        # Ensure indices are unique and sorted in ascending order
        df1 = df1[~df1.index.duplicated(keep='first')].sort_index()
        df2 = df2[~df2.index.duplicated(keep='first')].sort_index()
        df3 = df3[~df3.index.duplicated(keep='first')].sort_index()

        # Step 1
        u_index = df3.index.union(df2.index.union(df1.index))
        # print(f"{u_index} hapenning kekekjekek jekkeke")

        df3_n0 = df3.reindex(index=u_index,method='bfill').fillna(0)
        df2_n0 = df2.reindex(index=u_index,method='bfill').fillna(0)
        df1_n0 = df1.reindex(index=u_index,method='bfill').fillna(0)
        
        # df1.fillna(0, inplace=True)  # Replace NaNs with 0 (or a meaningful value)
        # df2.fillna(0, inplace=True)
        # df3.fillna(0, inplace=True)


        df3_n = pd.Series(df3_n0["solar"],dtype='float64').fillna(0)
        df2_n = pd.Series(df2_n0["fuel"],dtype='float64').fillna(0)
        df1_n = pd.Series(df1_n0["energy"],dtype='float64').fillna(0)
        
        
        
        # print("NaN values in data:", df3_n.isna().sum())
        # print("Infinite values in data:", np.isinf(df3_n).sum())
        
        
        # print("NaN values in data:", df2_n0.isna().sum())
        # print("Infinite values in data:", np.isinf(df2_n).sum())
        # print("NaN values in data:", df2_n.isna().sum())
        # print("Infinite values in data:", np.isinf(df2_n).sum())
        # print("NaN values in data:", df2_n.isna().sum())
        # print("Infinite values in data:", np.isinf(df2_n).sum())
        
        # print("NaN values in data:", df1_n0.isna().sum())
        # print("Infinite values in data:", np.isinf(df1_n).sum())

        # Energy Model
        y1_pred_wfv = pd.Series()
        history1 = df1_n.copy() #here we reload the dataset again from online DB
        # # Drop rows with NaN or Inf
        # history.replace([float('inf'), float('-inf')], float('nan')).dropna()

        for i in range(24):
            model = AutoReg(history1,lags=26).fit()
            next_pred = model.forecast()
            y1_pred_wfv = pd.concat([y1_pred_wfv, pd.DataFrame(next_pred)])
            history1 = pd.concat([history1, next_pred])
            print(f"Iteration {i} completed. DF1 {history1.last}")
    
    
        # Fuel Model
        print(df2_n.head())
        y2_pred_wfv = pd.Series()
        history2 = df2_n.copy() #here we reload the dataset again from online DB
        
        print(history2.head())
        
        # if history2.isna().sum() == 0:
        #     print("WORKING WORKING WORKING")
        # else:
        #     print("Data still contains NaNs. Check preprocessing.")
        
        for i in range(24):
            # print(f"Iteration {i} started...DF2")
            # print("NaN values in data:", history2.isna().sum())
            # print("Infinite values in data:", np.isinf(history2).sum())
            # print("NaN values in data:", df2_n.isna().sum())
            # print("Infinite values in data:", np.isinf(df2_n).sum())
            # Drop rows with NaN or Inf
            # history2.replace([np.inf, -np.inf], np.nan, inplace=True)
            # history2.dropna(inplace=True)

            # # Ensure history1 is numeric
            # history2 = history2.astype(float)

            # # Pass only clean data to AutoReg
            # if history2.isna().sum() == 0:
            #     model = AutoReg(history2, lags=26).fit()
            # else:
            #     print("Data still contains NaNs. Check preprocessing.")
            
            model = AutoReg(history2,lags=26).fit()
            next_pred = model.forecast()
            # print(next_pred)
            # print(next_pred)
            y2_pred_wfv = pd.concat([y2_pred_wfv, pd.DataFrame(next_pred)])
            history2 = pd.concat([history2, next_pred])
            print(f"Iteration {i} completed. DF2")
            print(f"Iteration {i} completed. DF1 {history2.last}")
            
        # Solar Model
        y3_pred_wfv = pd.Series()
        history3 = df3_n.copy() #here we reload the dataset again from online DB

        for i in range(24):
            # print(f"Iteration {i} started...DF3")
            # print("NaN values in data:", history3.isna().sum())
            # print("Infinite values in data:", np.isinf(history3).sum())
            # Drop rows with NaN or Inf
            # history3.replace([np.inf, -np.inf], np.nan, inplace=True)
            # history3.dropna(inplace=True)

            # # Ensure history1 is numeric
            # history3 = history3.astype(float)

            # # Pass only clean data to AutoReg
            # if history3.isna().sum() == 0:
            #     model = AutoReg(history3, lags=26).fit()
            # else:
            #     print("Data still contains NaNs. Check preprocessing.")
            
            model = AutoReg(history3,lags=26).fit()
            next_pred = model.forecast()
            # print(next_pred)
            y3_pred_wfv = pd.concat([y3_pred_wfv, pd.DataFrame(next_pred)])
            history3 = pd.concat([history3, next_pred])
            print(f"Iteration {i} completed. DF3  {history3.last}")
        
        # Amount Prediction for Energy Consumed
        y4_pred_wfv = pd.Series(dtype='float64')#here we reload the dataset again from online DB

        for i in range(len(y1_pred_wfv)): 
            # print(f"Value : {y1_pred_wfv[0].iloc[i]}")
            # print(f"Rate : {energyRate}")
            
            max = np.round(df1["energy"],2).max()
            min = np.round(df1["energy"],2)[np.round(df1["energy"],2) != 0].min()
            
            next_pred = energy_model.predict([[float(energyRate),float(y1_pred_wfv[0].iloc[i]),max,min]]) #pd.DataFrame({"amount": [next_pred[0]]}
            
            # print(f"Here is next prediction{next_pred[0]}")
            y4_pred_wfv = pd.concat([y4_pred_wfv,pd.Series([next_pred[0]], index=[i])])
            print(f"Here is next prediction{y4_pred_wfv[i]}")
            
        return {
            'energy': y1_pred_wfv,
            'fuel': y2_pred_wfv,
            'solar': y3_pred_wfv,
            'amount': y4_pred_wfv
        }
    except database.DatabaseError as db_err:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {db_err}")
    
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=f"Type conversion error: {val_err}")
    
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
    
    
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8008, log_level="info")

