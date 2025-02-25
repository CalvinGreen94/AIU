from flask import Flask, render_template, request
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
import os
import numpy as np

app = Flask(__name__)
def get_next_filename(base_filename='data', extension='json'):
    i = 0
    while True:
        filename = f"{base_filename}_{i}.{extension}"
        if not os.path.exists(filename):
            return filename
        i += 1
def save_data_to_json(new_data, base_filename='metadata\data', extension='json'):
    filename = get_next_filename(base_filename, extension)
    
    # Save the new data to a new file
    with open(filename, 'w') as json_file:
        json.dump(new_data, json_file, indent=4)
    
    print(f"Data saved to {filename}")


def predict_next_30_days(model, last_known_data, scaler):
    future_predictions = []
    current_input = last_known_data.reshape(1, -1)
    
    for _ in range(30):
        predicted_value = model.predict(current_input)[0]
        future_predictions.append(predicted_value)
        
        # Shift the input data and add the new prediction
        current_input = np.roll(current_input, -1)
        current_input[0, -1] = predicted_value  # Replace the latest value with prediction
    
    # Inverse transform predictions to original scale
    future_predictions = scaler.inverse_transform(
        np.hstack([np.zeros((30, current_input.shape[1] - 1)), np.array(future_predictions).reshape(-1, 1)])
    )[:, -1]
    
    return future_predictions

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the stock ticker from the form
        stock_ticker = request.form['ticker']
        shares = request.form['shares']
        balance = request.form['balance']
        salary = request.form['salary']
        
        try:
            # Fetch data from yfinance
            dat = yf.Ticker(stock_ticker)
            # dat.

            info = dat.info
            calendar = dat.calendar
            targets = dat.analyst_price_targets
            income_quarter = dat.quarterly_income_stmt
            history = dat.history(period='1d')
            options_chain = dat.option_chain(dat.options[0]).calls if dat.options else None
            info = dat.get_info()
            insider = dat.get_insider_purchases()
            insider_trans = dat.get_insider_transactions()
            sustainability = dat.get_sustainability()
            recommendation = dat.get_recommendations_summary()
            holders = dat.get_major_holders()
            

            data = yf.download(stock_ticker, start="2010-01-03", end="2025-01-14")
            data = pd.DataFrame(data)
            # data = data.drop(['Adj Close'],axis=1)
            print(data.tail())
            data = data.to_csv(f'data/stocks/stocks_portfolio/{stock_ticker}.csv',index=False,sep=',')
            data = pd.read_csv(f'data/stocks/stocks_portfolio/{stock_ticker}.csv',skiprows=[1])
            print('RETRIEVING STOCK DATA FOR {}'.format(str('stock_ticker"')))
            
            # data = data.drop(['Date'],axis = 1) 
            data['MA_5'] = data['Close'].rolling(window=5).mean()
            data['MA_30'] = data['Close'].rolling(window=30).mean()

            # Drop rows with missing values
            data.dropna(inplace=True)

            # Define target and features
            y = data[['High']]  # Target: High price
            X = data.drop(['High'], axis=1)  # Features: All other columns

            # Normalize features (do not scale the target)
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)

            # Split the data into training and testing sets using TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=4)

            # Initialize model (consider switching to RandomForest or another model)
            model ="USE LINEAR REGRESSION OR RANDOM FOREST REGRESSION)
            # import bot
            # bot()
            # bot.deeper()

            # Train the model and evaluate performance
            for train_index, test_index in tscv.split(X_scaled):
                X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
                # Train the model
                model.fit(X_train, y_train)
                joblib.dump(model, f'models/{stock_ticker}.joblib')
                model = joblib.load(f'models/{stock_ticker}.joblib')
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Evaluate the model
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                print(f'Mean Absolute Error: {mae}')
                print(f'R2 Score: {r2}')

            # Predict the price for the latest data
            latest_data = X_scaled[-1:].reshape(1, -1)  # Get the latest sample
            price_pred = model.predict(latest_data)
            print(f'Predicted High Price: {price_pred[0]}')
            pred = float(price_pred)
            # hist = float(y[-1:])
            accuracy = model.score(X_test,y_test)
            future_prices = predict_next_30_days(model, X_scaled[-1:], scaler)
            


        

            # graph_file = os.path.join(app.root_path, 'static', 'graph.html')
            # if os.path.exists(graph_file):
            #     with open(graph_file, 'r') as file:
            #         graph_html = file.read()
            # else:
            #     graph_html = "Graph not found."
            # context = request.form.get("Enter your question, Then Click the Button Below")
            # res = client.query(context)
            # answers = next(res.results).text
            # answers = str(answers)
            news = news = yf.Search(f"{stock_ticker}", news_count=1).news
        
            metadata = {
                        "name": f"{stock_ticker}_AI",
                        "symbol": f"{stock_ticker}",
                        # uri: imageUri,
                        "description": response,
                    }
            if accuracy >=0.87:
                save_data_to_json(metadata)
            
            # Pass the data to the result template
            return render_template(
                'result.html',
                # graph_html=graph_html,
                future=future_prices,
                # response = response,
                accuracy= r2,
                mae=mae,
                price_pred = price_pred,
                insider = insider,
                sustainability = sustainability,
                insider_trans = insider_trans,
                ticker=stock_ticker.upper(),
                info=info,
                calendar=calendar,
                targets=targets,
                income_quarter=income_quarter.to_html(classes='table table-striped'),
                history=history.to_html(classes='table table-striped'),
                options_chain=options_chain.to_html(classes='table table-striped') if options_chain is not None else "No options available"
            )
        except Exception as e:
            # Handle any errors
            return render_template('index.html', error=str(e))

    return render_template('index.html')
import time
if __name__ == "__main__":
    
    app.run(debug=True,host="0.0.0.0",port=3314)
    # time.sleep(10)
    # bot()


# import numpy as np 
# from sklearn.linear_model import LinearRegression 
 
# # Example data 
# dates = np.array([1, 2, 3, 4, 5]).reshape(-1, 1) 
# prices = np.array([100, 105, 110, 115, 120]) 
 
# # Create and train model 
# model = LinearRegression() 
# model.fit(dates, prices) 
 
# # Predict future price 
# future_date = np.array([6]).reshape(-1, 1) 
# predicted_price = model.predict(future_date) 
# print(predicted_price)  # Output predicted stock price for day 6
