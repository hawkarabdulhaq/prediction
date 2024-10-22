# Import librariess
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load the XLSX file
file_path = './data/input_data.xlsx'  # Path relative to the repository root
df = pd.read_excel(file_path)

# Check if data is loaded
print(df.head())  # Print the first few rows of data to verify

# Strip whitespace from columns only if data is present
if not df.empty:
    df.columns = df.columns.astype(str).str.strip()

# Check the data structure and column names
print(df.columns)

# Adjust for actual column names
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

    # Convert date to ordinal for regression
    df['Date_ordinal'] = df['Date'].apply(lambda date: date.toordinal())

    # Set up X (features) and y (target)
    X = df[['Date_ordinal']]
    y = df['IQD (Iraqi Dinar)']  # Use the exact column name you have

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Predict future values (next 10 days)
    last_date = df['Date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, 11)]
    future_dates_ordinal = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)
    predictions = model.predict(future_dates_ordinal)

    # Display predictions
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted_IQD': predictions})
    print(future_df)

    # Convert the 'Date' column in future_df to string format 'YYYY-MM-DD'
    future_df['Date'] = future_df['Date'].dt.strftime('%Y-%m-%d')

    # Convert Date column in original DataFrame to matplotlib-friendly format
    df['Date'] = pd.to_datetime(df['Date'])

    # Plot the actual data and predictions
    plt.figure(figsize=(10,6))
    plt.plot(df['Date'], df['IQD (Iraqi Dinar)'], label='Actual Data')  # Use correct column name
    plt.plot(pd.to_datetime(future_df['Date']), future_df['Predicted_IQD'], label='Predicted Data', linestyle='--', color='red')
    plt.title('Dollar vs Iraqi Dinar Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('IQD')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image in the artifacts directory
    output_image_path = '../artifacts/prediction_plot.png'
    plt.savefig(output_image_path)
    plt.show()

else:
    print("The 'Date' column was not found in the data.")
