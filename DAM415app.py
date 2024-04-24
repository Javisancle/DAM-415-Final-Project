#!/usr/bin/env python
# coding: utf-8

# ## Project Name
# #### Javier Sanchez Clemente
# #### Walsh University DAM 415 
# #### Professor Dr. Sabasi

# In[114]:


import psycopg2
from datetime import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from scipy import stats
import streamlit as st

# In[115]:


#Query the data needed for the project from PostgreSQL database

conn = psycopg2.connect(
    dbname='opc',
    user='postgres',
    password='postgresJavi',
    host='localhost'
)

customer_query = "SELECT * FROM customers"
customers = pd.read_sql_query(customer_query, conn)

orders_query = "SELECT * FROM orders"
orders = pd.read_sql_query(orders_query, conn)

warehouse_returns_query = "SELECT * FROM customerwarehousereturns"
returns = pd.read_sql_query(warehouse_returns_query, conn)

# Import another dataframe needed for the project from a local source
churn = pd.read_csv(r'C:\Users\javis\Downloads\opc_cus_churn_2022.csv')


# # DATA PROCESSING & EDA

# Eliminate columns from each dataframe that will not be used for this project
customers = customers[['cus_id', 'cus_join_date', 'tot_ord_qty', 'tot_ord_value']]
orders = orders[['ord_date', 'ord_ship_date', 'cus_id']]

# Creation new desired features in each dataframe

#Days since first purchase (assuming the date of report completion is 1/1/2023 since the churn dataset contains data until 2022) & select only necessary columns for analysis
customers['cus_join_date'] = pd.to_datetime(customers['cus_join_date'])
today = pd.to_datetime('2023-01-01')
customers['days_since_first'] = (today - customers['cus_join_date']).dt.days
customers1 = customers[['cus_id', 'tot_ord_qty', 'tot_ord_value','days_since_first']]

#Shipping time & select only necessary columns for analysis
pd.options.mode.chained_assignment = None  # Eliminate copy warning
orders['ord_ship_date'] = pd.to_datetime(orders['ord_ship_date'])
orders['ord_date'] = pd.to_datetime(orders['ord_date'])
orders['shipping_days'] = (orders['ord_ship_date'] - orders['ord_date']).dt.days
orders1 = orders[['cus_id', 'shipping_days']]

#Returns per customer & select only necessary columns for analysis. Also transform NaN to 0 since they mean that there wasn't a return
returns['n_returns'] = returns.groupby('cus_id')['cus_id'].transform('count')
returns1 = returns[['cus_id', 'n_returns']]


# All the dataframes will now be joined. As the goal of this project is to predict churn, the churn dataset will be used as the starting dataset.
df = pd.merge(churn, customers1, how = 'left', on = 'cus_id')
df = pd.merge(df, orders1, how = 'left', on = 'cus_id')
df = pd.merge(df, returns1, how = 'left', on = 'cus_id')


# Now it is time to fix the dataset and get it ready for EDA and modeling

# Convert income column into integers
df['cus_income'] = df['cus_income'].str.replace('[$,]', '', regex=True).astype(float).astype(int)

# Convert the financed_purchased to 0 and 1 (1=Y)
df['financed_purchase'] = df['financed_purchase'].map({'Y': 1, 'N': 0})

# The NaN values for n_returns corresnpond to customers without returns, so they will be replaced by 0
df['n_returns'] = df['n_returns'].fillna(0)


# Now, lets deal with NaNs and duplicates

# Duplicates
print('There are', df.duplicated().sum(), 'duplicates') # There are no duplicates


# I will proceed and fill the NaNs. As seen in the distributions, there are long tails present, so the median will be used to fill the NaNs
columns_to_impute = ['tot_ord_qty', 'tot_ord_value', 'days_since_first', 'shipping_days']

for column in columns_to_impute:
    median_value = df[column].median()
    df[column].fillna(median_value, inplace=True)
    
# Verify there are non NaNs left
df.isna().sum()
# Convert all values to integers
df['tot_ord_qty'] = df['tot_ord_qty'].astype(int)
df['tot_ord_value'] = df['tot_ord_value'].astype(int)
df['days_since_first'] = df['days_since_first'].astype(int)
df['shipping_days'] = df['shipping_days'].astype(int)
df['n_returns'] = df['n_returns'].astype(int)


df.info()


df.describe()


# Let's display some boxplots to see the distributions of the numerical values
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
sns.boxplot(ax=axes[0, 0], data=df, y='cus_income').set_title('Customer Income')
sns.boxplot(ax=axes[0, 1], data=df, y='tot_ord_qty').set_title('Total Order Quantity')
sns.boxplot(ax=axes[0, 2], data=df, y='tot_ord_value').set_title('Total Order Value')
sns.boxplot(ax=axes[1, 0], data=df, y='days_since_first').set_title('Days Since First Order')
sns.boxplot(ax=axes[1, 1], data=df, y='shipping_days').set_title('Shipping Days')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# Now it's time to deal with outliers, I will use z-scores to do so with a threshold of a z-score of 3
from scipy.stats import zscore

columns = ['cus_income', 'tot_ord_qty', 'tot_ord_value', 'days_since_first', 'shipping_days', 'n_returns']

for column in columns:
    
    df[column + '_zscore'] = zscore(df[column])
    outliers = df[df[column + '_zscore'].abs() > 3]

    high_cap = df[column].quantile(0.95)
    low_cap = df[column].quantile(0.05)
    df.loc[df[column + '_zscore'] > 3, column] = high_cap
    df.loc[df[column + '_zscore'] < -3, column] = low_cap

    df.drop(columns=[column + '_zscore'], inplace=True)


# # Models

# Build logistic regression model using skicit-learn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler

X = df.drop('churn', axis=1)
y = df.churn

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]


# Predict churn probabilities and add this column to the dataset

probabilities = model.predict_proba(X_scaled)[:, 1]
df_p = df
df_p['predicted_probability'] = probabilities
df[df['churn'] == 0].sort_values(by='predicted_probability', ascending=False)


# Build linear regression model to predict sale
from sklearn.linear_model import LinearRegression

df2 = pd.merge(df, orders[['cus_id', 'ord_ship_date']])
df2['ord_date'] = pd.to_datetime(df2['ord_ship_date'])
df2.set_index('ord_date', inplace=True)
df2.sort_index(inplace=True)
df2['time'] = (df2.index - df2.index[0]).days

X1 = df2['time'].values.reshape(-1, 1)
y2 = df2['tot_ord_value'].values

model2 = LinearRegression()
model2.fit(X1, y2)


# Predict the next four quarterly values
max_time = df2['time'].iloc[-1]  
quarterly_periods = 91.25
future_periods = 4  
future_times = np.array([max_time + (i+1) * quarterly_periods for i in range(4)]).reshape(-1, 1)
future_times = future_times.reshape(-1, 1)
future_dates = [df2.index[0] + pd.DateOffset(days=int(f_time)) for f_time in future_times.flatten()]
forecast = model2.predict(future_times)
predictions = model2.predict(future_times)

# # Streamlit Implementation

import streamlit as st # type: ignore

def about_page():
    st.title('Ruby Kestrel OPC Data App')
    st.title('About This Application')
    st.write("""
    Welcome! This application is designed to assist the OPC leadership by providing interactive data visualizations \
    and predictive analytics based on historical sales and customer data. Users can explore data, \
    see the impact of different variables on customer churn, and utilize a predictive model to identify \
    at-risk customers before they churn. 
    """)
    st.write("""
    OPC is having some issues related to churn. This application provides the data resources and insights to avoid this \
    this churn and allow to make informed decisions on the future based on current data. This application has different \
    sections. Explore the data, where the user will be able to get familiar with the data, variables and how these related. \
    The model section, which will include vital information about the model. And the conclusion section, which \
    will respond to the 4 business problems presented by OPC, which are the following.\n
        1. Identify trends or indicators that point to a customer's potential for churn.\n
        2. Identify and predict a new customer potential for churn.\n
        3. Predict with at least 75% certainty that a customer will churn or not.\n
        4. Identify potential revenue generation fot he next 4 quarters based on the previous 4 years of revenue data.                 
    """)
    st.write("""
    Javier Sanchez Clemente/ Walsh University/ DAM 415 Principles and Techniques of Data Analytics II/ Professor Dr. Sabasi)
             """)

# Function to display the EDA page
def eda_page():
    st.title('Explore the Data')
    st.write('Welcome to the explore the data section. In this sections you will be able to see some key graphs of some of the \
             key data for OPC. Select the variable you want to display the graphs of on the left side. You will be able to \
             see a histogram, a scatter plot and a correlation heatmap between all variables. Lastly, you can also select to only display graphs \
             for users that have only churned or users that have not. Enjoy!')


    # Add more plots and interactive elements here
    # Sidebar for filtering
    st.sidebar.header('Filter Data')

    # Filter by Customer Churn Status
    churn_status = st.sidebar.radio('Churn Status', ['All', 'Churned', 'Not Churned'])
    if churn_status == 'Churned':
        data_filtered = df[df['churn'] == 1]
    elif churn_status == 'Not Churned':
        data_filtered = df[df['churn'] == 0]
    else:
        data_filtered = df

    # Numeric column selection for histogram
    numeric_column = st.sidebar.selectbox('Select a column for the histogram', ['cus_income', 'tot_ord_qty', 'tot_ord_value', 'days_since_first', 'shipping_days', 'n_returns'])
    # Creating a histogram
    fig, ax = plt.subplots()
    ax.hist(data_filtered[numeric_column], bins=20, color='skyblue')
    ax.set_title(f'Histogram of {numeric_column}')
    ax.set_xlabel(numeric_column)
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Scatter plot settings for examining relationships
    x_axis_val = st.sidebar.selectbox('Select X-axis for scatter plot', ['tot_ord_qty', 'tot_ord_value', 'days_since_first', 'shipping_days'])
    y_axis_val = st.sidebar.selectbox('Select Y-axis for scatter plot', ['cus_income', 'n_returns', 'predicted_probability'])
    # Creating a scatter plot
    fig, ax = plt.subplots()
    ax.scatter(data_filtered[x_axis_val], data_filtered[y_axis_val], alpha=0.5)
    ax.set_title(f'Scatter plot of {x_axis_val} vs {y_axis_val}')
    ax.set_xlabel(x_axis_val)
    ax.set_ylabel(y_axis_val)
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader('Correlation Heatmap')
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Function to display the Model page
def model_page():
    st.title('Model Insights')
    st.write('Welcome to the model, where the magic happens!. Here is the information about the logistic regression model \
             that can predict churn based on previous data. This algorith learns on the available opc data and is able to predict \
             whether a customer will churn or not, as well as assigning a probability to each customer. On this part you will be able \
             to first see some scoring metrics about the model to see how good it does.')


    # Display model evaluation
    st.subheader('Model Evaluation Metrics')
    st.write('Classification Report:')
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)

# Final insights page
def conclusion_insights():
    st.title('Conclusion & Final Insights')
    st.write('''Here is the final insights part of this application, that answers the 4 business problems presented by OPC. \
             ''')
    st.markdown('''
                ## 1. Identify trends or indicators that point to a customer's potential for churn.
                ''')
    st.write('''Here is a plot that displays the importance of each feature used in the model. There are both positive and negative predictors \
             ''')
    coefficients = model.coef_[0]  
    feature_importance = pd.DataFrame(coefficients, index=X.columns, columns=['Coefficient'])
    feature_importance['Absolute Coefficient'] = feature_importance['Coefficient'].abs()
    feature_importance = feature_importance.sort_values(by='Absolute Coefficient', ascending=False)
    fig, ax = plt.subplots()
    feature_importance['Coefficient'].plot(kind='bar')
    ax.set_title('Feature Importance in Logistic Regression')
    ax.set_xlabel('Features')
    ax.set_ylabel('Coefficient Value')
    st.pyplot(fig)

    st.markdown('''
                ## 2. Identify and predict a new customer potential for churn. & 3. Predict with at least 75% certainty that a customer will churn or not.
                ''') 
    st.write('''As seen in the model page. The model clearly surpasses the 75% certainty threshold asked by OPC leadership. Now two sample \
             tables will be displayed. One that contains that churn probabilities of each user and the other that contains the users who have not \
             churned and have the highest probability of risk, as these are the priotity for OPC leadership.
             ''')
    
    #Show 2 dataframes and create buttons to download them
    st.dataframe(df)
    df_churn = df[df['churn'] == 0].sort_values(by='predicted_probability', ascending = False)

    def convert_df_to_csv(df): # Convert DataFrame to CSV string
        return df.to_csv(index=False).encode('utf-8')
    df_csv = convert_df_to_csv(df)
    st.download_button(
    label="Download data as CSV",
    data=df_csv,
    file_name='data.csv',
    mime='text/csv',
)
    df_churn_csv = convert_df_to_csv(df_churn)
    st.dataframe(df_churn)
    st.download_button(
    label="Download data as CSV",
    data=df_churn_csv,
    file_name='data2.csv',
    mime='text/csv',
)
    st.markdown(''' ## 4. Identify potential revenue generation fot he next 4 quarters based on the previous 4 years of revenue data.
                ''')
    st.write('''A simple linear regression model was able to predict sales. Here is a scatterplot showing it
             ''')
    
    # Show linear regression scatterplot that predicts sales
    fig, ax = plt.subplots()
    ax.scatter(df2.index, df2['tot_ord_value'], color='black', label='Actual Sales')
    ax.plot(df2.index, model2.predict(X1), color='blue', label='Regression Line')
    ax.scatter(future_dates, predictions, color='red', label='Predictions for Future Periods')
    ax.set_title('Yearly Sales Prediction')
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Sales')
    ax.legend()
    st.pyplot(fig)
    
    st.write(f"Q1 Forecasted Revenue: {forecast[0]:,.2f}")
    st.write(f"Q2 Forecasted Revenue: {forecast[1]:,.2f}")
    st.write(f"Q3 Forecasted Revenue: {forecast[2]:,.2f}")
    st.write(f"Q4 Forecasted Revenue: {forecast[3]:,.2f}")



# Main app function
def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["About", "Explore the Data", "Model Insights", "Insights & Conclusion"])
    if selection == "About":
        about_page()
    elif selection == "Explore the Data":
        eda_page()
    elif selection == "Model Insights":
        model_page()
    elif selection == "Insights & Conclusion":
        conclusion_insights()
if __name__ == "__main__":
    main()






