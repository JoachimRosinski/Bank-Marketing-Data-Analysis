import configparser
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

config = configparser.ConfigParser()

config.read('E:\\Python\\BankProject\\db_config.ini')
db_config = {
    'host': config['mysql']['host'],
    'user': config['mysql']['user'],
    'password': config['mysql']['password'],
    'database': config['mysql']['database']
}

db_url = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"

engine = create_engine(db_url)

try:
    query = "SELECT * FROM bank_marketing"
    df = pd.read_sql(query, engine)
    print(df.head())
    df.ffill(inplace=True)
    label_encoders = {}
    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']

    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    variable_names = {
        'age': 'Age',
        'job': 'Job',
        'marital': 'Marital Status',
        'education': 'Education',
        'default': 'Has Credit in Default',
        'balance': 'Average Yearly Balance',
        'housing': 'Has Housing Loan',
        'loan': 'Has Personal Loan',
        'contact': 'Contact Type',
        'day': 'Last Contact Day',
        'month': 'Last Contact Month',
        'duration': 'Last Contact Duration',
        'campaign': 'Number of Contacts in Campaign',
        'pdays': 'Days Since Last Contacted',
        'previous': 'Previous Contacts',
        'poutcome': 'Previous Campaign Outcome',
        'y': 'Subscribed to Term Deposit'
    }

    X = df.drop('y', axis=1)
    y = df['y']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    feature_importances = model.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    })

    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    print(importance_df)

    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()

    top_features = importance_df['Feature'].head(5).tolist()

    df['y'] = label_encoders['y'].inverse_transform(df['y'])
    for feature in top_features:
        if df[feature].dtype != 'object':
            correlation = df[df['y'] == 'yes'][feature].corr(df[df['y'] == 'yes']['y'])
            print(f"Correlation between '{variable_names[feature]}' and 'Subscribed to Term Deposit (yes)': {correlation:.2f}")

except Exception as e:
    print(f"Error: {e}")

finally:
    print("Process completed")