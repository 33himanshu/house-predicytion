#Import Libraries
import numpy as np
import pandas as pd
import joblib
import os 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#load data
df = pd.read_csv("data/ohe_data_reduce_cat_class.csv")
# Split data
X= df.drop('price', axis=1)
y= df['price']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=51)

# feature scaling
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define file paths
DATA_FILE = os.path.join(current_dir, "data", "ohe_data_reduce_cat_class.csv")
MODEL_FILE = os.path.join(current_dir, "banglore_home_prices_model.pickle") 

# Check if files exist
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Data file not found at: {DATA_FILE}")
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model file not found at: {MODEL_FILE}")

# Load data and model
df = pd.read_csv(DATA_FILE)
model = joblib.load(MODEL_FILE)


# it help to get predicted value of house  by providing features value 
def predict_house_price(bath,balcony,total_sqft_int,bhk,price_per_sqft,area_type,availability,location):

  x =np.zeros(len(X.columns)) # create zero numpy array, len = 107 as input value for model

  # adding feature's value accorind to their column index
  x[0]=bath
  x[1]=balcony
  x[2]=total_sqft_int
  x[3]=bhk
  x[4]=price_per_sqft

  if "availability"=="Ready To Move":
    x[8]=1

  if 'area_type'+area_type in X.columns:
    area_type_index = np.where(X.columns=="area_type"+area_type)[0][0]
    x[area_type_index] =1

  if 'location_'+location in X.columns:
    loc_index = np.where(X.columns=="location_"+location)[0][0]
    x[loc_index] =1

  # feature scaling
  x = sc.transform([x])[0] # give 2d np array for feature scaling and get 1d scaled np array

  return model.predict([x])[0] # return the predicted value by train XGBoost model


# Example usage:
bath_ex = 2
balcony_ex = 1
total_sqft_int_ex = 1200
bhk_ex = 2
price_per_sqft_ex = 5000.0
area_type_ex = "Super built-up  Area" # Note the double space from columns.json
availability_ex = "Ready To Move"
location_ex = "Whitefield"

predicted_price = predict_house_price(
    bath_ex,
    balcony_ex,
    total_sqft_int_ex,
    bhk_ex,
    price_per_sqft_ex,
    area_type_ex,
    availability_ex,
    location_ex
)

print(f"\n--- Example Prediction ---")
print(f"Input Features:")
print(f"  Bathrooms: {bath_ex}")
print(f"  Balconies: {balcony_ex}")
print(f"  Total Sqft: {total_sqft_int_ex}")
print(f"  BHK: {bhk_ex}")
print(f"  Price per Sqft: {price_per_sqft_ex}")
print(f"  Area Type: {area_type_ex}")
print(f"  Availability: {availability_ex}")
print(f"  Location: {location_ex}")
print(f"\nPredicted Price (Lakhs INR): {predicted_price:.2f}")
print(f"--------------------------")
