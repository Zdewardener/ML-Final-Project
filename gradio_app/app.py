import pandas as pd
import numpy as np
import gradio as gr
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


print("Importing data...")
Statewide = pd.read_csv("Statewide.csv")
Statewide_target = Statewide["TGT STATEWIDE PRIMARY"]
Statewide = Statewide.drop(["TGT STATEWIDE PRIMARY", "TGT PARTY AFFILIATION"], axis = 1)
print("IMPORTED!")

print("Splitting data into train/test.")
X_train, X_test, y_train, y_test = train_test_split(Statewide, Statewide_target, test_size = .50, stratify = Statewide_target)
print("TRAIN/TEST COMPLETE!")

print("Fitting model...")
bestForest = RandomForestClassifier(**{'n_estimators': 2000, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 14, 'criterion': 'gini'})
bestForest.fit(X_train, y_train)
print("MODEL TRAINED!")

cityList = ["BARRINGTON", "BRISTOL", "BURRILLVILLE", "CENTRAL FALLS", "CHARLESTOWN", "COVENTRY", "CRANSTON", "CUMBERLAND", "EAST GREENWICH", "EAST PROVIDENCE", "EXETER", "FOSTER", "GLOCESTER", "HOPKINTON", "JAMESTOWN", "JOHNSTON", "LINCOLN", "LITTLE COMPTON", "MIDDLETOWN", "NARRAGANSETT", "NEW SHOREHAM", "NEWPORT", "NORTH KINGSTOWN", "NORTH PROVIDENCE", "NORTH SMITHFIELD", "PAWTUCKET", "PORTSMOUTH", "PROVIDENCE", "RICHMOND", "SCITUATE", "SMITHFIELD", "SOUTH KINGSTOWN", "TIVERTON", "WARREN", "WARWICK", "WEST GREENWICH", "WEST WARWICK", "WESTERLY", "WOONSOCKET"]
dict_party = {"No Party": 0, "Unaffiliated": 1, "Democrat": 2, "Republican": 3, "Moderate": 4}
dict_cities = {"BARRINGTON": 0, "BRISTOL": 1, "BURRILLVILLE": 2, "CENTRAL FALLS": 3, "CHARLESTOWN": 4, "COVENTRY": 5, "CRANSTON": 6, "CUMBERLAND": 7, "EAST GREENWICH": 8, "EAST PROVIDENCE": 9, "EXETER": 10, "FOSTER": 11, "GLOCESTER": 12, "HOPKINTON": 13, "JAMESTOWN": 14, "JOHNSTON": 15, "LINCOLN": 16, "LITTLE COMPTON": 17, "MIDDLETOWN": 18, "NARRAGANSETT": 19, "NEW SHOREHAM": 20, "NEWPORT": 21, "NORTH KINGSTOWN": 22, "NORTH PROVIDENCE": 23, "NORTH SMITHFIELD": 24, "PAWTUCKET": 25, "PORTSMOUTH": 26, "PROVIDENCE": 27, "RICHMOND": 28, "SCITUATE": 29, "SMITHFIELD": 30, "SOUTH KINGSTOWN": 31, "TIVERTON": 32, "WARREN": 33, "WARWICK": 34, "WEST GREENWICH": 35, "WEST WARWICK": 36, "WESTERLY": 37, "WOONSOCKET": 38}
dict_election = {"voted": 1, "didn't vote": 0}

def get_propensity(city, zip, curparty, yob, e3, e4, e5, e6, e7, e8, p5, p6, p8):
  proba = bestForest.predict_proba(np.array([city, zip, curparty, yob, e3, e4, e5, e6, e7, e8, p5, p6, p8]).reshape(1, -1))
  proba = ((proba.tolist())[0][1] * 100)
  propensity = round(proba, 2)
  return f"Chance of having voted in the RI 2022 Statewide Primary Elections: {propensity}%"

def predict(city, zip, curparty, yob, e3, e4, e5, e6, e7, e8, p5, p6, p8):
  city = dict_cities.get(city)
  zip = int(zip)
  curparty = dict_party.get(curparty)
  yob = int(yob)
  e3 = dict_election.get(e3)
  e4 = dict_election.get(e4)
  e5 = dict_election.get(e5)
  e6 = dict_election.get(e6)
  e7 = dict_election.get(e7)
  e8 = dict_election.get(e8)
  p5 = dict_party.get(p5)
  p6 = dict_party.get(p6)
  p8 = dict_party.get(p8)
  return get_propensity(city, zip, curparty, yob, e3, e4, e5, e6, e7, e8, p5, p6, p8)


print("Starting Gradio app!")
app = gr.Interface(
    predict, 
    [
        gr.Dropdown(cityList, label = "City/Town"),
        gr.Number(label = "Zip Code"),
        gr.Radio(["No Party", "Unaffiliated", "Democrat", "Republican"], label = "Current Party"),
        gr.Number(label = "Year of Birth"),
        gr.Radio(["voted", "didn't vote"], label = "2021 Special Statewide Referenda Election"),
        gr.Radio(["voted", "didn't vote"], label = "2020 Statewide Election"),
        gr.Radio(["voted", "didn't vote"], label = "2020 Statewide Primary Election"),
        gr.Radio(["voted", "didn't vote"], label = "2020 Presidential Primary Election"),
        gr.Radio(["voted", "didn't vote"], label = "2018 Statewide General Election"),
        gr.Radio(["voted", "didn't vote"], label = "2018 Statewide Primary Election"),
        gr.Radio(["No Party", "Unaffiliated", "Democrat", "Republican", "Moderate"], label = "Party for 2020 Statewide Primary"),
        gr.Radio(["No Party", "Unaffiliated", "Democrat", "Republican", "Moderate"], label = "Party for 2020 Presidential Primary"),
        gr.Radio(["No Party", "Unaffiliated", "Democrat", "Republican", "Moderate"], label = "Party for 2018 Statewide Primary")
    ], "text"
)

app.launch()