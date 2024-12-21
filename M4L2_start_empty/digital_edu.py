#створи тут свій індивідуальний проект!
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_csv('train.csv')

data.drop(columns=["life_main","people_main","career_start","career_end","id","last_seen"],inplace=True)

def split_bdate(bdate):
    if not isinstance(bdate, str):
        return None
    parts = bdate.split('.')
    if len(parts)==3:
        return int(parts[2])
    elif len(parts)==2:
        return None

data["birth_year"]=data["bdate"].apply(split_bdate)

def fill_byear(row):
    if pd.isnull(row["birth_year"]):
        if row["sex"] == 1:
            return data[data["sex"]==1]["birth_year"].median()
        else:
            return data[data["sex"]==2]["birth_year"].median()
    return row["birth_year"]

data["birth_year"]=data.apply(fill_byear,axis=1)

def convert_sex(sex):
    if sex ==2:
        return 1
    else:
        return 0

data["sex"] = data["sex"].apply(convert_sex)

def convert_ef(education_form):
    if education_form == "Full-time":
        return 1
    else:
        return 0

data["education_form"] = data["education_form"].apply(convert_ef)

def convert_langs(langs):
    if "Українська" in str(langs):
        return 1
    else:
        return 0
    
data["langs"]=data["langs"].apply(convert_langs)

data = pd.get_dummies(data, columns=["education_status","relation","occupation_type"],drop_first=True)

data.fillna(0 ,inplace=True)

x = data.drop('result',axis=1)
y = data["result"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

scaler = StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

km = KNeighborsClassifier(n_neighbors=5)
km.fit(x_train,y_train)

y_pred = km.predict(x_test)
accuracy = accuracy_score(x_test,y_pred)

print(f"Accuracy:{accuracy:2f}")
