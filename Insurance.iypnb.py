
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('C:\\Users\\AA.Y TRADEERS\\Downloads\\Data.csv\\.venv\\insurance_data.csv')
print(df)
print(df['bought_insurance'].dtype)
print(df['age'].dtype)

"""
plt.scatter(df.age,
            df.bought_insurance,
            marker='*',
            color='red',
            label='Actual Data')
plt.xlabel('age',color='blue')
plt.ylabel('bought_insurance,(0=No,Yes=1)',color='blue')
plt.title('Age vs Bought Insurance',color='blue')

plt.show()
"""
print(df.shape)


X_train, X_test, y_train, y_test = train_test_split(
    df[['age']],df['bought_insurance'],
    test_size=0.2,
    random_state=42
    )
print(X_test)
print(y_train)




X = df[['age']]
y = df['bought_insurance']

model = LogisticRegression()
model.fit(X, y)

# 3. Yeh aasan function aapki di hui kisi bhi age par khud hi check karega
def check_insurance(user_age):
    # Jo bhi age user UI mein likhega, us ka table banega
    age_df = pd.DataFrame({'age': [user_age]})
    
    # Model khud hi haan ya naa ka faisla karega
    prediction = model.predict(age_df)[0]
    
    if prediction == 1:
        return "Haan, yeh customer insurance khareedega! (Yes)"
    else:
        return "Nahi, yeh customer insurance nahi khareedega. (No)"


model.predict(X_test)
accuracy = model.score(X_test,y_test)


# 2. predict_proba ka use karein
age_df = model.predict_proba(X_test)

# 3. Result ko print karein
print("Dono classes ki probability:", age_df)
print("Insurance NAHI lene ka imkaan (No = 0):", age_df[0][0])
print("Insurance LENE ka imkaan (Yes = 1):", age_df[0][1])

print(age_df = model.predict(age_df[[80]]))

"""
#train_test_split(df[['age']],df['bought_insurance'],test_size=0.2,random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    df[['age']],df['bought_insurance'],
    test_size=0.2,
    random_state=42
    )
print(X_test)
print(y_train)


model = LogisticRegression()
model.fit(X_train,y_train)
print('model is successfully trained')

print('===================')
model.predict(X_test)
accuracy = model.score(X_test,y_test)


# 2. predict_proba ka use karein
probablity = model.predict_proba(age_df)

# 3. Result ko print karein
print("Dono classes ki probability:", probablity)
print("Insurance NAHI lene ka imkaan (No = 0):", probablity[0][0])
print("Insurance LENE ka imkaan (Yes = 1):", probablity[0][1])



age_predicted = model.predict(age_df)
print(age_predicted)
print("Model ki Accuracy Score:", accuracy)
print(f"Model Kitne Percent Sahi Hai: {accuracy * 100}%")

"""



