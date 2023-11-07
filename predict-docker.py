import requests

url = 'http://localhost:9695/predict'

customer =  {"Gender": 'b',
 "Age": 39.25,
 "Debt": 9.5,
 "Married": 'u',
 "BankCustomer": 'g',
 "EducationLevel": 'm',
 "Ethnicity": 'v',
 "YearsEmployed": 6.5,
 "PriorDefault": 't',
 "Employed": 't',
 "CreditScore": 14,
 "DriversLicense": 'f',
 "Citizen": 'g',
 "ZipCode": 240,
 "Income": 4607
 }

response = requests.post(url, json=customer).json()

print(customer)
print(response)

if response['Approved'] == True:
    print('Credit Card Application is Approved')
else:
    print('Credit Card Application is Denied')