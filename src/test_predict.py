# https://sebhastian.com/no-module-named-requests/
# pip install requests
# For pip3:
# pip3 install requests
import requests

# Define the URL of your Flask API
url = 'http://127.0.0.1:12345/predict'

# Define the input data as a dictionary.
test_data = [
    {
        # This is (0, setosa).
        'sepal length (cm)': 5.1,
        'sepal width (cm)': 3.5,
        'petal length (cm)': 1.1, # 1.4
        'petal width (cm)': 0.2
    },
    {
        # This is (2, virginica).
        'sepal length (cm)': 6.3, # 7.3
        'sepal width (cm)': 2.9,
        'petal length (cm)': 6.3,
        'petal width (cm)': 1.8
    },
    {
        # This is (1, versicolor).
        'sepal length (cm)': 6.7,
        'sepal width (cm)': 2.8, # 3.1
        'petal length (cm)': 4.4,
        'petal width (cm)': 1.4
    }
]

# Send a POST request to the API with the input data
response = requests.post(url, json=test_data)

# Check the HTTP response status code
# For a full list of response codes, go to https://www.guru99.com/testing-rest-api-manually.html.
if response.status_code == 200:
    # Parse and print the JSON response (assuming it contains the prediction)
    prediction = response.json()
    print(prediction)
else:
    # Handle the case where the API request failed
    print(f'API Request Failed with Status Code: {response.status_code}')
    print(f'Response Content: {response.text}')
