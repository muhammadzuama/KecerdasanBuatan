import csv
import requests
import json
import datetime

from sklearn.metrics import classification_report

url = "http://localhost:5000/predict"


def getData(filename, start, end):
    dataToSave = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            elif  line_count > end:
                with open(f'result/{start}_{end}.json', 'w+') as f:
                    json.dump(dataToSave, f) 
                break
            elif line_count > start:
                data = {
                        'age' : row[2],
                        'gender': row[1], 
                        'hypertension': row[3],
                        'heart_disease': row[4],
                        'smoking_history': row[5],
                        'bmi': row[6],
                        'HbA1c_level': row[7],
                        'blood_glucose_level': row[8],
                        }  

                r = requests.post(url, data=data) # Sends form data
                result = r.json()
                dataToSave.append({
                    "id" : row[0],
                    "tested_at": str(datetime.datetime.now()),
                    "input" : data,
                    "result" : result
                })
                print(f'\t data index { row[0]} .')
            line_count += 1
        print(f'Processed {line_count} lines.')

def test(filename, filetest):
    result_file = open(filename, 'r')
    results = json.load(result_file)
    
    expected = []
    predicted = []
    
    with open(filetest) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:     
                line_count += 1
                continue
            elif line_count == len(results) + 1:
                print("HELLO")
                break
            expected.append(float(row[1]))
            line_count += 1
    
    for i in range(len(results)):
        predicted.append(float(results[i]['result']['prediction']))

    # Menghitung Confusion Matrix secara manual
    tp = tn = fp = fn = 0
    for e, p in zip(expected, predicted):
        if e == 1 and p == 1:
            tp += 1
        elif e == 0 and p == 0:
            tn += 1
        elif e == 0 and p == 1:
            fp += 1
        elif e == 1 and p == 0:
            fn += 1

    # Menghitung metrik evaluasi
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"F1 Score: {f1_score:.2f}")

    report = classification_report(expected, predicted)
    
    print('============================================')
    print('Classification Report:')
    print(report)


# getData('totest.csv', 0, 1000)
    
test('result/0_1000.json', 'expected.csv')






