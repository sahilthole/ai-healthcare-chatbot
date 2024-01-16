from flask import Flask, request, jsonify, render_template
from sklearn.tree import DecisionTreeClassifier,_tree
from sklearn import preprocessing
import joblib  # Assuming you used joblib to save your model
import pandas as pd
import re
import numpy as np
import csv
from sklearn.model_selection import train_test_split
training = pd.read_csv('Training.csv')
symptom_desc=pd.read_csv('symptom_Description.csv')
symptom_prec =pd.read_csv('symptom_precaution.csv')
symptom_severity=pd.read_csv('Symptom_severity.csv')
reduced_data = training.groupby(training['prognosis']).max()
le = preprocessing.LabelEncoder()
y = training['prognosis']
le.fit(y)
y = le.transform(y)
app = Flask(__name__)
severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()
def getDescription():
    global description_list
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)
def getSeverityDict():
    global severityDictionary
    with open('Symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass

def getprecautionDict():
    global precautionDictionary
    with open('symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)
getDescription()
getSeverityDict()
getprecautionDict()
# Load your machine learning model during app initialization
model1_path = 'models/decision_tree_model.pkl'
clf1 = joblib.load(model1_path)
model2_path ='models/svm_model.pkl'
clf2=joblib.load(model2_path)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_json=request.json
        print(input_json)
        symptom =input_json.get("Symptom")
        # Retrieve user input from the form
       
        # Perform prediction using the loaded model
        
        def check_pattern(dis_list,inp):
            pred_list=[]
            inp=inp.replace(' ','_')
            patt = f"{inp}"
            regexp = re.compile(patt)
            pred_list=[item for item in dis_list if regexp.search(item)]
            if(len(pred_list)>0):
                return 1,pred_list
            else:
                return 0,[]


        def print_disease(node):
            node = node[0]
            val  = node.nonzero() 
            disease = le.inverse_transform(val[0])
            return list(map(lambda x:x.strip(),list(disease)))

        def tree_to_code(tree, feature_names,disease_input):
            tree_ = tree.tree_
            feature_name = [
                feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                for i in tree_.feature
            ]

            chk_dis=",".join(feature_names).split(",")
            symptoms_present = []

            while True:

            
                
                conf,cnf_dis=check_pattern(chk_dis,disease_input)
                if conf==1:
                    
                    for num,it in enumerate(cnf_dis):
                        print()
                    if num!=0:
                        return jsonify({
                            "error" : "Please give valid symptoms"
                        })
                        conf_inp = int(input(""))
                    else:
                        conf_inp=0

                    disease_input=cnf_dis[conf_inp]
                    break
                    
                else:
                    return jsonify({
                        "error" : "Please give valid symptoms"
                    })

            
            def recurse(node, depth):
                indent = "  " * depth
                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    name = feature_name[node]
                    threshold = tree_.threshold[node]

                    if name == disease_input:
                        val = 1
                    else:
                        val = 0
                    if  val <= threshold:
                        return recurse(tree_.children_left[node], depth + 1)
                    else:
                        symptoms_present.append(name)
                        return recurse(tree_.children_right[node], depth + 1)
                else:
                    present_disease = print_disease(tree_.value[node])
                    
                    red_cols = reduced_data.columns 
                    symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
                    
                    return symptoms_given.values.tolist()

            return  recurse(0, 1)
            


        result = tree_to_code(clf1,training.columns,symptom)


        # Return the result as JSON
        return jsonify({'response': result})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/svm1', methods=['POST'])
def svm1():
    sympt = request.json
    sympt_list = sympt.get("Symptom_list")
    print(sympt_list)
    
    def sec_predict(symptoms_exp):
        df = training
        X = df.iloc[:, :-1]
        y = df['prognosis']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
        rf_clf = DecisionTreeClassifier()
        rf_clf.fit(X_train, y_train)

        symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
        input_vector = np.zeros(len(symptoms_dict))
        for item in symptoms_exp:
            input_vector[[symptoms_dict[item]]] = 1

        return rf_clf.predict([input_vector])

    predicted_disease =sec_predict(sympt_list)
    print(predicted_disease)

    def calc_condition(exp,days):
        sum=0
        for item in exp:
            sum=sum+severityDictionary[item]
        if((sum*days)/(len(exp)+1)>13):
            return 1
        else:
            return 0
    days = sympt.get("days")
    calc = calc_condition(sympt_list,days)
    desc = description_list[predicted_disease[0]]
    prec = precautionDictionary[predicted_disease[0]]
    print(calc)
    if calc==1:
        calc="You should take the consultation from doctor. "
    else:
        calc="It might not be that bad but you should take precautions."
    data = {
        "calc": calc,
        "desc": desc,
        "prec":" ".join([f"{i+1}. {item}" for i, item in enumerate(prec)])

    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)


    


