import os
from flask import Flask, request, flash, render_template, jsonify, json
from function import preprocess_data, result_svm
import csv
import pandas

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

#upload files
app.config['UPLOAD_FOLDER']='uploads'
ALLOWED_EXTENSION = set(['csv'])

def allowed_file(filename):
  return '.' in filename and \
    filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION

@app.route('/upload-file', methods=['GET', 'POST'])
def upload_file():
  if request.method == 'GET':
    return render_template('uploaddata.html')
    
  elif request.method == 'POST':
    # check if the post request has the file part
    if 'file' not in request.files:
      return redirect(request.url)

    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
      return redirect(request.url)

    if file and allowed_file(file.filename):
      file.filename = "dataset.csv"
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
      text = pandas.read_csv('uploads/dataset.csv', encoding='latin-1')
      # result = text.to_json(r'uploads/dataset.json')
      return render_template('uploaddata.html',tables=[text.to_html(classes='table table-bordered', table_id='dataTable')])

@app.route('/preprocessing', methods=['GET', 'POST'])
def preprocess():
  return render_template ('preprocessing.html')


@app.route('/preprocessing/result', methods=['GET', 'POST'])
def preprocessing():
  text = pandas.read_csv('uploads/dataset.csv', encoding='latin-1')
  text.drop(['Date','Author'], axis=1, inplace=True)
  text['Text'] = text['Text'].apply(lambda x:preprocess_data(x))
  text.to_csv('uploads/dataset_clear.csv', index = False, header = True)
  return render_template('preprocessing.html',tables=[text.to_html(classes='table table-bordered', table_id='dataTable')])

# def data(text):
#     text['label'] = text['label'].map({'positif': 2, 'negatif': 1, 'netral': 0})
#     X = text['Text'].fillna(' ')
#     y = text['label']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=2)
#     return X_train, X_test, y_train, y_test

labels = [
  'POSITIF', 'NEGATIF', 'NETRAL'
]

colors = [
  '#1cc88a', '#e74a3b', '#f6c23e'
]

@app.route('/grafik-data', methods=['GET', 'POST'])
def page():
  return render_template ('klasifikasisvm.html')


@app.route('/grafik-data/result', methods=['GET', 'POST'])
def klasifikasisvm():
  text = pandas.read_csv('uploads/dataset_clear.csv', encoding='latin-1')

  accuracy_rbf, y_test = result_svm(text)
  accuracy_rbf = (round(accuracy_rbf, 2) * 100)
  
  y_test = y_test.reset_index()
  netral, negatif, positif = y_test['label'].value_counts()
  total = positif + negatif + netral
  # print(y_test['label'].value_counts() )

  pie_labels = labels
  pie_colors = colors
  pie_values = [positif, negatif, netral]

  bar_labels = labels
  bar_values = [positif, negatif, netral]
  
  return render_template ('klasifikasisvm.html', tweet_positive = positif, tweet_negative = negatif, tweet_netral = netral, total_tweet = total, accuracy_rbf = accuracy_rbf, labels = pie_labels, colors = pie_colors, values = pie_values, bar_labels = bar_labels, bar_values = bar_values)

@app.route('/tesmodel', methods=['GET', 'POST'])
def tesmodelpage():
  return render_template ('tesmodel.html')

@app.route('/tesmodel/result', methods=['GET', 'POST'])
def tesmodel():
  # Loading model to compare the results
  model = pickle.load(open('uploads/rbf.model','rb'))
  vectorizer = pickle.load(open('uploads/vectorizer.model','rb'))

  text = request.form['text']
  original_text = request.form['text']

  hasilprepro = preprocess_data(text)
  hasiltfidf = vectorizer.transform([hasilprepro])

  # cek prediksi dari kalimat
  
  hasilsvm = model.predict(hasiltfidf)
  if hasilsvm == 0:
    hasilsvm = 'NETRAL'
  elif hasilsvm == 1:
    hasilsvm = 'NEGATIF'
  else:
    hasilsvm = 'POSITIF'
  
  return render_template ('tesmodel.html', original_text=original_text, hasilprepro=hasilprepro, hasilsvm=hasilsvm)

if __name__ == "__main__":
  app.run(debug=True)