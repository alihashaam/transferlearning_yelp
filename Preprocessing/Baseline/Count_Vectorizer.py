
import os, time
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

Business = 'Home Services' 

df = pd.read_csv('Preprocessed_'+Business+'_Reviews_Baseline.csv', index_col=0)

count_vect = CountVectorizer(lowercase=False)

start_loadingData = time.time()
X_train_counts = count_vect.fit_transform(df['text'])
end_loadingData = time.time()
print("Time to Process the data: " + str(end_loadingData - start_loadingData) + " second(s)")


from sklearn.externals import joblib
joblib.dump(X_train_counts, 'baseline_'+Business+'_x.pkl')
joblib.dump(df['stars'], 'baseline_'+Business+'_y.pkl')

