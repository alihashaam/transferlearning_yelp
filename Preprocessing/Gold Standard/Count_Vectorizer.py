
import os, time
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

Business = 'Bars' 
df = pd.read_csv('Processed Datasets with Feature Engineering/Featured_Preprocessed_'+Business+'_Reviews.csv', index_col=0)

count_vect = CountVectorizer(lowercase=False)

start_loadingData = time.time()
X_train_counts = count_vect.fit_transform(df['text'])
end_loadingData = time.time()
print("Time to Process the data: " + str(end_loadingData - start_loadingData) + " second(s)")


from sklearn.externals import joblib
joblib.dump(X_train_counts, 'count_vectorizers/golden_'+Business+'_x.pkl')
joblib.dump(df['stars'], 'count_vectorizers/golden_'+Business+'_y.pkl')

