import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.datasets import load_svmlight_file
import numpy as np
from sklearn import grid_search
from sklearn.grid_search import GridSearchCV
import operator

training_file= pd.read_csv('/Users/nsilverblatt/Documents/456/Final Project/TrainingData.csv')
test_file = pd.read_csv('/Users/nsilverblatt/Documents/456/Final Project/ScoreData.csv')
X_test_file = test_file.reindex(columns=['age','male','friend_cnt','avg_friend_age','avg_friend_male','friend_country_cnt','subscriber_friend_cnt','songsListened','lovedTracks','posts','playlists','shouts','delta_friend_cnt','delta_avg_friend_age','delta_avg_friend_male','delta_friend_country_cnt','delta_subscriber_friend_cnt','delta_songsListened','delta_lovedTracks','delta_posts','delta_playlists','delta_shouts','tenure','good_country','delta_good_country','user_id'])
X = training_file.reindex(columns=['age','male','friend_cnt','avg_friend_age','avg_friend_male','friend_country_cnt','subscriber_friend_cnt','songsListened','lovedTracks','posts','playlists','shouts','delta_friend_cnt','delta_avg_friend_age','delta_avg_friend_male','delta_friend_country_cnt','delta_subscriber_friend_cnt','delta_songsListened','delta_lovedTracks','delta_posts','delta_playlists','delta_shouts','tenure','good_country','delta_good_country', 'user_id'])
y = training_file['adopter']
# remove features before running model. ex: forward selection, filtering, etc. use business understanding of variables


#25 attributes starting from 0

#delete columns by randomly removing the name manually from line 14 (the X = training_file.reindex(...)


sm = SMOTE(k_neighbors = 15, ratio = .019)
X_resampled, y_resampled = sm.fit_sample(X, y)



i=0
while i<2:
	X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=.6)
	xgtrain = xgb.DMatrix(X_train, y_train)
	xgtest = xgb.DMatrix(X_test, y_test)
	xgActualTestForSubmission = xgb.DMatrix(X_test_file)
	param = {'silent':1, 'objective':'binary:logistic'}
	bst = xgb.train(param, xgtrain, num_boost_round=2)
	myseries = pd.Series(bst.get_fscore()).sort_values(axis=0)
	y_pred = bst.predict(xgtest)
	y_pred = [1. if y_cont > .28  else 0. for y_cont in y_pred]
	y_true = y_test
	if f1_score(y_true, y_pred) > .13:
		j=0
		while j<2:
			X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=.6)
			xgtrain = xgb.DMatrix(X_train, y_train)
			xgtest = xgb.DMatrix(X_test, y_test)
			xgActualTestForSubmission = xgb.DMatrix(X_test_file)
			param = {'silent':1, 'objective':'binary:logistic'}
			bst = xgb.train(param, xgtrain, num_boost_round=2)
			myseries = pd.Series(bst.get_fscore()).sort_values(axis=0)
			y_pred = bst.predict(xgtest)
			y_pred = [1. if y_cont > .28  else 0. for y_cont in y_pred]
			y_true = y_test
			if f1_score(y_true, y_pred) > .116:
				print(f1_score(y_true, y_pred))
				print(myseries)
				j=2
				i=2



X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=.6)
xgtrain = xgb.DMatrix(X_train, y_train)
xgtest = xgb.DMatrix(X_test, y_test)
xgActualTestForSubmission = xgb.DMatrix(X_test_file)
# TUNE MAX_DEPTH AND OTHER PARAMETERS
# SEE IF THERE'S CROSS VALIDATION
param = {'max_depth':7, 'silent':1}
# max_depth 7 with test_size .6: .133577789419
bst = xgb.train(param, xgtrain, num_boost_round=2)
y_pred = bst.predict(xgtest)
y_pred = [1. if y_cont > .28  else 0. for y_cont in y_pred]
y_true = y_test
print(f1_score(y_true, y_pred))
				






bst.save_model("model_file_name")
bst = xgb.Booster(param)
bst.load_model("model_file_name")
preds = bst.predict(xgActualTestForSubmission)
# KEEP FINE TUNING CUTOFF - CHANGING VALUES
# KEEP RUNNING SAME TO SEE DIFFERENT VALUES
preds = [1. if y_cont > 0.28 else 0. for y_cont in preds]



# FOR SUBMISSION: saves to .csv
X_test_file['adopter'] = preds
new_file = pd.DataFrame(X_test_file[['user_id', 'adopter']], dtype = 'double')
new_file.loc[86681] = ['user_id', 'adopter']
def putfirst(n,df):
    df.index = list(range(1,n)) + [0] + list(range(n,df.index[-1]+1))
    df.sort(inplace=True)
putfirst(86682, new_file)
new_file['user_id'] = new_file['user_id'].astype(str)
new_file.iloc[1:len(new_file), 1] = new_file.iloc[1:len(new_file), 1].astype(int)
new_file.to_csv('/Users/nsilverblatt/Downloads/working1', header=False, index=False)
#reset X_test_file for future code
X_test_file = test_file.reindex(columns=['age','male','friend_cnt','avg_friend_age','avg_friend_male','friend_country_cnt','subscriber_friend_cnt','songsListened','lovedTracks','posts','playlists','shouts','delta_friend_cnt','delta_avg_friend_age','delta_avg_friend_male','delta_friend_country_cnt','delta_subscriber_friend_cnt','delta_songsListened','delta_lovedTracks','delta_posts','delta_playlists','delta_shouts','tenure','good_country','delta_good_country','user_id'])





