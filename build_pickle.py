import cPickle as pickle
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

from sklearn.feature_extraction import text
from numpy.linalg import lstsq
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB



def get_data():
	df = pd.read_json('data.json')

	#Create binary fraud column
	df['fraud'] = 0
	df.loc[df['acct_type'] == 'fraudster_event', 'fraud'] = 1
	df.loc[df['acct_type'] == 'fraudster', 'fraud'] = 1
	df.loc[df['acct_type'] == 'fraudster_att', 'fraud'] = 1
	#Check
	# print "Should be 1293, it is... "+ sum(df['fraud'])

	#Train, test, split
	y = df['fraud']
	X = df[[         u'approx_payout_date',        u'body_length',
		     u'channels',            u'country',           u'currency',
	      u'delivery_method',        u'description',       u'email_domain',
		u'event_created',          u'event_end',    u'event_published',
		  u'event_start',       u'fb_published',                u'gts',
		u'has_analytics',         u'has_header',           u'has_logo',
		       u'listed',               u'name',        u'name_length',
		    u'num_order',        u'num_payouts',          u'object_id',
		     u'org_desc',       u'org_facebook',           u'org_name',
		  u'org_twitter',         u'payee_name',        u'payout_type',
	     u'previous_payouts',      u'sale_duration',     u'sale_duration2',
		     u'show_map',       u'ticket_types',           u'user_age',
		 u'user_created',          u'user_type',      u'venue_address',
		u'venue_country',     u'venue_latitude',    u'venue_longitude',
		   u'venue_name',        u'venue_state',              ]]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1234)
	return X_train, X_test, y_train, y_test


def build_pickle(df):
    #CLEAN HTML FUNCTION
    def get_text(cell):
        return BeautifulSoup(cell, 'html.parser').get_text()

    #Parse descriptions using html function above:
    df['description'] = df['description'].apply(get_text)
    clean = df['description']

    #All the parameters for the topic modeling.
    n_samples = len(clean)
    n_features = 500
    n_topics = 9
    n_top_words = 30

    my_additional_stopwords = ["la", "et", "en", "le", "les", "des", 'january', 'february',
                           'march', 'april', 'may', 'june', 'july', 'august', 'september',
                           'october', 'november', 'december', 'friday', 'thursday', 'saturday']
    stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stopwords)


    # Use tf-idf features for NMF.
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       max_features=n_features,
                                       stop_words=stop_words)
    tfidf = tfidf_vectorizer.fit_transform(clean)

    #Assign topics to existing
    topic_dict = {0:'topic1', 1:'topic2', 2:'topic3', 3:'topic4', 4: 'topic5', 5:'topic6',
                  6:'topic7', 7:'topic8', 8:'topic9'}

    nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)
    W = nmf.transform(tfidf)
    df['topic_index'] = np.argmax(W, axis=1)
    df['topic_index'] = df['topic_index'].replace(topic_dict)

    ###Create dummy variables to insert into model
    topic_dummies = pd.get_dummies(df['topic_index']).rename(columns = lambda x: 'topic_'+str(x))
    print topic_dummies.shape
    df = pd.concat([df,topic_dummies],axis=1)

    #Different model
    multinom = MultinomialNB()
    target = df['topic_index']
    model = multinom.fit(tfidf, target)


    return tfidf_vectorizer, model


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()
    tfidf_vectorizer, model = build_pickle(X_train)
    with open('vectorizer3.pkl', 'w') as f:
        pickle.dump(tfidf_vectorizer, f)
    with open('model3.pkl', 'w') as f:
        pickle.dump(model, f)
    # with open('data/model.pkl', 'w') as f:
    #     pickle.dump(model, f)
