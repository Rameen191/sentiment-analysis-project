from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def get_bow_features(X_train, X_test):
    vectorizer = CountVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    print(f"BoW Vocabulary Size : {len(vectorizer.vocabulary_)}")
    print(f"BoW Matrix Shape    : {X_train_vec.shape}")
    print(f"Sample Vocabulary   : {list(vectorizer.vocabulary_.keys())[:10]}")
    return X_train_vec, X_test_vec, vectorizer

def get_tfidf_features(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    print(f"TF-IDF Vocabulary Size : {len(vectorizer.vocabulary_)}")
    print(f"TF-IDF Matrix Shape    : {X_train_vec.shape}")
    print(f"Sample Vocabulary      : {list(vectorizer.vocabulary_.keys())[:10]}")
    return X_train_vec, X_test_vec, vectorizer