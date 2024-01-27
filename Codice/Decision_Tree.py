from sklearn.tree import DecisionTreeClassifier

def decision_tree(train_x, train_y, test_x, depth=6):
    # Creazione di un classificatore ad albero decisionale con profondità massima specificata(la migliore per questo dataset)
    dTree_clf = DecisionTreeClassifier(max_depth=depth, random_state=0)

    # Addestramento del classificatore utilizzando i dati di addestramento
    dTree_clf.fit(train_x, train_y)

    # Predizione delle etichette di classe per i dati di test
    y_pred = dTree_clf.predict(test_x)

    # Restituzione delle etichette predette
    return y_pred
 