# Importa le classi StandardScaler e SVC dal modulo common_imports
from common_imports import StandardScaler, SVC

# Definisce una funzione SVM_classifier che prende in input dati di addestramento, dati di test e un dizionario di parametri ottimali (opzionale)
def SVM_classifier(train_x, train_y, test_x, best_param=None):
    # Crea un oggetto StandardScaler per normalizzare i dati
    scaler = StandardScaler()
    
    # Adatta lo scaler ai dati di addestramento
    scaler.fit(train_x)
    
    # Verifica se sono forniti parametri ottimali
    if best_param is not None:
        # Controlla se il kernel è lineare
        if best_param.get('kernel') == 'linear':
            # Utilizza i parametri ottimali se forniti nel dizionario, altrimenti utilizza i valori di default
            svm_clf = SVC(C=best_param.get('C'), kernel=best_param.get('kernel'))
        else:
            # Utilizza i parametri ottimali se forniti nel dizionario, altrimenti utilizza i valori di default
            svm_clf = SVC(C=best_param.get('C'), kernel=best_param.get('kernel'), gamma=best_param.get('gamma'))
    else:
        # Parametri di default (i migliori per questo dataset)
        svm_clf = SVC(C=0.1, kernel='linear')
    
    # Addestra il classificatore SVM sui dati di addestramento normalizzati
    svm_clf.fit(scaler.transform(train_x), train_y)
    
    # Effettua previsioni sui dati di test normalizzati
    pred_y = svm_clf.predict(scaler.transform(test_x))
    
    # Restituisci le previsioni
    return pred_y
 