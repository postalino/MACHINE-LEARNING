# Import delle librerie necessarie
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier

# Definizione della funzione per la rete neurale artificiale
def Artificial_Neural_Network(train_x, train_y, test_x, hidden_layer=(30), scaler=MinMaxScaler(), alpha_value=0.001): 
    # Lista di scaler disponibili
    #si usa di default il migliore per questo dataset
    lista_scaler = [StandardScaler(), MinMaxScaler()]

    # Creazione di un classificatore MLP (Multi-Layer Perceptron)
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer,
        max_iter=100,
        alpha=alpha_value,
        solver="sgd",
        verbose=10,
        random_state=0,
        learning_rate_init=0.2,
        early_stopping=True
    )

    # Seleziona lo scaler specificato come parametro
    type_scaler = scaler
    
    # Trasforma i dati di addestramento utilizzando lo scaler
    scld_train_x = type_scaler.fit_transform(train_x)
    
    # Addestra il modello sulla base dei dati di addestramento trasformati
    mlp.fit(scld_train_x, train_y)

    # Trasforma i dati di test utilizzando lo stesso scaler
    scld_test_x = type_scaler.transform(test_x)

    # Ottieni le etichette predette
    predicted_labels = mlp.predict(scld_test_x)

    # Restituisci le etichette predette
    return predicted_labels
 