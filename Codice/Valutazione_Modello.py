from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay
from common_imports import accuracy_score, plt, StandardScaler, np

def grafici_valutazione(label_predict, test_y, predizioni_senza_preProcessing = None, type_model = None, test_y2 = None, preprocessing_selection = None, pre_pro_method = None):
    if(predizioni_senza_preProcessing is None):
        #plot matrice di confusione e curva roc del modello selezionato 
        dataset_senza_preprocessing(label_predict, test_y, type_model)
    else:
        #plot matrice di confusione e curva roc del modello SENZA PRE PROCESING(utile per fare un confronto) 
        #plot matrice di confusione e curva roc del modello selezionato 
        dataset_CON_preprocessing(label_predict, test_y, predizioni_senza_preProcessing, type_model, test_y2, preprocessing_selection, pre_pro_method)
 
def dataset_senza_preprocessing(label_predict, test_y, type_model, best_c = None, best_gamma = None, best_kernel = None, scaler = None, alpha_value = None, hidden_layer = None):
    
    # Crea una nuova figura con un Nome identificativo
    fig = plt.figure(num="Valutazione del Modello", figsize=(10, 4))

    # Crea una griglia di subplot con tre colonne
    axs = fig.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1]})

    matrice_confusione = confusion_matrix(test_y, label_predict, labels=["M", "B"])  
    disp = ConfusionMatrixDisplay(confusion_matrix=matrice_confusione, display_labels=["M", "B"])

    # Plotta la matrice di confusione nella prima colonna
    disp.plot(ax=axs[0])
    
    axs[0].set_title('Matrice di Confusione')

    #queste stronghe vengono stampate SOLO con il tuning del SVM ed ANN, per mostrare i migliori parametri
    if(type_model == "SVM" and best_c != None):
        fig.text(0.5, 0.75, f'Best C: {best_c}', ha='center', va='center', fontsize=11, color='black')
        fig.text(0.5, 0.70, f'Best Kernel: {best_kernel}', ha='center', va='center', fontsize=11, color='black')
        if(best_gamma != None):
            fig.text(0.5, 0.65, f'Best Gamma: {best_gamma}', ha='center', va='center', fontsize=11, color='black')

    if(type_model == "Rete Neurale" and hidden_layer != None):
        fig.text(0.5, 0.80, f'Best Layer: {hidden_layer}', ha='center', va='center', fontsize=11, color='black')
        fig.text(0.5, 0.75, f'Best Alpha: {alpha_value}', ha='center', va='center', fontsize=11, color='black')
        if(scaler is StandardScaler()): fig.text(0.5, 0.70, f'Best Scaler: Standard Scaler', ha='center', va='center', fontsize=11, color='black')
        else : fig.text(0.5, 0.70, f'Best Scaler: Min Max Scaler', ha='center', va='center', fontsize=11, color='black')

    # Converti le etichette in valori numerici (1 per "M" e 0 per "B") solo per la curva ROC
    test_y_numeric = (test_y == "M").astype(int)
    label_predict_numeric = list(map(lambda x: 1 if x == "M" else 0, label_predict))

    # Calcola la curva ROC
    fpr, tpr, thresholds = roc_curve(test_y_numeric, label_predict_numeric)
    roc_auc = auc(fpr, tpr)

    # Plotta la curva ROC nella terza colonna
    axs[2].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.3f})'.format(roc_auc))
    axs[2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axs[2].set_xlabel('False Positive Rate')
    axs[2].set_ylabel('True Positive Rate')
    axs[2].set_title('Curva ROC')
    axs[2].legend(loc="lower right",fontsize=7)

    # Rimuovi gli assi vuoti nella seconda colonna
    axs[1].axis('off')

    # Aggiusta gli assi per la curva ROC per adattarsi alla finestra
    axs[2].set_aspect('equal', adjustable='box')

    # Calcolo misure di valutazione
    accuracy = accuracy_score(test_y, label_predict)
    tn, fp, fn, tp = confusion_matrix(test_y, label_predict, labels=["M", "B"]).ravel()
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)  # Sensibilità
    F_score = (2 * recall * precision) / (recall + precision)

    nome = f'{type_model} Senza Pre-processing'

    size = 11
    # Aggiungi il testo delle metriche nel plot
    fig.text(0.5, 0.60, f'Accuratezza: {accuracy:.3f}', ha='center', va='center', fontsize= size, color='blue')
    fig.text(0.5, 0.55, f'Error Rate: {1-accuracy:.3f}', ha='center', va='center', fontsize= size, color='red')
    fig.text(0.5, 0.50, f'Sensibilità: {recall:.3f}', ha='center', va='center', fontsize= size, color='green')
    fig.text(0.5, 0.45, f'Specificità: {specificity:.3f}', ha='center', va='center', fontsize= size, color='purple')
    fig.text(0.5, 0.40, f'Precisione: {precision:.3f}', ha='center', va='center', fontsize= size, color='orange')
    fig.text(0.5, 0.35, f'F-score: {F_score:.3f}', ha='center', va='center', fontsize= size, color='cyan')
    fig.text(0.5, 0.90, nome, ha='center', va='center', fontsize= size, color='black')

    plt.tight_layout()
    plt.show()



def dataset_CON_preprocessing(label_predict, test_y, predizioni_senza_preProcessing, type_model, test_y2 = None,preprocessing_selection = None, pre_pro_method= None):
    # Crea una nuova figura con un Nome identificativo unico
    fig = plt.figure(num="Valutazione del Modello", figsize=(10, 5))

    # Crea una griglia di subplot con tre colonne
    axs = fig.subplots(2, 3, gridspec_kw={'width_ratios': [1, 1, 1]})

    #MATRICE DI CONFUSIONE SENZA PRE-PROCESSING
    matrice_confusione1 = confusion_matrix(test_y2, predizioni_senza_preProcessing, labels=["M", "B"])  
    disp1 = ConfusionMatrixDisplay(confusion_matrix=matrice_confusione1, display_labels=["M", "B"])
    # Plotta la matrice di confusione nella prima colonna
    disp1.plot(ax=axs[0][0])
    axs[0][0].set_title('Matrice di Confusione')

    #MATRICE DI CONFUSIONE CON PRE-PROCESSING
    matrice_confusione2 = confusion_matrix(test_y, label_predict, labels=["M", "B"])  
    disp2 = ConfusionMatrixDisplay(confusion_matrix=matrice_confusione2, display_labels=["M", "B"])
    # Plotta la matrice di confusione nella prima colonna
    disp2.plot(ax=axs[1][0])
    axs[1][0].set_title('Matrice di Confusione')

    # Calcolo misure di valutazione
    accuracy1 = accuracy_score(test_y2, predizioni_senza_preProcessing)
    error_rate1 = 1-accuracy1
    tn1, fp1, fn1, tp1 = confusion_matrix(test_y2, predizioni_senza_preProcessing, labels=["M", "B"]).ravel()
    specificity1 = tn1 / (tn1 + fp1)
    precision1 = tp1 / (tp1 + fp1)
    recall1 = tp1 / (tp1 + fn1)  # Sensibilità
    F_score1 = (2 * recall1 * precision1) / (recall1 + precision1)

    nome1 = f'{type_model} Senza Pre-processing'

    size = 11
    h = 0.82
    # Aggiungi il testo delle metriche nel plot
    fig.text(0.5, 0.82, f'Accuratezza: {accuracy1:.3f}', ha='center', va='center', fontsize= size, color='blue')
    fig.text(0.5, 0.77, f'Error Rate: {error_rate1:.3f}', ha='center', va='center', fontsize= size, color='red')
    fig.text(0.5, 0.72, f'Sensibilità: {recall1:.3f}', ha='center', va='center', fontsize= size, color='green')
    fig.text(0.5, 0.67, f'Specificità: {specificity1:.3f}', ha='center', va='center', fontsize= size, color='purple')
    fig.text(0.5, 0.62, f'Precisione: {precision1:.3f}', ha='center', va='center', fontsize= size, color='orange')
    fig.text(0.5, 0.57, f'F-score: {F_score1:.3f}', ha='center', va='center', fontsize= size, color='cyan')
    fig.text(0.5, 0.90, nome1, ha='center', va='center', fontsize= size, color='black')

    if(pre_pro_method is None):
        nome2 = f'{type_model} Con {preprocessing_selection}'
    else:
        if(preprocessing_selection == "Top Combo(Under/Over)"):
            nome2 = f'{type_model} Con {pre_pro_method}'
        else: nome2 = f'{type_model} Con {preprocessing_selection}: {pre_pro_method}'

    # Calcolo misure di valutazione
    accuracy2 = accuracy_score(test_y, label_predict)
    error_rate2 = 1-accuracy2
    tn2, fp2, fn2, tp2 = confusion_matrix(test_y, label_predict, labels=["M", "B"]).ravel()
    specificity2 = tn2 / (tn2 + fp2)
    precision2 = tp2 / (tp2 + fp2)
    recall2 = tp2 / (tp2 + fn2)  # Sensibilità
    F_score2 = (2 * recall2 * precision2) / (recall2 + precision2)

    size = 11
    # Aggiungi il testo delle metriche nel plot
    fig.text(0.5, 0.40, f'Accuratezza: {accuracy2:.3f}', ha='center', va='center', fontsize= size, color='blue')
    fig.text(0.5, 0.35, f'Error Rate: {error_rate2:.3f}', ha='center', va='center', fontsize= size, color='red')
    fig.text(0.5, 0.30, f'Sensibilità: {recall2:.3f}', ha='center', va='center', fontsize= size, color='green')
    fig.text(0.5, 0.25, f'Specificità: {specificity2:.3f}', ha='center', va='center', fontsize= size, color='purple')
    fig.text(0.5, 0.20, f'Precisione: {precision2:.3f}', ha='center', va='center', fontsize= size, color='orange')
    fig.text(0.5, 0.15, f'F-score: {F_score2:.3f}', ha='center', va='center', fontsize= size, color='cyan')
    fig.text(0.5, 0.50, nome2, ha='center', va='center', fontsize= size, color='black')


    
    # Calcola la curva ROC Senza pre-processing
    # Converti le etichette in valori numerici (1 per "M" e 0 per "B") solo per la curva ROC
    test_y_numeric1 = (test_y2 == "M").astype(int)
    label_predict_numeric1 = list(map(lambda x: 1 if x == "M" else 0, predizioni_senza_preProcessing))

    fpr1, tpr1, thresholds = roc_curve(test_y_numeric1, label_predict_numeric1)
    roc_auc1 = auc(fpr1, tpr1)

    # Plotta la curva ROC nella terza colonna
    axs[0][2].plot(fpr1, tpr1, color='darkorange', lw=2, label='ROC curve (area = {:.3f})'.format(roc_auc1))
    axs[0][2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axs[0][2].set_xlabel('False Positive Rate')
    axs[0][2].set_ylabel('True Positive Rate')
    axs[0][2].set_title('Curva ROC')
    axs[0][2].legend(loc="lower right",fontsize=7)

    # Converti le etichette in valori numerici (1 per "M" e 0 per "B") solo per la curva ROC
    test_y_numeric2 = (test_y == "M").astype(int)
    label_predict_numeric2 = list(map(lambda x: 1 if x == "M" else 0, label_predict))

    # Calcola la curva ROC CON pre-processing
    fpr2, tpr2, thresholds = roc_curve(test_y_numeric2, label_predict_numeric2)
    roc_auc2 = auc(fpr2, tpr2)

    # Plotta la curva ROC nella terza colonna
    axs[1][2].plot(fpr2, tpr2, color='darkorange', lw=2, label='ROC curve (area = {:.3f})'.format(roc_auc2))
    axs[1][2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axs[1][2].set_xlabel('False Positive Rate')
    axs[1][2].set_ylabel('True Positive Rate')
    axs[1][2].set_title('Curva ROC')
    axs[1][2].legend(loc="lower right",fontsize=7)

    # Rimuovi gli assi vuoti nella seconda colonna
    axs[0][1].axis('off')
    axs[1][1].axis('off')

    # Aggiusta gli assi per la curva ROC per adattarsi alla finestra
    axs[0][2].set_aspect('equal', adjustable='box')
    axs[1][2].set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()


def plot_tuning_knn(acc_train, acc_val, acc_test, dist_range):
    """
    Visualizza i risultati del tuning del KNN, inclusi gli andamenti
    delle accuratezze per diverse distanze e le migliori configurazioni.
    """
    # Crea una nuova figura con un nome identificativo unico e dimensioni specificate
    fig = plt.figure(num="Tuning KNN Custom", figsize=(10, 5))

    distance = ["Euclidea", "Manhattan", "Chebyshev"]

    # Crea una griglia di subplot con tre colonne
    axs = fig.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})

    k_range = range(1, 26)
    
    # Plot delle curve di accuratezza per le diverse distanze
    for i, dist in enumerate(dist_range):
        axs[0].plot(k_range, acc_train[i,:], lw=2, label=[dist + ' on train'])
        axs[0].plot(k_range, acc_val[i,:], lw=2, label=[dist + ' on val'])
        axs[0].plot(k_range, acc_test[i,:], lw=2, label=[dist + ' on test'])

    # Personalizza l'aspetto del grafico principale
    axs[0].set_xlim([1, max(k_range)])
    axs[0].grid(True, axis='both', zorder=0, linestyle='solid', color='k')
    axs[0].tick_params(labelsize=13)
    axs[0].set_xlabel('k_range', fontsize=15)
    axs[0].set_ylabel('Accuracy', fontsize=15)
    axs[0].set_title('Model Performance', fontsize=15)
    axs[0].legend()

    # Trova l'accuratezza massima per ciascuna distanza NEL VALIDATION TEST
    max_acc_Euclidea = max(acc_val[0])
    max_acc_Manhattan = max(acc_val[1])
    max_acc_Chebyshev = max(acc_val[2])

    # Salva le accuratezze massime in una lista
    all_acc = [max_acc_Euclidea, max_acc_Manhattan, max_acc_Chebyshev]

    # Trova la profondità associata all'accuratezza massima per ciascuna distanza
    best_k_Euclidea = k_range[np.argmax(acc_val[0])]
    best_k_Manhattan = k_range[np.argmax(acc_val[1])]
    best_k_Chebyshev = k_range[np.argmax(acc_val[2])]


    # Salva le migliori profondità in una lista
    all_k_best = [best_k_Euclidea, best_k_Manhattan, best_k_Chebyshev]

    # Trova l'indice della migliore accuratezza
    index_of_best_accuracy = all_acc.index(max(all_acc))

    # Ottieni i valori migliori per l'accuratezza, la profondità e la distanza
    max_acc = all_acc[index_of_best_accuracy]
    best_k = all_k_best[index_of_best_accuracy]
    best_distance = distance[index_of_best_accuracy]

    # Aggiungi testo sotto il secondo grafico per mostrare l'accuratezza massima, la profondità migliore e la distanza migliore
    axs[1].text(0.6, 0.55, f'Max Accuracy: {max_acc:.3f}', ha='center', va='center', fontsize=17, color='g')
    axs[1].text(0.6, 0.45, f'Best K: {best_k}', ha='center', va='center', fontsize=17, color='r')
    axs[1].text(0.6, 0.35, f'Best distance: {best_distance}', ha='center', va='center', fontsize=17, color='r')

    # Nascondi gli assi nel secondo subplot
    axs[1].axis('off')

    # Mostra il grafico
    plt.show()


def plot_tuning_AlberoDecisionale(acc_train, acc_val, acc_test, max_depth_range):
    # Creazione della figura
    fig = plt.figure(num="Tuning Albero Decisionale", figsize=(8, 5))

    # Creazione di un sistema di assi con due subplot
    axs = fig.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})

    # Plot delle accuratezze per allenamento, validazione e test in funzione di max_depth
    axs[0].plot(max_depth_range, acc_train, lw=2, color='r')
    axs[0].plot(max_depth_range, acc_val, lw=2, color='b')
    axs[0].plot(max_depth_range, acc_test, lw=2, color='g')

    # Configurazione dell'asse principale
    axs[0].set_xlim([1, max(max_depth_range)])
    axs[0].grid(True, axis='both', zorder=0, linestyle='solid', color='k')
    axs[0].tick_params(labelsize=13)
    axs[0].set_xlabel('H Depth', fontsize=15)
    axs[0].set_ylabel('Accuracy', fontsize=15)
    axs[0].set_title('Model Performance', fontsize=15)

    # Identificazione della profondità ottimale
    best_depth_index = np.argmax(acc_val)
    best_depth = max_depth_range[best_depth_index]
    best_score = max(acc_val)

    # Aggiunta della legenda al grafico principale
    axs[0].legend(['Train', 'Val', 'Test'], loc="lower right")

    # Formatta la stringa con un numero fisso di cifre decimali
    best_score = "{:.3f}".format(best_score)

    # Testo nel secondo subplot con informazioni sulla profondità ottimale e sull' acuratezza ottimale
    axs[1].text(0.6, 0.55, f'Best Val Score: {best_score}', ha='center', va='center', fontsize=15, color='b')
    axs[1].text(0.6, 0.45, f'Best Val Depth: {best_depth}', ha='center', va='center', fontsize=15, color='b')

    # Disabilitazione dell'asse nel secondo subplot
    axs[1].axis('off')

    # Visualizzazione del grafico
    plt.show()

def plot_tuning_Random_forest(all_score_Random_Forest, max_num_classifier, all_numbers_trees_forest):
    """
    Visualizza i risultati del tuning del modello Random Forest.

    Parameters:
    - all_score_Random_Forest (list): Lista delle performance del modello per diversi numeri di alberi.
    - max_num_classifier (int): Numero massimo di alberi nel forest considerati nel tuning.
    - all_numbers_trees_forest (list): Lista di tutti i numeri di alberi considerati nel tuning.
    """
    # Creazione della figura
    fig = plt.figure(num="Tuning Random Forest", figsize=(8, 5))

    # Creazione di un sistema di assi con due subplot
    axs = fig.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})

    # Plot delle performance del modello in funzione del numero di alberi
    axs[0].plot(all_numbers_trees_forest, all_score_Random_Forest, lw=2, color='g')

    # Personalizzazione dell'aspetto del subplot principale
    axs[0].set_xlim([1, max(all_numbers_trees_forest)])
    axs[0].grid(True, axis='both', zorder=0, linestyle='solid', color='k')
    axs[0].tick_params(labelsize=13)
    axs[0].set_xlabel('Number of Classifiers', fontsize=15)
    axs[0].set_ylabel('Accuracy', fontsize=15)
    axs[0].set_title('Model Performance', fontsize=15)

    # Trova l'indice del miglior numero di alberi e il miglior punteggio di accuratezza
    best_numberTrees_index = np.argmax(all_score_Random_Forest)
    best_number_trees = all_numbers_trees_forest[best_numberTrees_index]
    best_score = max(all_score_Random_Forest)

    # Aggiungi una legenda al subplot principale
    axs[0].legend(['Random Forest'], loc="lower right")

    # Formatta la stringa con un numero fisso di cifre decimali
    best_score = "{:.3f}".format(best_score)

    # Aggiungi testi sotto il secondo subplot per mostrare il miglior punteggio e il numero di alberi migliori
    axs[1].text(0.7, 0.55, f'Best Score: {best_score}', ha='center', va='center', fontsize=15, color='g')
    axs[1].text(0.7, 0.45, f'Best Number Trees: {best_number_trees}', ha='center', va='center', fontsize=15, color='g')

    # Nascondi gli assi nel secondo subplot
    axs[1].axis('off')

    # Visualizzazione del grafico
    plt.show()


def plot_roc_migliori_combinazioni(esiti_cancro_albero,test_y_albero, esiti_cancro_svm, test_y_svm, esiti_cancro_ann, test_y_ann, esiti_cancro_knn, test_y_knn, esiti_cancro_rf, test_y_rf):

    # Converti le etichette in valori numerici (1 per "M" e 0 per "B")
    esiti_cancro_albero = list(map(lambda x: 1 if x == "M" else 0, esiti_cancro_albero))
    test_y_albero = list(map(lambda x: 1 if x == "M" else 0, test_y_albero))

    esiti_cancro_svm = list(map(lambda x: 1 if x == "M" else 0, esiti_cancro_svm))
    test_y_svm = list(map(lambda x: 1 if x == "M" else 0, test_y_svm))

    esiti_cancro_ann = list(map(lambda x: 1 if x == "M" else 0, esiti_cancro_ann))
    test_y_ann = list(map(lambda x: 1 if x == "M" else 0, test_y_ann))

    esiti_cancro_knn = list(map(lambda x: 1 if x == "M" else 0, esiti_cancro_knn))
    test_y_knn = list(map(lambda x: 1 if x == "M" else 0, test_y_knn))

    esiti_cancro_rf = list(map(lambda x: 1 if x == "M" else 0, esiti_cancro_rf))
    test_y_rf = list(map(lambda x: 1 if x == "M" else 0, test_y_rf))
    


    # Calcola le curve ROC Albero Decisionale
    fpr_albero, tpr_albero, _ = roc_curve(test_y_albero, esiti_cancro_albero)
    roc_auc_albero = auc(fpr_albero, tpr_albero)

    # Calcola le curve ROC SVM
    fpr_svm, tpr_svm, _ = roc_curve(test_y_svm, esiti_cancro_svm)
    roc_auc_svm = auc(fpr_svm, tpr_svm)

    # Calcola le curve ROC ANN
    fpr_ann, tpr_ann, _ = roc_curve(test_y_ann, esiti_cancro_ann)
    roc_auc_ann = auc(fpr_ann, tpr_ann)

    # Calcola le curve ROC KNN
    fpr_knn, tpr_knn, _ = roc_curve(test_y_knn, esiti_cancro_knn)
    roc_auc_knn = auc(fpr_knn, tpr_knn)

    # Calcola le curve ROC Random Forest
    fpr_rf, tpr_rf, _ = roc_curve(test_y_rf, esiti_cancro_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)

    # Crea il plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_albero, tpr_albero, lw=2, label=f'Curva ROC DT (AUC = {roc_auc_albero:.2f})')
    plt.plot(fpr_svm, tpr_svm, lw=2, label=f'Curva ROC SVM (AUC = {roc_auc_svm:.2f})')
    plt.plot(fpr_ann, tpr_ann, lw=2, label=f'Curva ROC ANN (AUC = {roc_auc_ann:.2f})')
    plt.plot(fpr_knn, tpr_knn, lw=2, label=f'Curva ROC KNN (AUC = {roc_auc_knn:.2f})')
    plt.plot(fpr_rf, tpr_rf, lw=2, label=f'Curva ROC RF (AUC = {roc_auc_rf:.2f})')

    # Aggiungi le etichette e la legenda
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curve ROC')
    plt.legend(loc='lower right')

    # Mostra il plot
    plt.show()