from sklearn.model_selection import train_test_split
from common_imports import np, pd, warnings, MinMaxScaler, StandardScaler
from imblearn.under_sampling import RandomUnderSampler, InstanceHardnessThreshold, NearMiss, ClusterCentroids
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from Data_Set_Cancro import import_dataset, split_dataset
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
 
def stratificazione(X = None, y = None):
    if(X is None and y is None):
        X, y = import_dataset()
    #split con stratificazione
    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=0, test_size=0.35, stratify=y)
    return train_x, test_x, train_y, test_y
 
def bilanciamento(type = "Oversampling", tecnica="Random"):
    #import dataset
    X, y = import_dataset()
    if (type == "Undersampling"):
        if tecnica == "Random":
            rus = RandomUnderSampler(random_state=0)
            X_resampled, y_resampled = rus.fit_resample(X, y) 
        
        elif tecnica == "IHT":
            # Mappiamo le etichette da stringhe a numeri interi (M --> 1 e B--> 0)
            y = y.replace({'M': 1, 'B': 0})
            
            iht = InstanceHardnessThreshold(random_state=0)
            X_resampled, y_resampled = iht.fit_resample(X, y)
            
            # Rimappiamo le etichette numeriche a stringhe
            y_resampled = y_resampled.replace({1: 'M', 0: 'B'})
        
        elif tecnica == "NearMiss_v1":
            nm = NearMiss(version=1)
            X_resampled, y_resampled = nm.fit_resample(X, y)
        
        elif tecnica == "NearMiss_v2":
            nm = NearMiss(version=2)
            X_resampled, y_resampled = nm.fit_resample(X, y)
        
        elif tecnica == "ClusterCentroids":
            warnings.filterwarnings("ignore")
            
            cc = ClusterCentroids(random_state=0)
            X_resampled, y_resampled = cc.fit_resample(X, y)
            
            warnings.filterwarnings("default")
        
        else:
            raise ValueError("Tipo di bilanciamento non supportato: {}".format(type))
        
    elif(type == "Oversampling"):
        if(tecnica == "Random"):
            ros = RandomOverSampler(random_state=0)
            X_resampled, y_resampled = ros.fit_resample(X, y)
        
        elif(tecnica == "SMOTE"):
            sm = SMOTE(random_state=0)
            X_resampled, y_resampled = sm.fit_resample(X, y)
        
        elif(tecnica == "ADASYN"):
            ada = ADASYN(random_state=0)
            X_resampled, y_resampled = ada.fit_resample(X, y)
        
        else:
            raise ValueError("Tipo di bilanciamento non supportato: {}".format(type))
    else:
        raise ValueError("Tipo di bilanciamento non supportato: {}".format(type))
    
    #restituisco il dataset corrispondente bilanciato
    train_x, test_x, train_y, test_y = split_dataset(X_resampled, y_resampled)
    return train_x, test_x, train_y, test_y

def Undersampling_50(tecnica,X, y):
    if tecnica == "Random":
        rus = RandomUnderSampler(sampling_strategy={'B':285},random_state = 42)
        X_resampled, y_resampled = rus.fit_resample(X, y) 
        
    elif tecnica == "IHT":
        # Mappiamo le etichette da stringhe a numeri interi (M --> 1 e B--> 0)
        y = y.replace({'M': 1, 'B': 0})
            
        iht = InstanceHardnessThreshold(sampling_strategy={0:285},random_state = 42)
        X_resampled, y_resampled = iht.fit_resample(X, y)
            
        # Rimappiamo le etichette numeriche a stringhe
        y_resampled = y_resampled.replace({1: 'M', 0: 'B'})
        
    elif tecnica == "NearMiss_v1":
        nm = NearMiss(sampling_strategy={'B':285},version=1)
        X_resampled, y_resampled = nm.fit_resample(X, y)
        
    elif tecnica == "NearMiss_v2":
        nm = NearMiss(sampling_strategy={'B':285},version=2)
        X_resampled, y_resampled = nm.fit_resample(X, y)
        
    elif tecnica == "ClusterCentroids":
        warnings.filterwarnings("ignore")
            
        cc = ClusterCentroids(sampling_strategy={'B':285}, random_state = 42)
        X_resampled, y_resampled = cc.fit_resample(X, y)
            
        warnings.filterwarnings("default")
    return X_resampled, y_resampled
        
def Oversampling_50(tecnica, X, y):
    if(tecnica == "Random"):
        ros = RandomOverSampler(sampling_strategy={'M':285},random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        
    elif(tecnica == "SMOTE"):
        sm = SMOTE(sampling_strategy={'M':285},random_state=42)
        X_resampled, y_resampled = sm.fit_resample(X, y)
        
    elif(tecnica == "ADASYN"):
        ada = ADASYN(sampling_strategy={'M':285}, random_state=42)
        X_resampled, y_resampled = ada.fit_resample(X, y)
    return X_resampled, y_resampled

def unione_UnderSampling_OverSampling(model_selection):
    X, y = import_dataset()

    #NOTA BENE: tutte le combinazioni impostate di defalt sono state trovate con le funzioni nel
    # file Unione_Under_Over_sampling che prova tutte le combinazioni e trova le migliori per ogni modello! 
    if model_selection == "Albero Decisionale":
        #bilancio il dataset con esattamente il 50% di entrambe le label M e B
        X_resampled, y_resampled = Undersampling_50("IHT",X,y)
        X_resampled, y_resampled = Oversampling_50("Random",X_resampled, y_resampled)
        sampling_method = f'\nUnder: {"IHT"} Over: {"Random"}' 

        train_x, test_x, train_y, test_y = stratificazione(X_resampled, y_resampled)

    elif model_selection == "SVM":
        #bilancio il dataset con esattamente il 50% di entrambe le label M e B
        X_resampled, y_resampled = Undersampling_50("IHT",X,y)
        X_resampled, y_resampled = Oversampling_50("Random",X_resampled, y_resampled)
        sampling_method = f'\nUnder: {"IHT"} Over: {"Random"}' 

        #splitto il dataset bilanciato mantenendo la stessa distribuzione per non andare in Overfitting
        train_x, test_x, train_y, test_y = stratificazione(X_resampled, y_resampled)

    elif model_selection == "Rete Neurale":
        #bilancio il dataset con esattamente il 50% di entrambe le label M e B
        X_resampled, y_resampled = Undersampling_50("IHT",X,y)
        X_resampled, y_resampled = Oversampling_50("Random",X_resampled, y_resampled)
        sampling_method = f'\nUnder: {"IHT"} Over: {"Random"}'
        
        #splitto il dataset bilanciato mantenendo la stessa distribuzione per non andare in Overfitting
        train_x, test_x, train_y, test_y = stratificazione(X_resampled, y_resampled)

    elif model_selection == "KNN(Custom)":
        #bilancio il dataset con esattamente il 50% di entrambe le label M e B
        X_resampled, y_resampled = Undersampling_50("IHT",X,y)
        X_resampled, y_resampled = Oversampling_50("Random",X_resampled, y_resampled)
        sampling_method = f'\nUnder: {"IHT"} Over: {"Random"}' 

        #splitto il dataset bilanciato mantenendo la stessa distribuzione per non andare in Overfitting
        train_x, test_x, train_y, test_y = stratificazione(X_resampled, y_resampled)

    elif model_selection == "Random Forest(Custom)":
        #bilancio il dataset con esattamente il 50% di entrambe le label M e B
        X_resampled, y_resampled = Undersampling_50("IHT",X,y)
        X_resampled, y_resampled = Oversampling_50("SMOTE",X_resampled, y_resampled)
        sampling_method = f'\nUnder: {"IHT"} Over: {"SMOTE"}' 

        #splitto il dataset bilanciato mantenendo la stessa distribuzione per non andare in Overfitting
        train_x, test_x, train_y, test_y = stratificazione(X_resampled, y_resampled)

    #restituisco il corrispondente dataset bilanciato nel modo opportuno
    return train_x, test_x, train_y, test_y,sampling_method

def Feature_Selection(tecnica = "Correlation-Based"):
    #import dataset
    X, y = import_dataset()
    
    if tecnica == "Correlation-Based":
        # Memorizzo la matrice di correlazione
        corr_df = X.corr()

        # Seleziono gli indici che superano la soglia di correlazione
        indexes = np.where(corr_df > 0.93)

        # Ottengo gli indici delle colonne da rimuovere
        cols_to_remove = set() #senza ripetizioni
        for i, j in zip(*indexes):
            if i != j and i not in cols_to_remove and j not in cols_to_remove:
                cols_to_remove.add(j)

        # Rimuovo le colonne correlate
        cols_to_remove_list = list(cols_to_remove)
        X_reduced = X.drop(X.columns[cols_to_remove_list], axis=1)
        
    elif tecnica == "Variance Threshold":
        selector = VarianceThreshold(threshold=1)
        X_reduced = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])
 
    elif tecnica == "Select_K Best":
        
        selector = SelectKBest(mutual_info_classif, k=10)
        X_reduced = pd.DataFrame(selector.fit_transform(X, y), columns=X.columns[selector.get_support()])
       
    elif tecnica == "Sequential Feature":
        warnings.filterwarnings("ignore")

        sfs = SequentialFeatureSelector(KNeighborsClassifier(n_neighbors=1), n_features_to_select=10)
        X_reduced = pd.DataFrame(sfs.fit_transform(X, y), columns=X.columns[list(sfs.get_support(indices=True))])

        warnings.filterwarnings("default")
    else:
        raise ValueError("Tipo di Feature Selection non supportato: {}".format(type))

    #restituisco il ridotto dataset corrispondente
    train_x, test_x, train_y, test_y = split_dataset(X_reduced, y)
    return train_x, test_x, train_y, test_y

def Combinazione_Attributi(tecnica="PCA"):
    # Import dataset
    X, y = import_dataset()

    if tecnica == "PCA":
        pca = PCA(n_components=10, random_state= 0)
        X_reduced = pd.DataFrame(pca.fit_transform(X), columns=[f'PC{i+1}' for i in range(10)])

    elif tecnica == "Sparse Projection":
        srp = SparseRandomProjection(n_components=10, random_state= 0)
        X_reduced = pd.DataFrame(srp.fit_transform(X), columns=[f'SRP{i+1}' for i in range(10)])

    elif tecnica == "Gaussian Projection":
        grp = GaussianRandomProjection(n_components=10, random_state= 0)
        X_reduced = pd.DataFrame(grp.fit_transform(X), columns=[f'GRP{i+1}' for i in range(10)])

    elif tecnica == "Feature Agglomeration":
        fa = FeatureAgglomeration(n_clusters=10)
        X_reduced = pd.DataFrame(fa.fit_transform(X), columns=[f'Cluster{i+1}' for i in range(10)])

    else:
        raise ValueError("Tipo di Combinazione Attributi non supportata: {}".format(tecnica))

    # Restituisce il ridotto dataset corrispondente
    train_x, test_x, train_y, test_y = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
    return train_x, test_x, train_y, test_y

def Trasformazione_Attributi(tecnica="Standard Scaler"):
    # Import dataset
    X, y = import_dataset()

    if tecnica == "Standard Scaler":
        scaler = StandardScaler()
        X_transformed = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    elif tecnica == "MinMaxScaler":
        scaler = MinMaxScaler()
        X_transformed = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    elif tecnica == "Normalize":
        X_transformed = pd.DataFrame(normalize(X, norm='l1'), columns=X.columns)

    else:
        raise ValueError("Tipo di Trasformazione Attributi non supportata: {}".format(tecnica))

    # Suddivide il dataset trasformato
    train_x, test_x, train_y, test_y = train_test_split(X_transformed, y, test_size=0.2, random_state=0)

    # Restituisce il dataset trasformato e suddiviso
    return train_x, test_x, train_y, test_y