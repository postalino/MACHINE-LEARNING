from sklearn.model_selection import train_test_split
from common_imports import os, pd

#librerie per app EXE
#import pkg_resources

def import_dataset():
    # ottiene il percorso file del dataset
    file_path = os.path.join(os.path.dirname(__file__), "data.csv")
    
    #questa riga serve SOLO per acquisire il csv all'exe in fase di building
    #file_path = pkg_resources.resource_stream(__name__, 'data.csv')

    df = pd.read_csv(file_path, encoding='UTF-8')
    
    #seleziono l'etichetta di classe
    Y = df["diagnosis"]

    # Rimuovi colonne non necessarie
    df = df.drop(['id', 'Unnamed: 32', 'diagnosis'], axis=1)

    #raccolgo i nomi di tutte le colonne
    attributi = df.columns[:]
    
    #seleziono quelle di mio interesse tra gli attributi
    X = df[attributi[:]]
    
    return X, Y

def split_dataset(X = None, y = None):
    
    if(X is None and y is None):
        X, y = import_dataset()
    
    # Esegui lo split in training e test set
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.35, random_state=0)
    return train_x, test_x, train_y, test_y
     