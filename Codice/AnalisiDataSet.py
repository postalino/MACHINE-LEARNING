from common_imports import os, pd, plt, import_dataset, np
from io import StringIO
import seaborn as sns

#librerie per app EXE
#import pkg_resources

def informazioni_DataSet():
    # ottiene il percorso file del dataset
    file_path = os.path.join(os.path.dirname(__file__), "data.csv")

    #questa riga serve SOLO per acquisire il csv all'exe in fase di building
    #file_path = pkg_resources.resource_stream(__name__, 'data.csv')

    df = pd.read_csv(file_path, encoding='UTF-8')
    
    # Utilizzare StringIO per catturare l'output di df.info()
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    # Rimuovere le parti indesiderate dalla stringa
    lines = info_str.split('\n')
    start_index = lines.index("Data columns (total 33 columns):") + 1
    end_index = lines.index("dtypes: float64(31), int64(1), object(1)")
    column_info = "\n".join(lines[start_index:end_index])

    # Creare una stringa contenente tutte le informazioni
    result_str = "Informazioni generali:\n" + column_info + "\n\n"
    result_str += "Prime righe del DataFrame:\n" + str(df.head()) + "\n\n"
    result_str += "Ultime righe del DataFrame:\n" + str(df.tail()) + "\n\n"

    return result_str

def BoxPlot(Feature):
    X, y = import_dataset()

    # Utilizza seaborn per un aspetto più attraente
    sns.set(style="whitegrid")

    # Crea il box plot per l'attributo in base alle etichette
    plt.figure(figsize=(7, 5))
    sns.boxplot(x=y, y=X[Feature])

    # Aggiungi etichette e titolo
    plt.xlabel('Categoria')
    plt.ylabel('Valore')
    plt.title(f'Box plot di {Feature} per le categorie M e B')
    # Mostra il grafico
    plt.show()


# Anche se non richiamata esplicitamente è stata usata questa funzione per generare l'immagine 
def Matrice_Correlazione():
    # Importa il dataset, considerando solo le variabili indipendenti X
    X, _ = import_dataset()

    # Crea una nuova figura per la matrice di correlazione
    plt.figure(num='Matrice di Correlazione', figsize=(20, 20))

    # Aggiunge il titolo alla figura
    plt.title('Matrice di correlazione')

    # Calcola la matrice di correlazione e arrotondala a 2 decimali
    corr_map = X.corr().round(4)

    # Utilizza Seaborn per visualizzare la matrice di correlazione come una heatmap
    sns.heatmap(corr_map, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=plt.gca())
    
    # Aggiungi un po' di spazio alla parte superiore e inferiore della heatmap
    plt.subplots_adjust(top=0.92, bottom=0.15)

    # Mostra la figura
    plt.show()


def statistiche_DataSet():
    # ottiene il percorso file del dataset
    file_path = os.path.join(os.path.dirname(__file__), "data.csv")

    #questa riga serve SOLO per acquisire il csv all'exe in fase di building
    #file_path = pkg_resources.resource_stream(__name__, 'data.csv')

    df = pd.read_csv(file_path, encoding='UTF-8')

    # Rimuovere colonne non necessarie
    y = df["diagnosis"]
    df = df.drop(['id', 'Unnamed: 32','diagnosis'], axis=1)

    # Creare una stringa contenente tutte le informazioni
    # Shape del DataFrame
    result_str = "\n\nDimensioni: {}\n\n".format(df.shape)

    #occorrenze label dataset
    result_str += "\n\nOccorrenze label di classe: \n{}\n\n".format(y.value_counts())

    # Valori mancanti
    result_str += "\n\nValori mancanti:\n{}\n\n".format(df.isnull().sum())

    # Valori duplicati
    result_str += "\n\nValori duplicati:{}\n\n".format(df.duplicated().sum())

    # Stampa di minimi, massimi, media e deviazione standard
    result_str += "\n\nValori medi:\n {}\n\nDeviazione standard:\n{}\n\nMassimo:\n{}\n\nMinimo:\n{}\n\n1° quartile:\n{}\n\n2° quartile(moda):\n{}\n\n3° quartile:\n{}\n\n".format(
        df.describe().loc['mean'], df.describe().loc['std'], df.describe().loc['max'], df.describe().loc['min'],
        df.describe().loc['25%'], df.describe().loc['50%'], df.describe().loc['75%']
    )

    # Varianza
    result_str += "\n\nVarianza:\n{}\n".format(np.var(df, axis=0))

    return result_str
 