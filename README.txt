Benvenuti nel progetto di Machine Learning sviluppato da me, Davide Senette, 
come lavoro finale per l'esame di Machine Learning presso l'Università di Cagliari, 
Corso di Laurea in Informatica Applicata e Data Analytics(IADA).

Descrizione del Progetto:
Questo progetto rappresenta un componente fondamentale del corso di Machine Learning 2023-2024 
e offre l'opportunità di applicare le conoscenze acquisite attraverso un'implementazione pratica. 
L'obiettivo principale è quello di sviluppare un'applicazione che dimostri l'utilizzo 
delle tecniche di Machine Learning sul dataset Breast Cancer Wisconsin (Diagnostic), 
mostrando l'efficacia di vari algoritmi e strategie di analisi.

Contenuto del Repository:
All'interno della cartella "codice" troverete l'intero corpus del codice Python relativo al progetto. 
Questo include script e moduli che illustrano il processo di sviluppo e implementazione.

Note finali:
è possibile creare un file .exe dell'intero progetto usando da terminale il seguente comando

    pyinstaller --onefile --windowed --add-data "percorso file data.csv;." --add-data "percorso file Immagine progetto.jpg;." 
    --add-data "percorso file Matrice_correlazione.png;." --icon="percorso file logo.ico" main.py

Il file .exe risultante sarà completamente autonomo rispetto al codice sorgente etc.
