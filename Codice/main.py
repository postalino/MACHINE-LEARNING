#librerie menu'
from common_imports import os
import tkinter as tk
from tkinter import messagebox
from tkinter.constants import HORIZONTAL, VERTICAL, RIGHT, BOTTOM, ALL
from PIL import Image, ImageTk
from tkinter.scrolledtext import ScrolledText,Scrollbar
  
#libreria modelli
from common_imports import split_dataset
from Pre_Processing import stratificazione, bilanciamento, Feature_Selection, Combinazione_Attributi, Trasformazione_Attributi,unione_UnderSampling_OverSampling
from AnalisiDataSet import informazioni_DataSet, BoxPlot,statistiche_DataSet
from Decision_Tree import decision_tree
from Multi_Classifier import RandomForestCustom
from Knn import KNN_Classifier_Custom
from SVM import SVM_classifier
from ANN import Artificial_Neural_Network
from Tuning_modelli import tuning_Albero_decisionale, tuning_KNN_custom, tuning_SVM, tuning_ANN, tuning_Random_Forest
from Valutazione_Modello import grafici_valutazione
from Miglior_combinazione import best_combination, confronta_le_migliori_in_assoluto, confronto_migliori_campionamento,\
    confronto_migliori_bilanciamento,confronto_migliori_Feature_Selection, confronto_migliori_Combinazione_Attributi, \
    confronto_migliori_Trasformazione_Attributi, confronto_Senza_Pre_processing

#librerie per app EXE
#import pkg_resources

class MenuApp:
    def __init__(self, master):
        # Inizializzazione dell'interfaccia grafica
        self.master = master
        master.title("Progetto ML")
        master.geometry("590x300")
        master.resizable(False, False)

        # Frame per la parte dei bottoni
        self.button_frame = tk.Frame(master)
        self.button_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Frame per la parte dell'immagine con la linea di separazione
        self.image_frame = tk.Frame(master)
        self.image_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # Ottenimento del percorso completo dell'immagine
        image_path = os.path.join(os.path.dirname(__file__), "Immagine progetto.jpg")
        
        #questa riga serve SOLO per acquisire l'immagine all'exe in fase di building
        #image_path = pkg_resources.resource_stream(__name__, 'Immagine progetto.jpg')
        
        # Immagine input
        original_image = Image.open(image_path)

        # Ridimensiona l'immagine
        resized_image = original_image.resize((360, 300))
        self.photo = ImageTk.PhotoImage(resized_image)

        # Crea un'etichetta con l'immagine ridimensionata
        self.image_label = tk.Label(self.image_frame, image=self.photo)
        self.image_label.pack()

        # Crea un frame separator
        separator = tk.Frame(self.image_frame, width=2, bd=1, relief=tk.SUNKEN)
        separator.pack(fill=tk.Y, padx=5, pady=5)

        # Inizializza il frame dei bottoni principali
        self.initialize_main_menu_buttons()

    #           MENU PRINCIPALE
    def initialize_main_menu_buttons(self):
        distance = 5
        # Bottoni per le diverse operazioni principali
        self.data_Dati_Da_Analizzare_button = tk.Button(self.button_frame, text="Dati Da Analizzare", command=self.show_data_analysis, width=20, anchor="center")
        self.data_Dati_Da_Analizzare_button.pack(pady=distance)  # Aggiunto pady per spaziatura verticale

        self.modelli_button = tk.Button(self.button_frame, text="Addestra Modelli", command=self.show_modelli_button, width=20, anchor="center")
        self.modelli_button.pack(pady=distance)  # Aggiunto pady per spaziatura verticale

        '''self.Tuning_button = tk.Button(self.button_frame, text="Tuning Modelli", command=self.Tuning_modelli, width=20, anchor="center")
        self.Tuning_button.pack(pady=distance)  # Aggiunto pady per spaziatura verticale'''

        self.miglior_combinazione_button = tk.Button(self.button_frame, text="Miglior Combinazione", command=self.show_miglior_combinazione, width=20, anchor="center")
        self.miglior_combinazione_button.pack(pady=distance)  # Aggiunto pady per spaziatura verticale

        # Pulsante Exit per chiudere l'app
        self.exit_button = tk.Button(self.button_frame, text="Esci", command=self.exit_app, width=20, anchor="center")
        self.exit_button.pack(pady=distance)  # Aggiunto pady per spaziatura verticale

    #           PULIZIA
    def clear_buttons(self):
        # Ripulisce tutti i widget nel frame dei bottoni
        for widget in self.button_frame.winfo_children():
            widget.destroy()
    
    # Funzione per chiudere l'app
    def exit_app(self):
        self.master.destroy()

    #           QUIT MENU PRINCIPALE
    def show_main_menu(self):
        # Ripulisce i bottoni e mostra il menu principale
        self.clear_buttons()
        self.initialize_main_menu_buttons()
    
    #           DATI DA ANALIZZARE
    def show_data_analysis(self):
        # Ripulisce i bottoni e mostra il sotto-menu
        self.clear_buttons()
        self.initialize_data_analysis_buttons()
    
    def initialize_data_analysis_buttons(self):
        distance = 5 #distanza tra i bottoni
        button_width = 22  # Larghezza fissa per tutti i bottoni 

        # Bottone per ottenere informazioni sul dataset
        self.Informazioni_button = tk.Button(self.button_frame, text="Informazioni Dataset", command=self.Informazioni, width= button_width+1, anchor="center")
        self.Informazioni_button.pack(pady=distance)  # Aggiunto pady per spaziatura verticale

        # Etichetta per il testo sopra il pulsante delle features
        model_label = tk.Label(self.button_frame, text="Scegli Features", anchor="center", font=("Helvetica", 10))
        model_label.pack(pady= distance)

        # Dropdown delle feauters
        Feature_options = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
        
        # Variabile per la selezione della feature
        self.Feature_var = tk.StringVar(self.master)
        self.Feature_var.set(Feature_options[0])  # Imposta il valore predefinito
        self.Feature_dropdown = tk.OptionMenu(self.button_frame, self.Feature_var, *Feature_options)
        self.Feature_dropdown.config(width=button_width)
        self.Feature_dropdown.pack(pady=0)

        # Bottone per visualizzare il Box-Plot
        self.Informazioni_button = tk.Button(self.button_frame, text="Visualizza Box-Plot", command=self.visualizza_box_plot, width=button_width+1, anchor="center")
        self.Informazioni_button.pack(pady=0)  # Aggiunto pady per spaziatura verticale

        # Bottone per visualizzare la Matrice di Correlazione
        self.Matrice_button = tk.Button(self.button_frame, text="Matrice Di Correlazione", command=self.visualizza_matrice_correlazione, width=button_width + 1, anchor="center")
        self.Matrice_button.pack(pady=distance * 3)  # Aggiunto pady per spaziatura verticale

        # Bottone per visualizzare le Statistiche del Dataset
        self.Statistiche_button = tk.Button(self.button_frame, text="Statistiche Dataset", command=self.visualizza_statistiche, width=button_width + 1, anchor="center")
        self.Statistiche_button.pack(pady=distance)  # Aggiunto pady per spaziatura verticale

        # Pulsante Quit per tornare al menu principale
        self.quit_button = tk.Button(self.button_frame, text="<--", command=self.show_main_menu, width=button_width + 1, anchor="center")
        self.quit_button.pack(pady=distance)  # Aggiunto pady per spaziatura verticale
    
    def Informazioni(self):
        # Chiama la funzione per creare la finestra con ScrolledText
        text_widget = self.finestra_scrolled_text(self.master, "Informazioni DataFrame", 800, 590)

        # Chiama la funzione informazioni_DataSet e inserisce le informazioni nel widget di testo
        result_string = informazioni_DataSet()
        text_widget.insert(tk.END, result_string)
    
    def visualizza_box_plot(self):
        #cattura scelta dell'utente
        Feature_selection = self.Feature_var.get()
        #mostra il box plot della feauter selezionata
        BoxPlot(Feature_selection)

    def visualizza_matrice_correlazione(self):
        # Ottenimento del percorso completo dell'immagine
        image_path = os.path.join(os.path.dirname(__file__), "Matrice_correlazione.png")

        #questa riga serve SOLO per acquisire l'immagine all'exe in fase di building
        #image_path = pkg_resources.resource_stream(__name__, 'Matrice_correlazione.png')

        # Crea la finestra per lo scorrimento dell'immagine
        self.Finestra_Immagine(self.master,"Matrice Correlazione", image_path, 800, 590)

    def visualizza_statistiche(self):
        # Chiama la funzione per creare la finestra con ScrolledText
        text_widget = self.finestra_scrolled_text(self.master, "Statistiche Dataset", 800, 590)

        # Chiama la funzione statistiche_DataSet e inserisce le informazioni nel widget di testo
        result_string = statistiche_DataSet()
        text_widget.insert(tk.END, result_string)

    #           MODELLI
    def show_modelli_button(self):
        # Ripulisce i bottoni e mostra il sotto-menu
        self.clear_buttons()
        self.initialize_modelli_buttons()
        
    def initialize_modelli_buttons(self):
        distance = 5
        button_width = 20  # Larghezza fissa per tutti i bottoni

        # Label per il testo sopra i pulsanti
        preprocessing_label = tk.Label(self.button_frame, text="Pre-Processing", anchor="center", font=("Helvetica", 10))
        preprocessing_label.pack(pady= 0)

       
        # Dropdown per il preprocessing
        preprocessing_options = ["Nessuno", "Campionamento", "Undersampling", "Oversampling","Top Combo(Under/Over)","Feature Selection", "Combinazione Attributi", "Trasformazione Attributi"]
        self.preprocessing_var = tk.StringVar(self.master)
        self.preprocessing_var.set(preprocessing_options[0])  # Imposta il valore predefinito
        self.preprocessing_dropdown = tk.OptionMenu(self.button_frame, self.preprocessing_var, *preprocessing_options, command=self.update_preprocessing_options)
        self.preprocessing_dropdown.config(width=button_width)
        self.preprocessing_dropdown.pack(pady=distance)
            
        # Label per il testo sopra il pulsante del modello
        model_label = tk.Label(self.button_frame, text="Modello", anchor="center", font=("Helvetica", 10))
        model_label.pack(pady= 0)

        # Dropdown per il modello
        model_options = ["Albero Decisionale", "SVM", "Rete Neurale", "KNN(Custom)", "Random Forest(Custom)"]
        self.model_var = tk.StringVar(self.master)
        self.model_var.set(model_options[0])  # Imposta il valore predefinito
        self.model_dropdown = tk.OptionMenu(self.button_frame, self.model_var, *model_options)
        self.model_dropdown.config(width=button_width)
        self.model_dropdown.pack(pady=distance)
            
        # Pulsante Addestra
        self.visualizza_button = tk.Button(self.button_frame, text="Addestra", command=self.addestramento, width=button_width + 2, anchor="center")
        self.visualizza_button.pack(pady=distance)

        # Pulsante Quit
        self.quit_button = tk.Button(self.button_frame, text="<--", command=self.show_main_menu, width=button_width + 2, anchor="center")
        self.quit_button.pack(pady=distance)
    
    
    def update_preprocessing_options(self, event):
        tipo_preprocessing = self.preprocessing_var.get()
        button_width = 20

        # Nascondi i dropdown generati dinamicamente
        if hasattr(self, 'Undersampling_dropdown'):
            self.Undersampling_dropdown.pack_forget()
        if hasattr(self, 'Oversampling_dropdown'):
            self.Oversampling_dropdown.pack_forget()
        if hasattr(self, 'Feature_Selection_dropdown'):
            self.Feature_Selection_dropdown.pack_forget()
        if hasattr(self, 'Combinazione_dropdown'):
            self.Combinazione_dropdown.pack_forget()  
        if hasattr(self, 'Trasformazione_attributi_dropdown'):
            self.Trasformazione_attributi_dropdown.pack_forget()

        # Aggiorna l'interfaccia in base al tipo di preprocessing selezionato
        if tipo_preprocessing == "Undersampling":
            Undersampling_options = ["Random", "IHT", "NearMiss_v1", "NearMiss_v2", "ClusterCentroids"]
            self.Undersampling_var = tk.StringVar(self.master)
            self.Undersampling_var.set(Undersampling_options[0])
            self.Undersampling_dropdown = tk.OptionMenu(self.button_frame, self.Undersampling_var, *Undersampling_options)
            self.Undersampling_dropdown.config(width=button_width)
            self.Undersampling_dropdown.pack(pady=5, after=self.preprocessing_dropdown)  # Posiziona dopo il menu del preprocessing

        elif tipo_preprocessing == "Oversampling":
            Oversampling_options = ["Random", "SMOTE", "ADASYN"]
            self.Oversampling_var = tk.StringVar(self.master)
            self.Oversampling_var.set(Oversampling_options[0])
            self.Oversampling_dropdown = tk.OptionMenu(self.button_frame, self.Oversampling_var, *Oversampling_options)
            self.Oversampling_dropdown.config(width=button_width)
            self.Oversampling_dropdown.pack(pady=5, after=self.preprocessing_dropdown)  # Posiziona dopo il menu del preprocessing
            
        elif tipo_preprocessing == "Feature Selection":
            Feature_Selection_options = ["Correlation-Based", "Variance Threshold", "Select_K Best", "Sequential Feature"]
            self.Feature_Selection_var = tk.StringVar(self.master)
            self.Feature_Selection_var.set(Feature_Selection_options[0])
            self.Feature_Selection_dropdown = tk.OptionMenu(self.button_frame, self.Feature_Selection_var, *Feature_Selection_options)
            self.Feature_Selection_dropdown.config(width=button_width)
            self.Feature_Selection_dropdown.pack(pady=5, after=self.preprocessing_dropdown)  # Posiziona dopo il menu del preprocessing
            
        elif tipo_preprocessing == "Combinazione Attributi":
            Combinazione_options = ["PCA", "Sparse Projection", "Gaussian Projection", "Feature Agglomeration"]
            self.Combinazione_var = tk.StringVar(self.master)
            self.Combinazione_var.set(Combinazione_options[0])
            self.Combinazione_dropdown = tk.OptionMenu(self.button_frame, self.Combinazione_var, *Combinazione_options)
            self.Combinazione_dropdown.config(width=button_width)
            self.Combinazione_dropdown.pack(pady=5, after=self.preprocessing_dropdown)  # Posiziona dopo il menu del preprocessing
            
        elif tipo_preprocessing == "Trasformazione Attributi":
            Trasformazione_attributi_options = ["Standard Scaler", "MinMaxScaler", "Normalize"]
            self.Trasformazione_attributi_var = tk.StringVar(self.master)
            self.Trasformazione_attributi_var.set(Trasformazione_attributi_options[0])
            self.Trasformazione_attributi_dropdown = tk.OptionMenu(self.button_frame, self.Trasformazione_attributi_var, *Trasformazione_attributi_options)
            self.Trasformazione_attributi_dropdown.config(width=button_width)
            self.Trasformazione_attributi_dropdown.pack(pady=5, after=self.preprocessing_dropdown)  # Posiziona dopo il menu del preprocessing
                
    def addestramento(self, model_selection = None, preprocessing_selection = None, pre_pro_method = None):
        #split dataset
        train_x, test_x, train_y, test_y = split_dataset()
        
        if(preprocessing_selection is None and model_selection is None):
            # Messaggio all'utente per l'addestramento
            messagebox.showinfo("Addestramento Modello", "Clicca OK per continuare l'addestramento. \nAttenzione! Potrebbe volerci un po' di tempo")
            #cattura selezione delle scelte dell'utente
            preprocessing_selection = self.preprocessing_var.get()
            model_selection = self.model_var.get()

        # Gestisci l'addestramento in base alle scelte dell'utente
        if preprocessing_selection == "Nessuno":
            train_x, test_x, train_y, test_y = split_dataset()
            self.handle_model_selection(model_selection, train_x, train_y, test_x, test_y, preprocessing_selection)

        elif preprocessing_selection == "Campionamento":
            train_x, test_x, train_y, test_y = stratificazione()
            self.handle_model_selection(model_selection, train_x, train_y, test_x, test_y,preprocessing_selection)
        
        elif preprocessing_selection == "Undersampling" or preprocessing_selection == "Oversampling":
            if(pre_pro_method is None):
                if preprocessing_selection == "Undersampling":
                    sampling_method = self.Undersampling_var.get()
                else:
                    sampling_method = self.Oversampling_var.get()
            else:
                sampling_method = pre_pro_method

            train_x, test_x, train_y, test_y = bilanciamento(preprocessing_selection, sampling_method)
            self.handle_model_selection(model_selection, train_x, train_y, test_x, test_y,preprocessing_selection,sampling_method)
        
        elif preprocessing_selection == "Top Combo(Under/Over)":
            train_x, test_x, train_y, test_y,sampling_method = unione_UnderSampling_OverSampling(model_selection)
            self.handle_model_selection(model_selection, train_x, train_y, test_x, test_y,preprocessing_selection,sampling_method)
        
        elif preprocessing_selection == "Feature Selection":
            if(pre_pro_method is None):
                Selection_method = self.Feature_Selection_var.get()
            else:
                Selection_method = pre_pro_method
            
            train_x, test_x, train_y, test_y = Feature_Selection(Selection_method)
            self.handle_model_selection(model_selection, train_x, train_y, test_x, test_y,preprocessing_selection,Selection_method)
        
        elif preprocessing_selection == "Combinazione Attributi":
            if(pre_pro_method is None):
                Combinazione_method = self.Combinazione_var.get()
            else:
                Combinazione_method = pre_pro_method
           
            train_x, test_x, train_y, test_y = Combinazione_Attributi(Combinazione_method)
            self.handle_model_selection(model_selection, train_x, train_y, test_x, test_y,preprocessing_selection,Combinazione_method)
            
        elif preprocessing_selection == "Trasformazione Attributi":
            if(pre_pro_method is None):
                Trasformazione_method = self.Trasformazione_attributi_var.get()
            else:
                Trasformazione_method = pre_pro_method

            train_x, test_x, train_y, test_y = Trasformazione_Attributi(Trasformazione_method)
            self.handle_model_selection(model_selection, train_x, train_y, test_x, test_y,preprocessing_selection,Trasformazione_method)

    def handle_model_selection(self, model_selection, train_x, train_y, test_x, test_y, preprocessing_selection, pre_pro_method=None):
        esiti_cancro_noPreProc = None
        test_y2 = None

        if model_selection == "Albero Decisionale":
            if preprocessing_selection != "Nessuno":
                # Alleno il modello su dati non pre-processati per mostrare la differenza alla fine
                train_x2, test_x2, train_y2, test_y2 = split_dataset()
                esiti_cancro_noPreProc = decision_tree(train_x2, train_y2, test_x2)

            # Alleno il modello sui dati passati
            esiti_cancro = decision_tree(train_x, train_y, test_x)
            grafici_valutazione(esiti_cancro, test_y, esiti_cancro_noPreProc, model_selection, test_y2, preprocessing_selection, pre_pro_method)

        elif model_selection == "SVM":
            if preprocessing_selection != "Nessuno":
                # Alleno il modello su dati non pre-processati per mostrare la differenza alla fine
                train_x2, test_x2, train_y2, test_y2 = split_dataset()
                esiti_cancro_noPreProc = SVM_classifier(train_x2, train_y2, test_x2)

            # Alleno il modello sui dati passati
            esiti_cancro = SVM_classifier(train_x, train_y, test_x)
            grafici_valutazione(esiti_cancro, test_y, esiti_cancro_noPreProc, model_selection, test_y2, preprocessing_selection, pre_pro_method)

        elif model_selection == "Rete Neurale":
            if preprocessing_selection != "Nessuno":
                # Alleno il modello su dati non pre-processati per mostrare la differenza alla fine
                train_x2, test_x2, train_y2, test_y2 = split_dataset()
                esiti_cancro_noPreProc = Artificial_Neural_Network(train_x2, train_y2, test_x2)

            # Alleno il modello sui dati passati
            esiti_cancro = Artificial_Neural_Network(train_x, train_y, test_x)
            grafici_valutazione(esiti_cancro, test_y, esiti_cancro_noPreProc, model_selection, test_y2, preprocessing_selection, pre_pro_method)

        elif model_selection == "KNN(Custom)":
            if preprocessing_selection != "Nessuno":
                # Alleno il modello su dati non pre-processati per mostrare la differenza alla fine
                train_x2, test_x2, train_y2, test_y2 = split_dataset()
                KNN_Custom_1 = KNN_Classifier_Custom()
                esiti_cancro_noPreProc = KNN_Custom_1.fit_predict(train_x2, train_y2, test_x2)

            # Alleno il modello sui dati passati
            KNN_Custom_2 = KNN_Classifier_Custom()
            esiti_cancro = KNN_Custom_2.fit_predict(train_x, train_y, test_x)
            grafici_valutazione(esiti_cancro, test_y, esiti_cancro_noPreProc, model_selection, test_y2, preprocessing_selection, pre_pro_method)

        elif model_selection == "Random Forest(Custom)":
            if preprocessing_selection != "Nessuno":
                # Alleno il modello su dati non pre-processati per mostrare la differenza alla fine
                train_x2, test_x2, train_y2, test_y2 = split_dataset()
                RandomForest_Custom_1 = RandomForestCustom()
                esiti_cancro_noPreProc = RandomForest_Custom_1.fit_predict(train_x2, train_y2, test_x2)

            # Alleno il modello sui dati passati
            RandomForest_Custom_2 = RandomForestCustom()
            esiti_cancro = RandomForest_Custom_2.fit_predict(train_x, train_y, test_x)
            grafici_valutazione(esiti_cancro, test_y, esiti_cancro_noPreProc, model_selection, test_y2, preprocessing_selection, pre_pro_method)

    # MIGLIOR COMBINAZIONE
    def show_miglior_combinazione(self):
        # Ripulisce i bottoni e mostra il sotto-menu
        self.clear_buttons()
        self.initialize_combinazioni()
    
    def initialize_combinazioni(self):
        distance = 5
        button_width = 20  # Larghezza fissa per tutti i bottoni

        # Label per il testo sopra il pulsante del modello
        model_label = tk.Label(self.button_frame, text="Modello", anchor="center", font=("Helvetica", 10))
        model_label.pack(pady= 0)

        # Dropdown per il modello
        model_options = ["Albero Decisionale", "SVM", "Rete Neurale", "KNN(Custom)", "Random Forest(Custom)"]
        self.model_comb_var = tk.StringVar(self.master)
        self.model_comb_var.set(model_options[0])  # Imposta il valore predefinito
        self.model_comb_dropdown = tk.OptionMenu(self.button_frame, self.model_comb_var, *model_options)
        self.model_comb_dropdown.config(width=button_width)
        self.model_comb_dropdown.pack(pady=distance)

        # Pulsante cerca Combinazione singola
        self.search_button = tk.Button(self.button_frame, text="Cerca Combinazione", command=self.search_combination, width=button_width + 2, anchor="center")
        self.search_button.pack(pady=distance)

        # Label per il testo sopra il pulsante del modello
        model_label = tk.Label(self.button_frame, text="Confronto Tra Modelli", anchor="center", font=("Helvetica", 10))
        model_label.pack(pady= 0)
        
        # Dropdown per il modello
        combination_options = ["Best Combo", "Best sampling", "Best Balancing", "Best C.A.", "Best F.S.","Best T.A.", "Best No P.P."]
        self.combination_options_var = tk.StringVar(self.master)
        self.combination_options_var.set(combination_options[0])  # Imposta il valore predefinito
        self.combination_options_dropdown = tk.OptionMenu(self.button_frame, self.combination_options_var, *combination_options)
        self.combination_options_dropdown.config(width=button_width)
        self.combination_options_dropdown.pack(pady=distance)

        # Pulsante Mostra tutti i modelli con le migliori combinazioni
        self.search_button = tk.Button(self.button_frame, text="Visualizza", command= self.confronta_le_migliori_combinazioni , width=button_width + 2, anchor="center")
        self.search_button.pack(pady=distance)

        # Pulsante Quit
        self.quit_button = tk.Button(self.button_frame, text="<--", command=self.show_main_menu, width=button_width + 2, anchor="center")
        self.quit_button.pack(pady=distance)

    def confronta_le_migliori_combinazioni(self):
        combinazione_selection = self.combination_options_var.get()

        '''
        best combo --> migliori combinazioni in assoluto di tutti i modelli
        Best sampling --> migliori combinazioni con il campionamento
        Best Balancing --> migliori combinazioni con il Bilanciamento
        Best C.A. --> migliori combinazioni con la combinazione di attributi
        Best F.S. --> migliori combinazioni con la Selezione degli attributi
        Best T.A. --> migliori combinazioni con la Trasformazione degli attributi
        Best No P.P. --> confronto performance modelli senza pre-processing
        '''
        if(combinazione_selection == "Best Combo"):
            confronta_le_migliori_in_assoluto()
        elif(combinazione_selection == "Best sampling"):
            confronto_migliori_campionamento()
        elif(combinazione_selection ==  "Best Balancing"):
            confronto_migliori_bilanciamento()
        elif(combinazione_selection == "Best C.A."):
            confronto_migliori_Combinazione_Attributi()
        elif(combinazione_selection == "Best F.S."):
            confronto_migliori_Feature_Selection()
        elif(combinazione_selection == "Best T.A."):
            confronto_migliori_Trasformazione_Attributi()
        elif(combinazione_selection == "Best No P.P."):
            confronto_Senza_Pre_processing()



    def search_combination(self):

        model_selection = self.model_comb_var.get()
        # Ottiene la miglior combinazione e avvia l'addestramento con i parametri ottenuti
        combinazione = best_combination(model_selection)
        self.addestramento(combinazione[0], combinazione[1], combinazione[3])
    '''
    # TUNING MODELLI
    def Tuning_modelli(self):
        # Ripulisce i bottoni e mostra il sotto-menu
        self.clear_buttons()
        self.initialize_Tuning_modelli_buttons()

    def initialize_Tuning_modelli_buttons(self):
        distance = 5 
        # Bottoni per le diverse operazioni di tuning
        self.DThree_tuning_button = tk.Button(self.button_frame, text="DThree Tuning", command = tuning_Albero_decisionale, width=20, anchor="center")
        self.DThree_tuning_button.pack(pady=distance)

        self.SVM_tuning_button = tk.Button(self.button_frame, text="SVM Tuning", command=tuning_SVM, width=20, anchor="center")
        self.SVM_tuning_button.pack(pady=distance)

        self.ANN_tuning_button = tk.Button(self.button_frame, text="ANN Tuning", command=tuning_ANN, width=20, anchor="center")
        self.ANN_tuning_button.pack(pady=distance)

        self.KNN_tuning_button = tk.Button(self.button_frame, text="KNN Tuning", command= tuning_KNN_custom, width=20, anchor="center")
        self.KNN_tuning_button.pack(pady=distance)

        self.KNN_tuning_button = tk.Button(self.button_frame, text="Random Forest Tuning", command= tuning_Random_Forest, width=20, anchor="center")
        self.KNN_tuning_button.pack(pady=distance)

        # Pulsante Quit per tornare al menu principale
        self.quit_button = tk.Button(self.button_frame, text="<--", command=self.show_main_menu, width=20, anchor="center")
        self.quit_button.pack(pady=distance)
        
        # Mostra una finestra di informazioni prima di procedere
        messagebox.showinfo("Visualizzazione Tuning", "Clicca OK per continuare. \nAttenzione! Potrebbe volerci molto tempo")
    '''
    # FINESTRE
    def finestra_scrolled_text(self, master, title, width, height):
        # Crea una nuova finestra
        window = tk.Toplevel(master)
        window.title(title)
        window.geometry(f"{width}x{height}")

        # Crea uno ScrolledText per visualizzare le informazioni con barra di scorrimento
        text_widget = ScrolledText(window, wrap=tk.NONE, width=100, height=30)
        text_widget.pack(expand=True, fill='both', padx=10, pady=10)

        # Disabilita l'opzione di inserimento per rendere il widget di sola lettura
        text_widget.bind("<Key>", lambda e: "break")

        # Crea una barra di scorrimento orizzontale
        scrollbar_x = Scrollbar(window, command=text_widget.xview, orient=HORIZONTAL)
        scrollbar_x.pack(side=tk.BOTTOM, fill='x')

        text_widget['xscrollcommand'] = scrollbar_x.set

        # Aggiungi lo scorrimento con la rotella del mouse
        text_widget.bind_all("<MouseWheel>", lambda e: self.on_mousewheel(text_widget, e))

        return text_widget  # Ritorna il widget di testo così che possa essere utilizzato per l'inserimento di testo

    def Finestra_Immagine(self, master, title, img_path, width, height):
        # Crea una nuova finestra
        Matrice_confusione = tk.Toplevel(master)
        Matrice_confusione.title(title)

        # Carica l'immagine utilizzando PIL
        image = Image.open(img_path)

        # Crea un frame per contenere la canvas principale e la scrollbar verticale
        frame = tk.Frame(Matrice_confusione)
        frame.pack(side="top", fill="both", expand=True)

        # Crea una canvas all'interno del frame
        canvas = tk.Canvas(frame, width=width, height=height)
        canvas.pack(side="left", fill="both", expand=True)

        # Aggiungi una scrollbar verticale
        v_scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill="y")

        # Aggiungi una scrollbar orizzontale
        h_scrollbar = tk.Scrollbar(Matrice_confusione, orient=tk.HORIZONTAL, command=canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill="x")

        # Configura la canvas per utilizzare le scrollbar
        canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)

        # Inserisci l'immagine nella canvas
        image_tk = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor="nw", image=image_tk)

        # Imposta la dimensione della canvas per consentire la scrollbar
        canvas.config(scrollregion=canvas.bbox(tk.ALL))

        # Aggiungi lo scorrimento con la rotella del mouse
        canvas.bind_all("<MouseWheel>", lambda e: self.on_mousewheel(canvas, e))

        # Mostra la finestra
        Matrice_confusione.mainloop()

    def on_mousewheel(self, text_widget, event):
        # Imposta lo scorrimento orizzontale della text_widget in base all'evento della rotella del mouse
        text_widget.yview_scroll(-1 * (event.delta // 120), "units")

if __name__ == "__main__":
    # Crea una finestra principale utilizzando Tkinter
    root = tk.Tk()
    # Crea un'istanza della classe MenuApp, passando la finestra principale come argomento
    app = MenuApp(root)
    # Avvia il ciclo principale per l'applicazione Tkinter
    root.mainloop()
