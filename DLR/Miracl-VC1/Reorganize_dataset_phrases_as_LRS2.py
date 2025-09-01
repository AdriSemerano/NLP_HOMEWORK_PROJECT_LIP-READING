import os
import shutil
import re

# --- Configuration ---
# Directory base of your current dataset structure
source_base_dir = "path_to_MIRACL-VC1"

# Directory where the new, reorganized dataset will be created
destination_base_dir = r"path_to_new_MIRACL-VC1"

# Mappatura degli ID delle frasi alle frasi testuali
phrase_mapping = {
    "01": "Stop navigation",
    "02": "Excuse me",
    "03": "I am sorry",
    "04": "Thank you",
    "05": "Good bye",
    "06": "I love this game",
    "07": "Nice to meet you",
    "08": "You are welcome",
    "09": "How are you",
    "10": "Have a good time"
}


# --- Funzione per l'estrazione di informazioni dal nome del file ---
def parse_filename(filename):
    """
    Estrae l'ID del soggetto, l'ID della frase e il numero dell'istanza
    da un nome di file video.
    Es. 'F01_02_01.mp4' -> ('F01', '02', '01')
    """
    match = re.match(r"([MF]\d+)_(\d+)_(\d+)\.mp4$", filename)
    if match:
        return match.groups()
    return None


# --- Funzione principale per la ristrutturazione ---
def restructure_dataset(source_dir, dest_dir, mapping):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Creata directory di destinazione: {dest_dir}")

    # Itera attraverso le cartelle delle frasi (es. "Stop navigation")
    for phrase_folder in os.listdir(source_dir):
        phrase_path = os.path.join(source_dir, phrase_folder)
        if not os.path.isdir(phrase_path):
            continue

        # Ottieni l'ID numerico della frase dalla mappatura
        phrase_id = None
        for key, value in mapping.items():
            if value == phrase_folder:
                phrase_id = key
                break

        if not phrase_id:
            print(f"Avviso: Cartella '{phrase_folder}' non trovata nella mappatura. Saltando.")
            continue

        # Itera attraverso i video all'interno della cartella della frase
        for video_filename in os.listdir(phrase_path):
            if not video_filename.endswith('.mp4'):
                continue

            # Ottieni i dettagli del video dal nome del file
            parts = parse_filename(video_filename)
            if not parts:
                print(f"Avviso: Nome file non valido '{video_filename}'. Saltando.")
                continue

            speaker_id, _, instance_num = parts

            # Percorso del video di origine
            source_video_path = os.path.join(phrase_path, video_filename)

            # Crea la cartella del soggetto di destinazione
            speaker_dest_path = os.path.join(dest_dir, speaker_id)
            if not os.path.exists(speaker_dest_path):
                os.makedirs(speaker_dest_path)

            # Nuovo nome del file video e del file di testo
            new_filename_base = f"{phrase_id}_{instance_num}"
            new_video_path = os.path.join(speaker_dest_path, f"{new_filename_base}.mp4")
            new_txt_path = os.path.join(speaker_dest_path, f"{new_filename_base}.txt")

            # Sposta il video nella nuova posizione
            shutil.move(source_video_path, new_video_path)

            # Crea il file di testo con il contenuto della frase
            with open(new_txt_path, 'w') as f:
                f.write(mapping[phrase_id].upper())

            print(f"Spostato '{video_filename}' a '{new_video_path}'")
            print(f"Creato file di testo '{os.path.basename(new_txt_path)}'")

    print("\nRistrutturazione del dataset completata!")


# Esegui la funzione

restructure_dataset(source_base_dir, destination_base_dir, phrase_mapping)
