import xmltodict
import numpy as np

def convert_xml_to_mot(xml_path, output_txt_path):
    """
    Converts an XML annotation file to MOTChallenge format (.txt).
    
    Parameters:
    - xml_path: str -> Path to the XML file.
    - output_txt_path: str -> Path where the MOTChallenge format file will be saved.
    """
    with open(xml_path, 'r') as xml_file:
        tracks = xmltodict.parse(xml_file.read())['annotations']['track']

    with open(output_txt_path, "w") as txt_file:
        for track in tracks:
            object_id = int(track['@id'])  # ID del objeto
            label = track['@label']

            # Solo incluir objetos de tipo 'car'
            if label == 'car':
                for box in track['box']:
                    frame = int(box['@frame'])  # Frame ID
                    xtl = float(box['@xtl'])  # X top-left
                    ytl = float(box['@ytl'])  # Y top-left
                    xbr = float(box['@xbr'])  # X bottom-right
                    ybr = float(box['@ybr'])  # Y bottom-right
                    w = xbr - xtl  # Width
                    h = ybr - ytl  # Height

                    # Formato MOTChallenge:
                    # frame_id, object_id, x, y, width, height, confidence, class, visibility
                    line = f"{frame}, {object_id}, {xtl:.2f}, {ytl:.2f}, {w:.2f}, {h:.2f}, 1, -1, -1\n"
                    txt_file.write(line)

    print(f"Archivo convertido y guardado en: {output_txt_path}")

# Uso de la función
xml_path = "./data/ai_challenge_s03_c010-full_annotation.xml"
output_txt_path = "./data/ground_truth.txt"
convert_xml_to_mot(xml_path, output_txt_path)
