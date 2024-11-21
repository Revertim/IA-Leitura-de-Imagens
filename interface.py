import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import filedialog, Tk, Button, Label
from PIL import Image, ImageTk
from find_contours import find_contours
import matplotlib.pyplot as plt

model = load_model('model.h5')

plate_cascade = cv2.CascadeClassifier("archive/indian_license_plate.xml")

def preprocess_image(image):
    img = cv2.resize(image, (28, 28))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def segment_characters(image):

    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))

    dimensions = [img_binary_lp.shape[0]/6, img_binary_lp.shape[0]/2,
                  img_binary_lp.shape[1]/10, 2*img_binary_lp.shape[1]/3]

    char_list = find_contours(dimensions, img_binary_lp)
    
    for i, char_img in enumerate(char_list):
        cv2.imwrite(f"char_{i}.png", char_img)

    return char_list

def classify_character(image):
    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed)
    predicted_class = np.argmax(prediction)
    return predicted_class

def show_results(char_list):
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, c in enumerate(characters):
        dic[i] = c

    output = []
    for ch in char_list:
        img = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img)
        img = img.reshape(1, 28, 28, 3)
        y_ = np.argmax(model.predict(img), axis=1)[0]
        character = dic[y_]
        output.append(character)

    plate_number = ''.join(output)
    return plate_number

def fix_dimension(img):
    new_img = np.zeros((28, 28, 3))
    for i in range(3):
        new_img[:, :, i] = img
    return new_img

def detectar_placa():
    if panel.image:
        imagem_cv = cv2.cvtColor(np.array(panel.image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in plates:
            cv2.rectangle(imagem_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)

            plate_img = imagem_cv[y:y+h, x:x+w]
            characters = segment_characters(plate_img)
            
            plate_number = show_results(characters)
            print(f"Número detectado: {plate_number}")

            plt.figure(figsize=(10, 6))
            for i, ch in enumerate(characters):
                img = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
                plt.subplot(3, 4, i + 1)
                plt.imshow(img, cmap='gray')
                plt.title(f'Caractere: {plate_number[i]}')
                plt.axis('off')
            plt.show()

        plate_image = Image.fromarray(imagem_cv)
        plate_image = plate_image.resize((300, 300))
        plate_image = ImageTk.PhotoImage(plate_image)
        panel.config(image=plate_image)
        panel.image = plate_image


def carregar_imagem():
    caminho_arquivo = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp"), ("Todos os Arquivos", "*.*")])
    if caminho_arquivo:
        imagem = cv2.imread(caminho_arquivo)
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        imagem = Image.fromarray(imagem)
        imagem = imagem.resize((300, 300))
        imagem = ImageTk.PhotoImage(imagem)
        panel.config(image=imagem)
        panel.image = imagem

root = Tk()
root.title("Detecção de Placas de Carro")

btn_carregar = Button(root, text="Carregar Imagem", command=carregar_imagem)
btn_carregar.pack()

btn_detectar = Button(root, text="Detectar Placa", command=detectar_placa)
btn_detectar.pack()

panel = Label(root)
panel.pack()

root.mainloop()
