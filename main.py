# Importing Necessary Libraries
import tkinter as tk
from tkinter import *
from tkinter import filedialog
import tensorflow as tf
import cv2
from PIL import Image, ImageTk
from mtcnn.mtcnn import MTCNN

from helper_functions import *

detector = MTCNN()
class_name = ("Blue", "Blue Gray", "Brown", "Brown Gray", "Brown Black", "Green", "Green Gray", "Other")
EyeColor = {
    class_name[0]: ((166, 21, 50), (240, 100, 85)),
    class_name[1]: ((166, 2, 25), (300, 20, 75)),
    class_name[2]: ((2, 20, 20), (40, 100, 60)),
    class_name[3]: ((20, 3, 30), (65, 60, 60)),
    class_name[4]: ((0, 10, 5), (40, 40, 25)),
    class_name[5]: ((60, 21, 50), (165, 100, 85)),
    class_name[6]: ((60, 2, 25), (165, 20, 65))
}


def check_color(hsv, color):
    if (hsv[0] >= color[0][0]) and (hsv[0] <= color[1][0]) and (hsv[1] >= color[0][1]) and hsv[1] <= color[1][1] and (
            hsv[2] >= color[0][2]) and (hsv[2] <= color[1][2]):
        return True
    else:
        return False


def find_class(hsv):
    color_id = 7
    for i in range(len(class_name) - 1):
        if check_color(hsv, EyeColor[class_name[i]]) == True:
            color_id = i

    return color_id


def eye_color(image):
    imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[0:2]
    imgMask = np.zeros((image.shape[0], image.shape[1], 1))

    result = detector.detect_faces(image)
    if result == []:
        print('Warning: Can not detect any face in the input image!')
        return

    left_eye = result[0]['keypoints']['left_eye']
    right_eye = result[0]['keypoints']['right_eye']

    eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
    eye_radius = eye_distance / 15  # approximate

    cv2.circle(imgMask, left_eye, int(eye_radius), (255, 255, 255), -1)
    cv2.circle(imgMask, right_eye, int(eye_radius), (255, 255, 255), -1)

    eye_class = np.zeros(len(class_name), np.float32)

    for y in range(0, h):
        for x in range(0, w):
            if imgMask[y, x] != 0:
                eye_class[find_class(imgHSV[y, x])] += 1

    main_color_index = np.argmax(eye_class[:len(eye_class) - 1])
    total_vote = eye_class.sum()

    print("\n\nDominant Eye Color: ", class_name[main_color_index])
    print("\n **Eyes Color Percentage **")
    for i in range(len(class_name)):
        print(class_name[i], ": ", round(eye_class[i] / total_vote * 100, 2), "%")
    confe = round(eye_class[main_color_index] / total_vote * 100, 2)
    label = 'Dominant Eye Color: %s' % class_name[main_color_index]
    return label, confe


# Loading the Model
model = tf.keras.models.load_model('/home/ash/PycharmProjects/Eye_Colour_detection_and_baldness_detection/baldness_prediction_model-20220831T154025Z-001/baldness_prediction_model')

# Initializing the GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Baldness & Eye Color Detector')
top.configure(background='#CDCDCD')

# Initializing the Labels (1 for age and 1 for Sex)
label1 = Label(top, background="#CDCDCD", font=('arial', 15, "bold"))
label2 = Label(top, background="#CDCDCD", font=('arial', 15, 'bold'))
sign_image = Label(top)


# Defining Detect function which detects the age and gender of the person in image using the model
def show_pred_image(filename):
    image = load_and_prep_image(filename, scale=False)
    image = image[:, :, :3]
    pred_prob = model.predict(tf.expand_dims(image, axis=0))

    if pred_prob >= 0.5:
        pred_class = "NotBald"
    else:
        pred_class = "Bald"
        pred_prob = 1.0 - pred_prob

    return pred_class, pred_prob


def Detect(file_path):
    global label_packed
    pred_class, pred_prob = show_pred_image(file_path)
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    eyeColor, percent = eye_color(image)
    print(f"Person is {pred_class}")
    print(f"Eye colour is {eyeColor} = {percent}%")
    label1.configure(foreground="#011638", text=eyeColor)
    label2.configure(foreground="#011638", text=pred_class)


# Defining Show_detect button function
def show_Detect_button(file_path):
    Detect_b = Button(top, text="Detect Image", command=lambda: Detect(file_path), padx=10, pady=5)
    Detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    Detect_b.place(relx=0.79, rely=0.46)


# Defining Upload Image Function
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        label2.configure(text='')
        show_Detect_button(file_path)
    except:
        pass


upload = Button(top, text="Upload an Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand=True)
label1.pack(side="bottom", expand=True)
label2.pack(side="bottom", expand=True)
heading = Label(top, text="Baldness and Eye colour Detection", pady=20, font=('arial', 20, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()
top.mainloop()
