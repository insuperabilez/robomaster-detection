import sys
import time
import tkinter as tk
from tkinter.messagebox import askokcancel

from PIL import Image, ImageTk
from robomaster import robot
from startmobilenet import start_mobilenet
from startresnet101 import start_resnet101
from startresnet50 import start_resnet50
from startyolo import start_yolo
from torch.cuda import is_available as cuda_available

# image = cv2.imread("image.jpg")
device = 'cuda' if cuda_available() else 'cpu'
ep_camera = None


def stop_connection():
    global ep_camera
    global main_window
    if ep_camera is not None:
        ep_camera.stop_video_stream()
        ep_robot.close()
    if askokcancel("Выход", "Закрыть приложение?"):
        main_window.destroy()
        sys.exit()


def change_window_content():
    global ep_camera
    for widget in main_window.winfo_children():
        widget.destroy()
    try:
        ep_robot = robot.Robot()
        ep_robot.initialize()
        ep_camera = ep_robot.camera
    except Exception:
        print('Соединение не установлено')
    print('Соединение установлено')
    time.sleep(1)
    button1 = tk.Button(main_window, text="YOLO8n ", font=("Arial", 16), width=15,
                        command=lambda: start_yolo(ep_camera, device))
    button2 = tk.Button(main_window, text="resnet 101", font=("Arial", 16), width=15,
                        command=lambda: start_resnet101(ep_camera, device))
    button3 = tk.Button(main_window, text="resnet 50", font=("Arial", 16), width=15,
                        command=lambda: start_resnet50(ep_camera, device))
    button4 = tk.Button(main_window, text="mobilenet ssd", font=("Arial", 16), width=15,
                        command=lambda: start_mobilenet(ep_camera, device))
    panel = tk.Label(main_window)
    panel.pack(pady=10)

    for button in [button1, button2, button3, button4]:
        button.pack(side=tk.LEFT, anchor=tk.SE)
        if ep_camera is None:
            button.config(state=tk.DISABLED)

    image = Image.open('background1.png')
    imgtk = ImageTk.PhotoImage(image)
    panel.imgtk = imgtk
    panel.config(image=imgtk)

    disconnect_button = tk.Button(main_window, text="X", font=("Arial", 16), width=20, command=stop_connection)
    disconnect_button.pack(side=tk.BOTTOM, anchor=tk.SE)


main_window = tk.Tk()
main_window.title("Главное окно")
main_window.geometry("800x600")

connect_button = tk.Button(main_window, text="Подключиться", font=("Arial", 16), width=20,
                           command=change_window_content)
connect_button.pack(pady=40)

main_window.protocol("WM_DELETE_WINDOW", stop_connection)
main_window.mainloop()
