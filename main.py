import tkinter as tk
import transformers
import time
from startyolo import start_yolo
from startresnet1 import start_resnet
from startdetrresnet import start_detr
from startsegment import start_segment
from functools import partial
from robomaster import robot
#image = cv2.imread("image.jpg")
def stop_connection():
    pass
    #ep_camera.stop_video_stream()
    #ep_robot.close()
def change_window_content():
    for widget in main_window.winfo_children():
        widget.destroy()

    ep_camera = None
    try:
        ep_robot = robot.Robot()
        ep_robot.initialize()
        ep_camera = ep_robot.camera
    except Exception:
        print('Соединение не установлено')
        # Создаем 4 независимых кнопки
    print('Соединение установлено')
    time.sleep(1)
    button1 = tk.Button(main_window, text="YOLO8n ", font=("Arial", 16), width=15, command=lambda: start_yolo(ep_camera))
    button2 = tk.Button(main_window, text="detr resnet 101", font=("Arial", 16), width=15,command=lambda:start_resnet(ep_camera))
    button3 = tk.Button(main_window, text="detr resnet 50", font=("Arial", 16), width=15, command=start_detr)
    button4 = tk.Button(main_window, text="Segmentation", font=("Arial", 16), width=15, command=start_segment)

    # Размещаем кнопки
    for button in [button1, button2, button3, button4]:
        button.pack(pady=10)

    disconnect_button = tk.Button(main_window, text="X", font=("Arial", 16), width=20, command=stop_connection)
    disconnect_button.pack(side=tk.BOTTOM, anchor=tk.SE)
# Создаем главное окно
main_window = tk.Tk()
main_window.title("Главное окно")
main_window.geometry("800x600")

# Создаем кнопку "Подключиться"
connect_button = tk.Button(main_window, text="Подключиться", font=("Arial", 16), width=20,command=change_window_content)
connect_button.pack(pady=40)

# Запускаем цикл обработки событий
main_window.mainloop()
#ep_camera.stop_video_stream()
#ep_robot.close()