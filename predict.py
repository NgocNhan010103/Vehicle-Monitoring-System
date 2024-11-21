import math
import time
import tkinter as tk
from tkinter.ttk import *
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog
import pandas as pd
from ultralytics import YOLO
import pandas as pd
from tracker import *
from collections import defaultdict
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from statistics import mean

cap = None
is_camera_on = False
frame_skip_threshold = 3
model = YOLO('yolov8n.pt')
video_paused = False

previous_positions = {}
previous_time = time.time()
pixel_to_meter_ratio = 8 

down = {}
up = {}
counter_in = []
counter_out = []
traffic_data = [] 

vehicle_count_in = defaultdict(int)
vehicle_count_out = defaultdict(int)
vehicle_speed_in = defaultdict(list)
vehicle_speed_out = defaultdict(list)

Vehicle_speed_violation = []

tracker = Tracker()
count = 0
max_speed = 80
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
row_num = 0
direction = ''
violation_status = False
is_scrollbar = False
img_width, img_height = 120, 100
num_columns = 3  
current_image_count = 0

count_text = None
img_frame = None
scrollable_frame = None


if not os.path.exists('detected_frames'):
    os.makedirs('detected_frames')

#read file with file path and return list
def read_classes_from_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

#default colors for vehicles bouding box
def compute_color_for_vehicle(vehicle):
    color_dict = {
        'person': (85, 45, 255),
        'car': (222, 82, 175),
        'bus': (0, 204, 255),
        'truck': (0, 149, 255)
    }
    if vehicle in color_dict:
        return color_dict[vehicle]
    else:
        palette = [85, 45, 255]
        color = [int((p * (hash(vehicle) ** 2 - hash(vehicle) + 1)) % 255) for p in palette]
        return tuple(color)

def start_webcam():
    global cap, is_camera_on, video_paused, row_num
    if not is_camera_on:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return
        is_camera_on = True
        video_paused = False
        row_num = 0
        #reset_canvas()
        #reset_chart_data()
        update_canvas()  

def stop_webcam():
    global cap, is_camera_on, video_paused
    if cap is not None:
        cap.release()
        is_camera_on = False
        video_paused = False

def pause_resume_video():
    global video_paused
    video_paused = not video_paused

def select_file():
    global cap, is_camera_on, video_paused, row_num
    if is_camera_on:
        stop_webcam()  
    file_path = filedialog.askopenfilename(filetypes=[('Video files', '*.mp4 *.avi *.mov')])
    if file_path:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return
        is_camera_on = True
        video_paused = False
        row_num = 0
        #reset_canvas()
        #reset_chart_data()
        update_canvas()  

#estimate speed
def estimate_speed(prev_location, curr_location, elapsed_time):
    distance = math.sqrt(math.pow(curr_location[0] - prev_location[0], 2) + math.pow(curr_location[1] - prev_location[1], 2))

    if elapsed_time > 0:
        speed_pixels_per_second = distance / elapsed_time
    else:
        speed_pixels_per_second = 0
    speed_mps = speed_pixels_per_second / pixel_to_meter_ratio
    speed_kmh = speed_mps * 3.6
    return speed_kmh

def calculate_average_speed(vehicle, direction):
    if direction =='In':
        speeds = vehicle_speed_in.get(vehicle, [])
        print(f"Speeds for {vehicle}: {speeds}") 
    else:
        speeds = vehicle_speed_out.get(vehicle, [])

    if speeds:
        return sum(speeds) / len(speeds)
    return 0

#checking vehicle whether there is a violation or not
def is_violation(id, speed_kmh, x3, x4, y3, y4, frame):
    global violation_status, max_speed
    global current_image_count
    if speed_kmh > max_speed:
        if id not in Vehicle_speed_violation:
            vehicle_image = frame[y3:y4, x3:x4] 
            vehicle_filename = f'detected_frames/{id}.jpg'
            cv2.imwrite(vehicle_filename, cv2.cvtColor(vehicle_image, cv2.COLOR_RGB2BGR)) 
            Vehicle_speed_violation.append(id)

            vehicle_pil_image = Image.fromarray(vehicle_image)
            vehicle_pil_image = vehicle_pil_image.resize((img_width, img_height), Image.Resampling.LANCZOS)
            vehicle_photo = ImageTk.PhotoImage(vehicle_pil_image)

            vehicle_display_frame = tk.Frame(img_frame)

            vehicle_label = tk.Label(vehicle_display_frame, image=vehicle_photo)
            vehicle_label.image = vehicle_photo
            vehicle_label.pack()

            speed_label = tk.Label(vehicle_display_frame, text=f'Speed: {int(speed_kmh)} km/h', fg='black')
            speed_label.pack()

            row = current_image_count // num_columns
            col = current_image_count % num_columns
            vehicle_display_frame.grid(row=row, column=col, padx=5, pady=5)

            current_image_count += 1

            count_text.config(text='Number of violating vehicles :' f'{current_image_count}' )

            violation_status = True
    else:
        violation_status = False

    img_frame.update_idletasks()
    scrollable_canvas.config(scrollregion=scrollable_canvas.bbox('all'))

    return violation_status

#draw chart for traffic vehicle
def update_traffic_chart():
    global traffic_data
    if len(traffic_data) > 30:
        traffic_data.pop(0)
    traffic_ax.clear()
    traffic_ax.plot(traffic_data, color='green')
    traffic_ax.set_title('Traffic Volume Over Time')
    traffic_ax.set_xlabel('Time (s)')
    traffic_ax.set_ylabel('Vehicle Count')
    traffic_canvas.draw()

#delete and initialize scroll aere
def reset_canvas():
    if count_text:
        count_text.destroy()  
    if img_frame:
        img_frame.destroy()
    if scrollable_frame:
        scrollable_frame.destroy()
    initialize_vehicle_info_frame() 

#Clear chart data and update data with new frame
def reset_chart_data():
    global traffic_data
    traffic_data.clear()  
    update_traffic_chart()  

#init frame display scroll
def initialize_vehicle_info_frame():
    global count_text, img_frame, scrollable_frame

    count_text = tk.Label(counter_canvas, text='Number of violating vehicles : 0', font=('Helvetica', 15), fg='white', bg='darkgreen')
    counter_canvas.create_window((screen_width - 1500) // 2, 20, window=count_text)
    
    img_frame = tk.Frame(scrollable_canvas)
    scrollable_canvas.create_window((0, 0), window=img_frame, anchor='nw')
    
    img_frame.update_idletasks()
    scrollable_canvas.config(scrollregion=scrollable_canvas.bbox('all'))
    
    scrollable_frame = tk.Frame(scrollable_frame_canvas)
    scrollable_frame_canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')

    scrollable_frame.update_idletasks()
    scrollable_frame_canvas.config(scrollregion=scrollable_frame_canvas.bbox('all'))

#Insert vehicle to scroll, if violate speed, warning
def add_vehicle_info(img,vehicle_type, speed, direction, violation_status, row_num):
    global row_bg_color
    if violation_status == 'True':
        row_bg_color = 'red'
    else:
        row_bg_color = 'white'

    row_frame = tk.Label(scrollable_frame, bg=row_bg_color)
    row_frame.grid(row=row_num, columnspan=4, sticky='ew',pady=(1,1))

    vehicle_img_label = tk.Label(row_frame, image=img)
    vehicle_img_label.image = img
    vehicle_img_label.grid(row=0, column=0, padx=30, sticky='w')

    vehicle_type_label = tk.Label(row_frame, text=vehicle_type.capitalize(), font=('Helvetica', 10), fg='black', bg=row_bg_color)
    vehicle_type_label.grid(row=0, column=1, padx=90, sticky='w')

    speed_info_label = tk.Label(row_frame, text=f'{speed} km/h', font=('Helvetica', 10), fg='black', bg=row_bg_color)
    speed_info_label.grid(row=0, column=2, padx=80, sticky='w')

    direction_label = tk.Label(row_frame, text=direction, font=('Helvetica', 10), fg='black', bg=row_bg_color)
    direction_label.grid(row=0, column=3, padx=60, sticky='w')

    violation_status_label = tk.Label(row_frame, text=violation_status, font=('Helvetica', 10), fg='black', bg=row_bg_color)
    violation_status_label.grid(row=0, column=4, padx=80, sticky='w')

    scrollable_frame.update_idletasks()
    scrollable_frame_canvas.config(scrollregion=scrollable_frame_canvas.bbox('all'))

def quit_app():
    stop_webcam()
    root.quit()
    root.destroy()

#Update frames were predicted to canvas
def update_canvas():
    global is_camera_on, video_paused, previous_positions, count,violation_status, row_num, direction, max_speed
    y_offset = 60
    if is_camera_on:
        if not video_paused:
            ret, frame = cap.read()
            if not ret:
                return

            #Change sz and color suit canvas
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (1500,735) )

            selected_class = class_selection.get()
            max_speed = int(class_speed.get())

            #predict with model 
            results = model.predict(frame, iou=0.4)
            a = results[0].boxes.data
            a = a.detach().cpu().numpy()
            px = pd.DataFrame(a).astype('float')
            list = []

            for index, row in px.iterrows():
                x1 = int(row[0])
                y1 = int(row[1])
                x2 = int(row[2])
                y2 = int(row[3])
                d = int(row[5])
                #c = class_list[d]
                if 0 <= d < len(class_list):
                    c = class_list[d]
                else:
                    break

                #if All, draw bouding box all object, else draw bouding box for object have class == c
                if selected_class == 'All' or c == selected_class:
                    list.append([x1, y1, x2, y2])
                    color = compute_color_for_vehicle(c)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f'{c}', (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)
            
            bbox_id = tracker.update(list)

            #browse through each bouding box and take coordinates 
            for bbox in bbox_id:
                x3, y3, x4, y4, id = bbox
                cx = int(x3 + x4) // 2
                cy = int(y3 + y4) // 2

                if id in previous_positions:
                    prev_cx, prev_cy, prev_time = previous_positions[id]
                    elapsed_time = time.time() - prev_time

                    #estimate speed with prev (current) position and elapsed time
                    speed_kmh = estimate_speed((prev_cx,prev_cy), (cx,cy), elapsed_time)

                    cv2.putText(frame, f'{int(speed_kmh)} Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)
                    
                    vehicle_image = frame[y3:y4, x3:x4] 
                    vehicle_pil_image = Image.fromarray(vehicle_image)
                    vehicle_pil_image = vehicle_pil_image.resize((50, 50), Image.Resampling.LANCZOS)
                    vehicle_photo = ImageTk.PhotoImage(vehicle_pil_image)

                    handle_vehicle_crossing(cx, cy , speed_kmh, bbox, frame, c, vehicle_photo)
                   
                previous_positions[id] = (cx, cy, time.time())
            
            #insert chart data and draw 
            traffic_data.append(len(bbox_id))
            update_traffic_chart()

            draw(frame)
            
            #save vehicle img into the folder
            #frame_filename = f'detected_frames/frame_{count}.jpg'
            #cv2.imwrite(frame_filename, frame)

            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            video_canvas.img = photo
            video_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    #call func update_canvas() after 10ms
    video_canvas.after(1, update_canvas)

def handle_vehicle_crossing(cx,cy, speed_kmh, bbox, frame, c, vehicle_photo):
    global violation_status, row_num, counter_in, counter_out, direction
    x3, y3, x4, y4, id = bbox
    #Counting vehicle from top to button
    if red_line[0][1] < (cy+offset) and red_line[0][1] > (cy-offset):
        if cx > red_line[0][0] and cx < red_line[1][0]:
            violation_status = is_violation(id, speed_kmh, x3, x4, y3, y4, frame)
            if id not in counter_in:
                row_num += 1
                counter_in.append(id)
                vehicle_count_in[c] += 1
                vehicle_speed_in[c].append(int(speed_kmh))
                direction = 'In'
                add_vehicle_info(vehicle_photo, str(c),int(speed_kmh), direction, str(violation_status), row_num)
     #counting vehicle from bottom to top
    if blue_line[0][1] < (cy+offset) and blue_line[0][1] > (cy-offset):
        if cx > blue_line[0][0] and cx < blue_line[1][0]:
            violation_status = is_violation(id, speed_kmh, x3, x4, y3, y4, frame)
            if id not in counter_out:
                row_num += 1
                counter_out.append(id)
                vehicle_count_out[c] += 1
                vehicle_speed_out[c].append(int(speed_kmh))
                direction = 'Out'
                add_vehicle_info(vehicle_photo, str(c),int(speed_kmh), direction, str(violation_status), row_num)
                
def draw(frame):
    global vehicle_count_in, vehicle_count_out
    cv2.line(frame, red_line[0], red_line[1], (255,0,0), 2) 
    cv2.line(frame, blue_line[0], blue_line[1], (0,0,255), 2)
            
    #vehicles count text location on frame
    green_color = (0, 128, 0)
    frame_width = frame.shape[1] 
    top_left_left = (0, 0)
    top_left_right = (frame_width , 0)
    green_color = (0, 128, 0)

    lines_in = len(vehicle_count_in) + 1  
    height_in = lines_in * 33 

    lines_out = len(vehicle_count_out) + 1 
    height_out = lines_out * 33  # each text line have height: 33px

    # update height for retangle
    bottom_right_left = (300, height_in)  
    bottom_right_right = (frame_width - 300, height_out)  

    # draw retangle with height was caculated
    cv2.rectangle(frame, top_left_left, bottom_right_left, green_color, -1)
    cv2.rectangle(frame, top_left_right, bottom_right_right, green_color, -1)

    cv2.putText(frame, 'IN:', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    count_text = str(len(counter_in))
    x_pos_for_count = frame.shape[1] - cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] - frame_width + 150
    cv2.putText(frame, count_text, (x_pos_for_count, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(frame, 'OUT:', (1210, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    count_text = str(len(counter_out))
    x_pos_for_count = frame.shape[1] - cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] - 140
    cv2.putText(frame, count_text, (x_pos_for_count, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

    y_offset = 60  
    for vehicle, count_in in vehicle_count_in.items():
        average_speed_in = calculate_average_speed(vehicle, 'In')

        vehicle = vehicle.capitalize()
        text_in = f'{vehicle} In: {count_in} ({int(average_speed_in)} Km/h)'
        cv2.putText(frame, f'{vehicle} In:', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        count_text = str(count_in)

        x_pos_for_count = frame.shape[1] - cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] - frame_width + 150
        cv2.putText(frame, count_text, (x_pos_for_count, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        x_pos_for_speed = x_pos_for_count + cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] + 10
        cv2.putText(frame, f'({int(average_speed_in)} Km/h)', (x_pos_for_speed, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        y_offset += 30  

    y_offset = 60  
    for vehicle, count_out in vehicle_count_out.items():
        average_speed_in = calculate_average_speed(vehicle, 'Out')

        vehicle = vehicle.capitalize()
        text_out = f'{vehicle} Out: {count_out}'
        cv2.putText(frame, f'{vehicle} Out:', (1210, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        count_text_out = str(count_out)

        x_pos_for_count_out = frame.shape[1] - cv2.getTextSize(count_text_out, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] - 140
        cv2.putText(frame, count_text_out, (x_pos_for_count_out, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        x_pos_for_speed = x_pos_for_count_out + cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] + 10
        cv2.putText(frame, f'({int(average_speed_in)} Km/h)', (x_pos_for_speed, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        y_offset += 30  



#location red line and blue line
red_line = [(0, 350), (700, 350)]
blue_line = (750, 350), (1500, 350)
offset = 6


class_list = read_classes_from_file('coco.txt')
max_speed_list = read_classes_from_file('speed.txt')
#init GUI by tkinter
root = tk.Tk()
root.title('Vehicle Monitoring System')

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

#root.attributes('-fullscreen', True)
top_container = tk.Canvas(root, width=screen_width, height=735)
top_container.grid(row=0, column=0, sticky='nsew')

video_canvas = tk.Canvas(top_container, width=1500, height=735, bg='black')
video_canvas.pack(side='left', fill='y')

violation_and_count_canvas = tk.Canvas(top_container, width=screen_width - 1500, height=735, bg='white')
violation_and_count_canvas.pack(side='right', fill='y')

counter_canvas = tk.Canvas(violation_and_count_canvas, width=screen_width - 1500, height=40, bg='darkgreen')
counter_canvas.pack(side='top', fill='x')  

count_text = tk.Label(counter_canvas, text='Number of violating vehicles : 0', font=('Helvetica', 15), fg='white', bg='darkgreen')
counter_canvas.create_window((screen_width - 1500) // 2, 20, window=count_text)

violation_canvas = tk.Canvas(violation_and_count_canvas, height=735 - 50)
violation_canvas.pack(side='top', fill='both', expand=True)  

vehicle_frame = tk.Frame(violation_canvas)
vehicle_frame.pack(fill='both', expand=True)

#Canvas include vehicle img and scrollbar
scrollable_canvas = tk.Canvas(vehicle_frame)
scrollable_canvas.pack(side='left', fill='both', expand=True)

img_frame = tk.Frame(scrollable_canvas)
scrollable_canvas.create_window((0, 0), window=img_frame, anchor='nw')

img_frame.update_idletasks()
scrollable_canvas.config(scrollregion=scrollable_canvas.bbox('all'))

violation_scrollbar = tk.Scrollbar(vehicle_frame, orient='vertical', command=scrollable_canvas.yview)
violation_scrollbar.pack(side='right', fill='y')  
scrollable_canvas.config(yscrollcommand=violation_scrollbar.set)

#Button canvas
button_canvas = tk.Canvas(root, width=screen_width, height=100)
button_canvas.grid(row=1, column=0, sticky='nsew')

button_frame = tk.Frame(button_canvas, width=screen_width, height=100)
button_frame.pack()

class_speed = tk.StringVar()
class_speed.set('60')
class_speed_label = tk.Label(button_frame, text='Max Speed:')
class_speed_label.pack(side='left')
class_speed_entry = tk.OptionMenu(button_frame, class_speed, '60', *max_speed_list)
class_speed_entry.pack(side='left')

class_selection = tk.StringVar()
class_selection.set('All')
class_selection_label = tk.Label(button_frame, text='Select Class:')
class_selection_label.pack(side='left')
class_selection_entry = tk.OptionMenu(button_frame, class_selection, 'All', *class_list)
class_selection_entry.pack(side='left')

play_button = tk.Button(button_frame, text='Open Webcam', command=start_webcam)
play_button.pack(side='left')

stop_button = tk.Button(button_frame, text='Stop', command=stop_webcam)
stop_button.pack(side='left')

file_button = tk.Button(button_frame, text='Select File', command=select_file)
file_button.pack(side='left')

pause_button = tk.Button(button_frame, text='Pause/Resume', command=pause_resume_video)
pause_button.pack(side='left')

quit_button = tk.Button(button_frame, text='Quit', command=quit_app)
quit_button.pack(side='left')

#Bottom canvas
bottom_container = tk.Canvas(root, width=screen_width, height=screen_height-835, bg='white')
bottom_container.grid(row=2, column=0, sticky='nsew')

traffic_analysis_canvas = tk.Canvas(bottom_container, width=1000, height=screen_height-835, bg='white')
traffic_analysis_canvas.pack(side='left', fill='both', expand=True)

traffic_analysis_chart = tk.Canvas(bottom_container, width=screen_width-1000, height=screen_height-835, bg='black')
traffic_analysis_chart.pack(side='right', fill='both', expand=True)

root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)
root.grid_columnconfigure(0, weight=1)


top_canvas = tk.Canvas(traffic_analysis_canvas,width=1000, height=100, bg='lightblue')
top_canvas.grid(row=0, column=0, sticky='nsew')

bottom_canvas = tk. Canvas(traffic_analysis_canvas,width=1000,height=252 , bg='black')
bottom_canvas.grid(row=1, column=0, sticky='nsew')

#Header in GUI
header_frame = tk.Label(top_canvas, bg='lightblue')
header_frame.pack(fill='both', expand=True)

vehicle_img_label = tk.Label(header_frame, text='VEHICLE IMAGE', font=('Helvetica', 10, 'bold'), fg='black', bg='lightblue')
vehicle_img_label.grid(row=0, column=0, padx=20)

vehicle_type_label = tk.Label(header_frame, text='VEHICLE TYPE', font=('Helvetica', 10, 'bold'), fg='black', bg='lightblue')
vehicle_type_label.grid(row=0, column=1, padx=50)

speed_info_label = tk.Label(header_frame, text='SPEED (Km/h)', font=('Helvetica', 10, 'bold'), fg='black',  bg='lightblue')
speed_info_label.grid(row=0, column=2, padx=50)

direction_label = tk.Label(header_frame, text='DIRECTION', font=('Helvetica', 10, 'bold'), fg='black',  bg='lightblue')
direction_label.grid(row=0, column=3, padx=50)

violation_status_label = tk.Label(header_frame, text='VIOLATION STATUS', font=('Helvetica', 10, 'bold'), fg='black',  bg='lightblue')
violation_status_label.grid(row=0, column=4, padx=40)

#Scroll
scrollable_frame_canvas = tk.Canvas(bottom_canvas, width=1000, height=252, bg='white')
scrollable_frame_canvas.pack(side='left', fill='both', expand=True)

scrollable_frame = tk.Frame(scrollable_frame_canvas)
scrollable_frame_canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')

scrollable_frame.update_idletasks()
scrollable_frame_canvas.config(scrollregion=scrollable_frame_canvas.bbox('all'))

scrollbar = tk.Scrollbar(bottom_canvas, orient='vertical', command=scrollable_frame_canvas.yview)
scrollbar.pack(side='right', fill='y')
scrollable_frame_canvas.configure(yscrollcommand=scrollbar.set)

#Chart Traffic Volume Over Time
chart_frame = tk.Frame(traffic_analysis_chart)  
chart_frame.pack(fill='both', expand=True)

fig, traffic_ax = plt.subplots(figsize=(5, 2))
fig.tight_layout()
traffic_canvas = FigureCanvasTkAgg(fig, master=chart_frame)
traffic_canvas.get_tk_widget().pack(fill='both', expand=True)

root.mainloop()
