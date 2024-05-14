import pika
import cv2
import numpy as np
import os
import threading
import base64
import time
import requests
import re
global transcription, boxes ,tracker_flag
coordinates = None
boxes = None
tracker_flag = False

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

channel.queue_delete(queue='video_frames')
channel.queue_delete(queue='transcription')

channel.queue_declare(queue='video_frames')
channel.queue_declare(queue='transcription')
channel.queue_declare(queue='audio')
url = 'http://128.205.43.182:5000/chat'  # URL of the server's chat endpoint



# def conversion(boxes):
#   if len(boxes) == 4:
#     cx,cy,w,h = boxes
#     x1 = int((cx - w / 2) * frame.shape[1])
#     y1 = int((cy - h / 2) * frame.shape[0])
#     x2 = int((cx + w / 2) * frame.shape[1])
#     y2 = int((cy + h / 2) * frame.shape[0])
#     print(x1,y1,x2,y2)
#     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.imshow("bounding box",frame)



def dino_chat(frame_path, transcription, url, channel):
    global boxes,tracker_initialized
    with open(frame_path, 'rb') as f:
        image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        data = {'image_base64': image_base64, 'prompt': transcription}
        response = requests.post(url, json=data)
        if response.status_code == 200:
            received_json = response.json()['answer']
            if received_json:  
                print(f"Received from server: {received_json}")
                boxes = received_json[0]
                tracker_initialized = False
                print("dino flag: ",tracker_initialized)
                print(boxes, type(boxes))
            else:
                print("Received empty response")
        else:
            print('Failed to send request. Status code:', response.status_code)


def consume_transcription():    
    def transcription_callback(ch, method, properties, body):
        global transcription
        transcription = body.decode('utf-8').strip()
        phrases_LLAVA = ["what do you see", "capture an image", "describe what do you see"]
        phrases_DINO = ["locate", "track", "go to"]
        if any(phrase_LLAVA in transcription.lower() for phrase_LLAVA in phrases_LLAVA):
            print("start chat")
            channel.basic_publish(exchange='', routing_key='audio', body="Sure, give me a minute to process")
            send_frame_and_transcription('tmp/frame.jpg', transcription, url, channel)
        elif any(phrase_DINO in transcription.lower() for phrase_DINO in phrases_DINO):
            print("DINO chat")
            dino_chat('tmp/frame.jpg', transcription, url, channel)
        else:
            print("continue chat")
            send_transcription(transcription, url, channel)
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='transcription')
    channel.basic_consume(queue='transcription', on_message_callback=transcription_callback, auto_ack=False)
    print("Waiting for transcription... Press 'q' to exit")
    channel.start_consuming()

def send_frame_and_transcription(frame_path, transcription, url, channel):
    frame = cv2.imread(frame_path)
    if frame is not None:
        _, image_data = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        data = {'image_base64': image_base64, 'prompt': transcription}
        response = requests.post(url, json=data)
        response_str = str(response.json()['answer'])
        if response.status_code == 200:
            print(f"Received from server: {response.json()['answer']}")
            channel.queue_declare(queue='audio')
            channel.basic_publish(exchange='', routing_key='audio', body=response_str)
        else:
            print('Failed to send request. Status code:', response.status_code)

def send_transcription(transcription, url, channel):
    data = {'prompt': transcription}
    response = requests.post(url, json=data)
    response_str = str(response.json()['answer'])
    if response.status_code == 200:
        print(f"Received from server: {response.json()['answer']}")
        channel.queue_declare(queue='audio')
        channel.basic_publish(exchange='', routing_key='audio', body=response_str)
    else:
        print('Failed to send request. Status code:', response.status_code)

def consume_video():
    global tracker_flag, tracker_initialized
    tracker = cv2.TrackerCSRT_create() 
    tracker_initialized = False 

    def video_callback(ch, method, properties, body):
        global frame, boxes, tracker_flag, tracker_initialized
        try:
            frame_bytes = np.frombuffer(body, dtype=np.uint8)
            frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)            
            if boxes is not None and not tracker_initialized:
                print("Initializing tracker with boxes")
                cx, cy, w, h = boxes
                x1 = int((cx - w / 2) * frame.shape[1])
                y1 = int((cy - h / 2) * frame.shape[0])
                x2 = int((cx + w / 2) * frame.shape[1])
                y2 = int((cy + h / 2) * frame.shape[0])
                tracker_initialized = True
                tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
            if tracker_initialized:
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = [int(i) for i in bbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.imshow("track",frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("reassigning tracker")
                        boxes = None
                        bbox = None
                        tracker_initialized = False
                cv2.imwrite('tmp/frame_annotated_new.jpg', frame)
            if not os.path.exists('tmp'):
                os.makedirs('tmp')
            cv2.imwrite('tmp/frame.jpg', frame)
        except Exception as e:
            print(f"Error handling frame: {str(e)}")
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='video_frames')
    channel.basic_consume(queue='video_frames', on_message_callback=video_callback, auto_ack=True)
    print("Waiting for video frames... Press 'q' to exit")
    channel.start_consuming()

video_thread = threading.Thread(target=consume_video)
video_thread.start()

transcription_thread = threading.Thread(target=consume_transcription)
transcription_thread.start()

video_thread.join()
transcription_thread.join()
