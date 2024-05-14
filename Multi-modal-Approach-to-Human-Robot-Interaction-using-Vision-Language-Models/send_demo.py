# vim: set fileencoding=utf-8 :
import sys
import pika
import numpy as np
from naoqi import ALProxy
import cv2
import threading
import time

exit_flag = False

ip_addr = "192.168.0.22"
port_num = 9559

motion = ALProxy("ALMotion", ip_addr, port_num)
speechClient = ALProxy("ALTextToSpeech", ip_addr, port_num)
motion.moveInit()

names = "HeadPitch"  # Select the joint for head yaw movement
angles = [1.0, -1.0]  # Define angles for left and right movement
fractionMaxSpeed = 0.1 

awareness = ALProxy("ALBasicAwareness",ip_addr,port_num)
posture = ALProxy("ALRobotPosture",ip_addr,port_num)

awareness.setEnabled(False)

time.sleep(2)
print("Stopping awareness")


posture.goToPosture("StandInit",0.5)
motion.setAngles(names, 0.0, fractionMaxSpeed)


width = 320
height = 240
image = np.zeros((height, width, 3), np.uint8)


videoClient = ALProxy("ALVideoDevice", ip_addr, port_num)
cameraID = 0  # 0 for top camera
resolution = 1  # 1 for QVGA (320x240)
colorSpace = 13  # BGR
captureDevice = videoClient.subscribeCamera("pepper_send12", cameraID, resolution, colorSpace, 10)


connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_delete(queue='audio')
channel.queue_delete(queue='video_frames')
channel.queue_delete(queue='transcription')

channel.queue_declare(queue='video_frames')
channel.queue_declare(queue='transcription')  # Declare the transcription queue
channel.queue_declare(queue='audio')


print("Ready to capture")

def process_audio(audio):
    # Process the received transcription and use text-to-speech
    speechClient.say(audio)

def consume_audio():
    def audio_callback(ch, method, properties, body):
        global audio
        print(body)
        audio = body.decode('utf-8')
        print(audio)
        process_audio(audio)

    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='audio')
    channel.basic_consume(queue='audio', on_message_callback=audio_callback, auto_ack=False)
    print("Waiting for audio... Press 'q' to exit")

    # Start consuming messages
    channel.start_consuming()


def send_frames():
    # Capture image
    channel.queue_declare(queue='video_frames')
    
    while True:
        result = videoClient.getImageRemote(captureDevice)

        if result == None:
            print('cannot capture.')
        elif result[6] == None:
            print('no image data string.')
        else:
            values = map(ord, list(result[6]))
            i = 0
            for y in range(0, height):
                for x in range(0, width):
                    image.itemset((y, x, 0), values[i + 0])
                    image.itemset((y, x, 1), values[i + 1])
                    image.itemset((y, x, 2), values[i + 2])
                    i += 3
        
        frame_data = cv2.imencode('.jpg', image)[1].tobytes()

        channel.basic_publish(exchange='', routing_key='video_frames', body=frame_data)
        cv2.imshow('image', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    connection.close()

def audio_callback(ch, method, properties, body):
    print("received")
    transcription = body.decode('utf-8')
    print(transcription)

def main():
    global exit_flag

    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_delete(queue='audio')
    channel.queue_declare(queue='audio')
    frames_thread = threading.Thread(target=send_frames)
    frames_thread.start()
    # Continuously listen for commands
    while not exit_flag:
        try:
            method_frame, header_frame, body1 = channel.basic_get(queue='audio', auto_ack=True)
            if body1:
                print("received")
                transcription = body1.decode('utf-8')
                transcription = str(transcription)
                print(transcription)
                process_audio(transcription)
        except pika.exceptions.StreamLostError as e:
            # Handle the stream lost error, e.g., reconnect
            print("StreamLostError: Reconnecting...")
            connection.close()
            connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
            channel = connection.channel()
            channel.queue_declare(queue='audio')
            channel.basic_consume(queue='audio', on_message_callback=audio_callback, auto_ack=True)
            channel.start_consuming()

    # Set the exit flag to True when the main thread is ready to exit
    exit_flag = True

if __name__ == "__main__":
    main()
