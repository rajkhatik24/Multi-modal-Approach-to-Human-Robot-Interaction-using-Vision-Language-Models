from faster_whisper import WhisperModel
import pvporcupine 
import struct
import pyaudio
import pika
import numpy as np
import wave
import os


connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='transcription')


def record_chunk(p, stream, file_path, chunk_length=2):
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()


def transcription_chunk(model, file_path):
    global transcription
    segments, info = model.transcribe(file_path, beam_size=7)
    transcription = ''.join(segment.text for segment in segments)
    return transcription
def main():
    porcupine = None
    pa = None
    audio_stream = None
    model_size = "medium.en"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    try:
        porcupine = pvporcupine.create(
            access_key='l5uWEn+O7oDa56pBaShL80ZUOYgGOx2Bev4AINoTsZncEoRXn/lu/Q==',
            keywords=['jarvis']
        )

        pa = pyaudio.PyAudio()
        audio_stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length)

        print("Listening... Press Ctrl+C to exit.")
        while True:
            pcm = audio_stream.read(porcupine.frame_length)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

            result = porcupine.process(pcm)
            if result >= 0:
                print("Wake word 'jarvis' detected!")
                chunk_file = "temp_chunk.wav"
                record_chunk(pa, audio_stream, chunk_file)
                transcription = transcription_chunk(model, chunk_file)
                print("Transcription:", transcription)
                channel.basic_publish(exchange='', routing_key='transcription', body=transcription)
                os.remove(chunk_file)

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        if porcupine is not None:
            porcupine.delete()

        if audio_stream is not None:
            audio_stream.close()

        if pa is not None:
            pa.terminate()

if __name__ == "__main__":
    main()
