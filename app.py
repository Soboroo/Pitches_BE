import os
import time

from flask import Flask, request
from werkzeug.utils import secure_filename
from speech_speed_offline import real_time_speech_rate
import json
import threading
from openai import OpenAI
from flask_cors import CORS
import subprocess


app = Flask(__name__)
CORS(app, resources={r'*': {'origins': '*'}})

audio_path = "audio"

# 파일 업로드
@app.route('/fileupload', methods=['POST'])
def file_upload():
    print(request.content_type)
    file = request.files['file']

    filename = secure_filename(file.filename)
    # filename + timestamp
    filename = filename.split(".")[0] + "_" + str(int(time.time())) + "." + filename.split(".")[1]
    os.makedirs(audio_path, exist_ok=True)
    file.save(os.path.join(audio_path, filename))

    wav_filename = filename.split(".")[0] + ".wav"
    wav_path = os.path.join(audio_path, wav_filename)
    convert_webm_to_wav(os.path.join(audio_path, filename), wav_path)
    # return json.dumps(real_time_speech_rate(os.path.join(audio_path, filename)))


    # return json.dumps(real_time_speech_rate(os.path.join(audio_path, filename)))

    # analyse audio file
    # data = real_time_speech_rate(os.path.join(audio_path, filename))

    thread = threading.Thread(target=create_report, args=(os.path.join(audio_path, wav_filename),))
    thread.start()

    # save result as json file
    result_path = os.path.join(audio_path, filename.split(".")[0] + ".json")

    # with open(result_path, "w") as f:
    #     json.dump(data, f, indent=4)

    return result_path

def convert_webm_to_wav(input_file, output_file):
    command = [
        "ffmpeg",
        "-i", input_file,     # 입력 파일
        "-acodec", "pcm_s16le",  # 오디오 코덱 설정
        "-ar", "44100",        # 샘플링 레이트 (Hz)
        "-ac", "2",            # 채널 수
        output_file
    ]

    # ffmpeg 명령 실행
    try:
        subprocess.run(command, check=True)
        print(f"변환이 완료되었습니다: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"오류 발생: {e}")



def create_report(file_path):
    # 파일 경로를 받아서 해당 파일의 분석 결과를 반환
    data = real_time_speech_rate(file_path)

    # 결과를 JSON 파일로 저장
    result_path = file_path.split(".")[0] + ".json"
    with open(result_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"분석 결과 저장 완료: {result_path}")


@app.route('/report', methods=['GET'])
def report():
    # query string으로 파일명을 받아서 해당 파일의 분석 결과를 반환
    filename = request.args.get('filename')
    # result_path = os.path.join(audio_path, filename.split(".")[0] + ".json")
    #
    # with open(result_path, "r") as f:
    #     data = json.load(f)
    #
    # return data

    # mp3 파일 존재 여부 확인
    # mp3 파일이 있지만 json 파일이 없는 경우 처리가 끝나지 않았다는 것을 의미
    wav_path = os.path.join(audio_path, filename + ".wav")
    json_path = os.path.join(audio_path, filename + ".json")

    if os.path.exists(wav_path) and not os.path.exists(json_path):
        return {"status": "processing"}
    elif os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        return data
    else:
        return {"status": "error"}

@app.route('/gpt', methods=['POST'])
def gpt_route():
    # return handleChat(request)
    client = OpenAI(api_key='<open-ai-key>')

    assistantId = '<assistant-id>'

    try:
        userQuestion = request.form['stt_script'] + '\n' + request.form['emotion_data']
        # userQuestion = req.body.question
        threadId = ''
        # threadId = req.body.thread_id

        if threadId == "":
            thread = client.beta.threads.create()
            threadId = thread.id

            client.beta.threads.messages.create(threadId, role="user", content= userQuestion)

            run = client.beta.threads.runs.create_and_poll(thread_id = threadId, assistant_id = assistantId)

            run_status = client.beta.threads.runs.retrieve(thread_id = threadId, run_id = run.id)

            while run_status.status != "completed":
                time.sleep(1)
                run_status = client.beta.threads.runs.retrieve(thread_id = threadId, run_id = run.id)

                if run_status.status in ["failed", "cancelled", "expired"]:
                    print(f"Run status: '{run_status.status}', unable to execute the request.")
                    # return {"status": 500, "message": "Assistant execution failed"}
                    return {"status": "error"}

            messages = client.beta.threads.messages.list(threadId)
            last_message_for_run = next(
                (message for message in messages.data if message.run_id == run.id and message.role == "assistant"),
                None
            )

            if last_message_for_run:
                print(last_message_for_run.content[0].text.value)
                return {
                    "response": last_message_for_run.content[0].text.value
                }
            else:
                # return {"status": 500, "message": "Assistant did not provide a response."}
                return {"status": "error"}

    except Exception as error:
        print(error)
        # return {"status": 500, "message": "An error occurred"}
        return {"status": "error"}