from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy as np
import matplotlib.pyplot as plt
import time
import moviepy.editor as moviepy

model_dir = "./SenseVoiceSmall"

model = AutoModel(
    model=model_dir,
    # vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    # device="cuda:0",
)

# en
# res = model.generate(
#     input="wwdc2024-10183_hd.mp3",
#     cache={},
#     language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
#     use_itn=True,
#     batch_size_s=60,
#     merge_vad=True,  #
#     merge_length_s=15,
# )

def load_mp3(file_path):
    audio = AudioSegment.from_wav(file_path)
    audio = audio.set_frame_rate(16000).set_channels(1)  # FunASR 권장 스펙
    return audio

def stream_audio(audio, chunk_size_ms=5000):
    chunks = make_chunks(audio, chunk_size_ms)
    for chunk in chunks:
        yield chunk.raw_data  # 실시간 스트림처럼 chunk를 생성

def calculate_speech_rate(transcript, duration_sec):
    # words = transcript.split()
    words = transcript[0]["text"].split()
    word_count = len(words)
    speech_rate = word_count / duration_sec * 60  # 분당 단어 수로 계산
    return speech_rate


def real_time_speech_rate(file_path):
    data = []
    audio = load_mp3(file_path)
    i = 0
    for chunk_data in stream_audio(audio):
        # 이 부분에서 FunASR STT 사용하여 텍스트 변환
        transcript = model.generate(input=chunk_data, cache={}, language="auto", use_itn=True, batch_size_s=60, merge_vad=True, merge_length_s=15)
        # print(transcript[0]["text"])
        text = rich_transcription_postprocess(transcript[0]["text"])
        # 일단 가정으로 아래와 같은 dummy text 사용
        # transcript = "이것은 샘플 텍스트입니다."  # 변환된 텍스트 (dummy 예시)

        duration_sec = 5  # 현재 chunk 길이 (1초 단위)
        speech_rate = calculate_speech_rate(transcript, duration_sec)

        emo = transcript[0]["text"].split('<|')[2][:-2]
        # print(f"현재 말의 감정: {emo}")

        start_time = i * duration_sec
        end_time = (i + 1) * duration_sec

        # print(f"현재 말의 빠르기 (WPM): {speech_rate:.2f}")

        data.append({
            "start_time": start_time,
            "end_time": end_time,
            "speech_rate": speech_rate,
            "emotion": emo,
            "transcript": text
        })

        # 주기적으로 업데이트
        # time.sleep(duration_sec)
        i += 1
        # if i == 100:
        #     break

    return data

# text = rich_transcription_postprocess(res[0]["text"])
# print(text)

# file_path = 'wwdc2024-10183_hd.mp3'  # 분석할 MP3 파일 경로
# file_path = 'korean_sample.mp3'
# real_time_speech_rate(file_path)
#
# for i in data:
#     print(i)
#
# # graph of speech rate
# x = [i["start_time"] for i in data]
# y = [i["speech_rate"] for i in data]
#
# plt.plot(x, y)
# plt.xlabel('Time (s)')
# plt.ylabel('Speech Rate (WPM)')
# plt.title('Speech Rate Over Time')
# plt.show()