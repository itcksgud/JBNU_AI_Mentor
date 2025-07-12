from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import requests
import json

class Pipeline:
    class Valves(BaseModel):
        # LLM 에이전트(8001번 포트) URL / 타임아웃 설정
        agent_url: str = "http://host.docker.internal:8001/mock"
        timeout: float = 200.0

    def __init__(self):
        # 파이프라인 이름 설정
        self.name = "File_pipeline"
        self.valves = self.Valves()

    async def on_startup(self):
        # 파이프라인 시작 시 호출되는 함수
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        # 파이프라인 종료 시 호출되는 함수
        print(f"on_shutdown:{__name__}")

    def pipe(
        self,
        user_message: str, # 사용자로부터 입력된 질문
        model_id: str, # 모델 ID
        messages: List[dict], # 전체 대화 히스토리
        body: dict # 요청 본문
    ) -> Union[str, Generator, Iterator]:
        
        # 1) UI에서 입력된 질문을 그대로 전달위해 payload 생성
        payload = {
            "stream": body.get("stream", False),
            "model": model_id,
            "messages": messages  # 전체 히스토리 그대로 전달
        }

        try:
            r = requests.post(
                self.valves.agent_url,
                json=payload,
                timeout=self.valves.timeout
            )
            r.raise_for_status()

            if body["stream"]:
                return r.iter_lines()
            else:
                return r.json()

        except Exception as e:
            return f"LLM Agent error: {e}"
