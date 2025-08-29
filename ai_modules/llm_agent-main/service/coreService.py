import json
import requests
import re
import tiktoken 

from util.llmClient import LlmClient
from util.dbClient import DbClient

class CoreService():
    
    def __init__(self):
        self.llmClient = LlmClient()
        self.dbClient = DbClient()
        self.dbClient.connect()

        self.prompt_tools = [
            {"tool_name": "FINISH", "tool_description": "If the user's question is already fully answered, reply with FINISH.", "api_key": "FINISH", "api_value": "<FINISH>", "api_url": "FINISH"},
            {"tool_name": "JBNU_SQL", "tool_description": "Accepts a natural‐language query about JBNU (colleges, departments, courses, professors) and returns the corresponding data. Do not send raw SQL.", "api_key": "query", "api_value": "<USER_QUESTION>", "api_url": "http://localhost:7999/agent"},
            {"tool_name": "FALLBACK", "tool_description": "Use for general questions, or questions that can be answered from the conversation history (e.g., 'what did I just ask?', 'what is my name?').", "api_key": "query", "api_value": "<USER_QUESTION>", "api_url": "http://localhost:7998/agent"},
            {"tool_name": "VECTOR_SEARCH", "tool_description": "Recommends courses similar to a given course name based on vector similarity. The 'count' defaults to 5 if the user does not specify a number.", "api_key": "count, key", "api_value": "<INTEGER_DEFAULT_5>, <COURSE_NAME>", "api_url": "http://localhost:7997/search"},
            {"tool_name": "CURRICULUM_RECOMMEND", "tool_description": "Use this tool when the user asks for a 'learning path', 'curriculum', 'study plan', or describes a long-term learning goal. It accepts a natural-language goal and returns a personalized curriculum recommendation.", "api_key": "query", "api_value": "<USER_GOAL>", "api_url": "http://localhost:7996/chat"}
        ]

        self.tool_urls = {t['tool_name']: t.get('api_url', t['tool_name']) for t in self.prompt_tools}

    def _trim_messages_to_fit_token_limit(self, messages: list, max_tokens: int = 8000):
        """메시지 리스트를 최신순으로 max_tokens에 맞게 잘라냅니다."""
        try:
            encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        token_count = 0
        trimmed_messages = []

        for msg in reversed(messages):
            # ★★★ 수정됨: 딕셔너리 접근(msg['...'])을 객체 속성 접근(msg....)으로 변경 ★★★
            msg_tokens = len(encoding.encode(msg.role)) + len(encoding.encode(msg.content)) + 10
            
            if token_count + msg_tokens > max_tokens:
                break
            
            token_count += msg_tokens
            trimmed_messages.append(msg)
        
        return list(reversed(trimmed_messages))
    
    def _build_api_body(self, api_key_str: str, api_value_str: str) -> dict:
        """
        LLM이 생성한 "key1, key2" 와 "value1, value2" 문자열을
        {"key1": "value1", "key2": value2} 딕셔너리로 파싱합니다.
        """
        api_body = {}
        if not all(isinstance(s, str) for s in [api_key_str, api_value_str]):
            return api_body

        keys = [k.strip() for k in api_key_str.split(',')]
        values = [v.strip() for v in api_value_str.split(',')]

        if len(keys) != len(values):
            print(f"Warning: Mismatched keys and values. Treating as single query.")
            return {"query": api_value_str}

        for key, value in zip(keys, values):
            # 값이 정수 형태이면 int로 변환
            if value.isdigit():
                api_body[key] = int(value)
            else:
                api_body[key] = value
        
        return api_body


    # --------------------------------------------------------------------------
    # 1. 라우터 프롬프트 (첫 번째 LLM 호출용)
    #    - 작업이 단일 단계로 끝날지, 여러 단계가 필요할지 판단
    # --------------------------------------------------------------------------
    ROUTER_SYSTEM_PROMPT = """
You are an expert planning agent. Your primary job is to analyze the user's **ultimate goal** from their conversation history and decide on a precise plan.

--- RULES ---
1.  **Analyze the user's ultimate goal.** Do not get distracted by the length or complexity of their question.
2.  **Crucial Rule:** If the user asks for a "learning path," "curriculum," "study plan," "learning process," or describes a long-term learning goal, you **MUST** classify it as a `single_step` and use the `CURRICULUM_RECOMMEND` tool.
3.  Choose a plan:
    * `single_step`: Use this if the user's final goal can be achieved with **one single tool call**.
    * `multi_step`: Use this **only if** the user explicitly asks for **multiple, separate pieces of information** that require a sequence of tool calls (e.g., "Find X, **and then** find Y").

--- RESPONSE FORMAT ---
You must return a single, valid JSON object.
- For `single_step`, you must provide "tool_name", "api_key", and "api_value".
- For `multi_step`, you only need to provide "plan_type" and "reason".

--- EXAMPLES ---

Example 1 (Curriculum Request - IMPORTANT):
User Query: "I want to learn computer graphics and 3D design to build city simulations. What learning process should I follow?"
Your JSON Response:
{{
    "plan_type": "single_step",
    "tool_name": "CURRICULUM_RECOMMEND",
    "api_key": "query",
    "api_value": "A learning path for building city simulations using computer graphics and 3D design.",
    "reason": "The user is asking for a detailed learning path, which directly maps to the CURRICULUM_RECOMMEND tool."
}}

Example 2 (Simple DB Query):
User Query: "Tell me about the computer science department."
Your JSON Response:
{{
    "plan_type": "single_step",
    "tool_name": "JBNU_SQL",
    "api_key": "query",
    "api_value": "Information about the computer science department.",
    "reason": "The user is asking for a single piece of information about a department."
}}

Example 3 (Vector Search with count):
User Query: "Recommend 3 courses similar to 'Database'."
Your JSON Response:
{{
    "plan_type": "single_step",
    "tool_name": "VECTOR_SEARCH",
    "api_key": "key, count",
    "api_value": "Database, 3",
    "reason": "The user specifically asked for 3 courses similar to 'Database'."
}}

Example 4 (Multi-Step Request):
User Query: "Find the department for Professor Kim, and then list the courses offered by that department."
Your JSON Response:
{{
    "plan_type": "multi_step",
    "reason": "The user has two distinct goals that must be executed in sequence: first find the professor's department (JBNU_SQL), and then find the department's courses (another JBNU_SQL call)."
}}

--- AVAILABLE TOOLS ---
{tools}

Now, analyze the conversation and create your plan.
"""

    # --------------------------------------------------------------------------
    # 2. ReAct 에이전트 프롬프트 (여러 번 돌리는 경우에만 사용)
    # --------------------------------------------------------------------------

    REACT_AGENT_SYSTEM_PROMPT = """
You are a sophisticated AI agent that reasons step-by-step to answer a user's request.
Based on the user's conversation and your previous actions and observations (your scratchpad), you must decide on the next single action to take.

You have a maximum of 5 steps to gather all necessary information. You must use the "FINISH" tool by step 5.

Available Tools:
{tools}

To decide your next action, you MUST respond with a single JSON object with the following keys:
- "thought": Your reasoning for choosing the next tool. Briefly explain what you know and what you need to find out next.
- "tool_name": The name of the tool to use. If you have gathered all necessary information, use "FINISH".
- "api_key": The key(s) for the chosen tool's API.
- "api_value": The value(s) for the chosen tool's API.

Analyze the scratchpad below and determine your next action.

--- SCRATCHPAD ---
{scratchpad}
"""


    def run_agent(self, messages: list):
        """
        메인 에이전트 실행 함수 (새로운 진입점)
        1. 라우터를 호출하여 계획 수립
        2. 계획에 따라 단일 실행 또는 멀티스텝 체인 실행
        """
        print("Step 1: Routing Query...")
        trimmed_messages = self._trim_messages_to_fit_token_limit(messages)
        routing_decision = self._route_query(trimmed_messages)
        plan_type = routing_decision.get("plan_type")

        if plan_type == "single_step":
            print("Plan: Single step. Executing directly.")
            print(f"tool_name:{routing_decision.get('tool_name')}, reason:{routing_decision.get('reason')}")
            tool_name = routing_decision.get("tool_name")
            api_key = routing_decision.get("api_key")
            api_value = routing_decision.get("api_value")
            api_body = self._build_api_body(api_key, api_value)

            tool_result_json = self._execute_single_tool(tool_name, api_key, api_value, trimmed_messages)


            history_data = {
                "steps": [
                    {
                        "step_number": 1,
                        "tool_name": tool_name,
                        "tool_input": api_body,
                        "tool_response": tool_result_json.get('message') or json.dumps(tool_result_json),
                        "reason": routing_decision.get('reason', 'Single step execution')
                    }
                ]
            }

            return history_data
        elif plan_type == "multi_step":
            print("Plan: Multi step. Starting ReAct chain.")
            initial_reason = routing_decision.get("reason", "No initial plan provided.")
            return self._execute_multi_step_chain(trimmed_messages, initial_reason)
        else:
            print("Error: Could not determine a plan. Falling back to default.")
            tool_name = "FALLBACK"
            tool_result_json = self._execute_single_tool(tool_name, None, None, trimmed_messages)

            history_data = {
                "steps": [
                    {
                        "step_number": 1,
                        "tool_name": tool_name,
                        "tool_input": {"query": messages[-1].content},
                        "tool_response": tool_result_json.get('message') or json.dumps(tool_result_json),
                        "reason": "Router failed, executing fallback."
                    }
                ]
            }
            return history_data

    def _route_query(self, messages: list):
        """(내부 함수) 첫 번째 LLM 호출: 어떤 계획을 사용할지 결정"""
    
        system_prompt_for_router = self.ROUTER_SYSTEM_PROMPT.format(
            tools=json.dumps(self.prompt_tools, ensure_ascii=False, indent=2)
        )


        # ★★★ 새로운 방식으로 LLM 호출 ★★★
        request_messages = [
            {"role": "system", "content": system_prompt_for_router},
            # client_question_str_list의 각 요소를 user 메시지로 변환
            *({"role": msg.role, "content": msg.content} for msg in messages)
        ]
        # json_mode=True로 설정하여 안정적으로 JSON을 받음
        resp = self.llmClient.call_llm(request_messages, json_mode=True)
        raw = resp.choices[0].message.content.strip()

        try:
            # 이제 re.search가 필요 없을 가능성이 높지만, 안전을 위해 유지
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"plan_type": "fallback", "reason": "Failed to parse router response."}
        # json_mode=True로 설정하여 안정적으로 JSON을 받음


    def _execute_single_tool(self, tool_name: str, api_key: str, api_value: str, messages: list):
        """(내부 함수) 단일 도구를 직접 실행하고 결과를 반환"""
        if tool_name not in self.tool_urls:
            return {"error": f"Tool '{tool_name}' not found."}

        url = self.tool_urls[tool_name]

        if tool_name == "FALLBACK":
            body_to_send = {"messages": [m.dict() if hasattr(m, 'dict') else m for m in messages]}
        else:
            body_to_send = self._build_api_body(api_key, api_value)

        print(f"Executing single tool: {tool_name} with body: {body_to_send}")
        
        try:
            response = requests.post(url, json=body_to_send)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def _execute_multi_step_chain(self, messages: list, initial_reason: str):
        print("\n--- Starting ReAct Chain with Scratchpad ---")
        
        # 1. 스크래치패드 초기화: 최초 대화 기록을 담습니다.
        scratchpad = "Conversation History:\n"
        scratchpad += "".join([f"- {msg.role}: {msg.content}\n" for msg in messages])

        scratchpad += f"Initial Plan: {initial_reason}\n"
        
        history_data = {"steps": []}
        step_number = 1


        while step_number <= 5:
            print(f"\n--- ReAct Step {step_number} ---")
            
            system_prompt = self.REACT_AGENT_SYSTEM_PROMPT.format(
                tools=json.dumps(self.prompt_tools, ensure_ascii=False, indent=2),
                scratchpad=scratchpad
            )
            request_messages = [{"role": "system", "content": system_prompt}]
            
            resp = self.llmClient.call_llm(request_messages, json_mode=True)
            raw = resp.choices[0].message.content.strip()
            
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                print("Error: Could not parse LLM thought process. Finishing.")
                break

            thought = data.get("thought", "").strip()
            tool_name = data.get("tool_name", "FINISH").strip()
            api_key = data.get("api_key")
            api_value = data.get("api_value")
            
            print(f"Thought: {thought}")

            scratchpad += f"\nStep {step_number}:\nThought: {thought}\n"

            if tool_name == "FINISH":
                print("FINISH signal received. Proceeding to final summarization.")
                break

            # 4. '행동' 기록 및 실행
            api_body = self._build_api_body(api_key, api_value)
            scratchpad += f"Action: Using tool '{tool_name}' with input {json.dumps(api_body, ensure_ascii=False)}\n"
            print(f"Action: Using tool '{tool_name}' with input {api_body}")

            tool_result_json = self._execute_single_tool(tool_name, api_key, api_value, messages)
            tool_result = tool_result_json.get('message') or json.dumps(tool_result_json)

            # 5. '관찰' 기록
            scratchpad += f"Observation: {tool_result}\n"
            print(f"Observation: {tool_result}")

            history_data["steps"].append({
                "step_number": step_number, "tool_name": tool_name,
                "tool_input": api_body, "tool_response": tool_result, "reason": thought
            })
            step_number += 1

        # 6. 최종 요약 단계
        print("\n--- Final Summarization Step ---")
        summarizer_prompt = f"""
        Based on the entire scratchpad provided below, generate a final, comprehensive answer for the user in Korean.
        Synthesize all the observations into a natural, helpful response.

        --- SCRATCHPAD ---
        {scratchpad}
        """
        # FALLBACK 도구는 대화 기록을 받으므로, 요약 프롬프트를 새 대화처럼 구성
        summarizer_messages = [{"role": "user", "content": summarizer_prompt}]
        
        # FALLBACK 도구 호출 (api_key, api_value는 None으로 전달)
        final_response_json = self._execute_single_tool("FALLBACK", None, None, summarizer_messages)
        final_answer = final_response_json.get("message", "답변을 종합하는 데 실패했습니다.")
        
        # 최종 답변을 history_data에 추가
        history_data["steps"].append({
            "step_number": "Final", "tool_name": "Summarizer(FALLBACK)",
            "tool_input": "Final Scratchpad", "tool_response": final_answer, "reason": "Synthesizing final answer."
        })

        return history_data
            