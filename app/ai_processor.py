import json

import httpx
from bs4 import BeautifulSoup
from faker import Faker
from googlesearch import search
from openai import OpenAI
from tenacity import retry, stop_after_attempt

from app.config import settings
from app.dto.request import AgentResponse, PredictionRequest, PredictionResponse
from app.app_logger import logger


class AIQueryProcessor:
    def __init__(self):
        self._faker = Faker()
        self._ai_client = OpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_ENDPOINT)
        self._model_name = settings.OPENAI_MODEL_NAME
        self._system_instruction = settings.SYSTEM_INSTRUCTION

    async def process_request(self, request: PredictionRequest) -> PredictionResponse:
        question, options = (request.query.split('\n1.', 1) + [""])[:2]
        logger.debug(f'Processing query: {request.query}, Parsed question: {question}, Options: {options}')

        sources = await self._find_sources(question, max_results=1)
        logger.debug(f'Found sources: {sources}')

        web_content = await self._aggregate_web_content(sources)
        prompt = await self._generate_prompt(question, options, web_content)

        ai_output = await self._query_ai_model(prompt)
        logger.debug(f'AI Output: {ai_output}')

        response = PredictionResponse(
            id=request.id,
            answer=None if not options else ai_output.answer,
            reasoning=ai_output.reasoning + self._append_signature(),
            sources=sources
        )
        logger.debug(f'Final Response: {response}')
        return response

    async def _generate_prompt(self, question: str, options: str, context: str) -> str:
        return f'''
        Вопрос: {question}\n
        Выберите правильный вариант:
        {options}\n
        Формат ответа:
        {{ "answer": номер варианта, "reasoning": объяснение, "sources": [список источников] }}
        Если вариантов нет, укажите "answer": null.

        Информация из веб-источников:
        {context}
        '''

    @retry(stop=stop_after_attempt(3))
    async def _fetch_webpage(self, url: str) -> str:
        headers = {"User-Agent": self._faker.user_agent()}
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            return BeautifulSoup(response.text, "html.parser").get_text(separator="\n", strip=True)

    async def _aggregate_web_content(self, urls: list[str]) -> str:
        content = []
        for url in urls:
            try:
                content.append(await self._fetch_webpage(url))
            except Exception as e:
                logger.error(f'Failed to retrieve {url}: {e}')
        return "\n\n".join(content)

    async def _find_sources(self, query: str, max_results: int = 3) -> list[str]:
        return [url for url in search(self._sanitize_query(query), num_results=max_results)]

    def _sanitize_query(self, text: str) -> str:
        return text.split("\n1.", 1)[0]

    @retry(reraise=True, stop=stop_after_attempt(3))
    async def _query_ai_model(self, prompt: str) -> AgentResponse:
        response = self._ai_client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": self._system_instruction},
                {"role": "user", "content": prompt},
            ]
        )
        logger.debug(f"AI Response: {response}")
        parsed_content = response.choices[0].message.content.strip().replace("```json", '').replace("```", '')
        return AgentResponse(**json.loads(parsed_content))

    def _append_signature(self) -> str:
        return f"\nОтвет сформирован с использованием модели {self._model_name}"
