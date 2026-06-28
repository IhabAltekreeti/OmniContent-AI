import time
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import LLMChain

from omnicontent.config import OPENROUTER_KEY, LLM_MODEL, LLM_BASE_URL, get_logger

log = get_logger("pipeline.agents")

llm = ChatOpenAI(
    model=LLM_MODEL,
    openai_api_key=OPENROUTER_KEY,
    openai_api_base=LLM_BASE_URL,
)

log.info("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

_knowledge_base = [
    Document(page_content="Fitness content gets 40% more engagement when posted between 7-9 AM."),
    Document(page_content="Organic reach peaks on social media when using 3-5 relevant hashtags."),
    Document(page_content="The ideal video duration for TikTok and Reels is 15-30 seconds. Catch attention in the first 3 seconds."),
    Document(page_content="Using strong call-to-actions (CTA) like Comment below increases conversion rates by 25%."),
    Document(page_content="Food content performs best with close-up shots and satisfying sounds (ASMR-style)."),
    Document(page_content="Educational content with a clear hook in the first line retains 60% more viewers."),
    Document(page_content="Travel content benefits from drone shots and golden-hour lighting."),
    Document(page_content="Tech reviews perform best when they show the product in real use within the first 5 seconds."),
]

_chunks = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20).split_documents(_knowledge_base)
_vectorstore = Chroma.from_documents(_chunks, embeddings)
retriever = _vectorstore.as_retriever(search_kwargs={"k": 2})

log.info("RAG vector store ready.")

researcher_prompt = PromptTemplate(
    input_variables=["keyword", "context"],
    template=(
        "You are a social media researcher.\n"
        "Extract viral content insights for the keyword: \"{keyword}\".\n\n"
        "Knowledge context:\n{context}\n\n"
        "Provide 3-5 short, actionable bullet points."
    ),
)

scriptwriter_prompt = PromptTemplate(
    input_variables=["keyword", "research"],
    template=(
        "You are a viral TikTok scriptwriter.\n"
        "Write a 30-second script for the keyword: \"{keyword}\".\n\n"
        "Research insights:\n{research}\n\n"
        "Reply ONLY with this exact JSON:\n"
        "{{\n"
        "    \"scenes\": [\n"
        "        {{\"id\": 1, \"description\": \"visual scene description\", \"duration\": 10}},\n"
        "        {{\"id\": 2, \"description\": \"visual scene description\", \"duration\": 10}},\n"
        "        {{\"id\": 3, \"description\": \"visual scene description\", \"duration\": 10}}\n"
        "    ],\n"
        "    \"voiceover\": \"30-second narration text in English (max 500 characters)\",\n"
        "    \"duration\": 30\n"
        "}}"
    ),
)

guard_prompt = PromptTemplate(
    input_variables=["script"],
    template=(
        "Review this video script:\n{script}\n\n"
        "Criteria:\n"
        "1. Is the voiceover under 500 characters?\n"
        "2. Are there exactly 3 scenes?\n\n"
        "Return ONLY: {{\"approved\": true/false, \"feedback\": \"error description if any\"}}"
    ),
)

researcher_chain   = LLMChain(llm=llm, prompt=researcher_prompt)
scriptwriter_chain = LLMChain(llm=llm, prompt=scriptwriter_prompt)
guard_chain        = LLMChain(llm=llm, prompt=guard_prompt)


def _parse_json(text: str) -> dict:
    import re, json
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No valid JSON found in model output. Raw output: {text[:200]}")
    try:
        return json.loads(match.group())
    except Exception as e:
        raise ValueError(f"JSON parse error: {e}. Raw output: {text[:200]}")


def _invoke_with_retry(chain, inputs: dict, max_attempts: int = 4, base_delay: int = 15):
    for attempt in range(1, max_attempts + 1):
        try:
            response = chain.invoke(inputs)
            log.info(f"LLM response received (attempt {attempt}).")
            return response["text"]
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
            if is_rate_limit and attempt < max_attempts:
                wait = base_delay * attempt
                log.warning(f"Rate limit hit, waiting {wait}s (attempt {attempt}/{max_attempts})...")
                time.sleep(wait)
                continue
            raise


def run_agent_pipeline(keyword: str, max_retries: int = 3) -> dict:
    log.info(f"Starting RAG retrieval: {keyword!r}")
    context = "\n".join(d.page_content for d in retriever.invoke(keyword))
    research = _invoke_with_retry(researcher_chain, {"keyword": keyword, "context": context})

    script = None
    for attempt in range(1, max_retries + 1):
        log.info(f"Script generation attempt {attempt}/{max_retries}...")
        raw = _invoke_with_retry(scriptwriter_chain, {"keyword": keyword, "research": research})
        try:
            script = _parse_json(raw)
        except ValueError as e:
            log.warning(f"Could not parse scriptwriter output: {e}")
            continue

        guard_raw = _invoke_with_retry(guard_chain, {"script": __import__("json").dumps(script)})
        try:
            guard = _parse_json(guard_raw)
        except ValueError:
            log.warning("Could not parse guard output, accepting script.")
            return script

        if guard.get("approved"):
            log.info("Script approved by Brand Guard.")
            return script

        log.info("Guard rejected: " + str(guard.get("feedback", "")))
        research += "\nFEEDBACK: " + str(guard.get("feedback", ""))

    if script is None:
        raise RuntimeError(f"Could not generate a valid script in {max_retries} attempts.")

    log.warning("Guard approval not obtained, using last generated script.")
    return script
