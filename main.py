"""
main.py
=======
Demo: a two-agent pipeline using the raw-agent framework.

Pipeline
--------
1. Research Agent  – gathers key facts about a topic using simulated tools.
2. Writer Agent    – takes the research output and drafts a short report.

Run
---
    GEMINI_API_KEY=your_key python main.py
"""

import sys
import os
import arxiv
from typing import List
from pydantic import BaseModel
# Make sure the agents package is importable regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agents"))

from raw_agent import Tool, Agent, Task, RawAgent  # noqa: E402  (path hack above)


class Paper(BaseModel):
    title: str
    authors: List[str]
    summary: str
    link: str


# ---------------------------------------------------------------------------
# 1.  Define some simple tools
# ---------------------------------------------------------------------------

def web_search(query: str) -> str:
    """Simulates a web search — replace with a real API call if you like."""
    fake_db = {
        "Python":
            "Python is a high-level, interpreted, general-purpose programming "
            "language created by Guido van Rossum (1991). It emphasises "
            "readability and supports multiple paradigms including OOP and "
            "functional programming. As of 2024 it is the most popular language "
            "according to the TIOBE index.",
        "AI agents":
            "An AI agent is an autonomous entity that perceives its environment "
            "and takes actions to achieve goals. Modern LLM-based agents (e.g. "
            "ReAct, AutoGPT, CrewAI) combine large language models with tool use, "
            "memory, and planning to tackle multi-step tasks.",
    }
    for key, value in fake_db.items():
        if key.lower() in query.lower():
            return value
    return f"No results found for '{query}'."

def search_papers(topic: str, max_results: int = 3) -> List[Paper]:
    """
    Search arXiv for papers using the updated Client API
    Returns list of Paper Pydantic models
    """
    client_arxiv = arxiv.Client(page_size=max_results)
    search = arxiv.Search(query=topic, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    results = []
    for paper in client_arxiv.results(search):
        results.append(Paper(
            title=paper.title,
            authors=[author.name for author in paper.authors],
            summary=paper.summary,
            link=paper.entry_id
        ))
    return results

def word_count(text: str) -> str:
    """Returns the word count of the provided text."""
    count = len(text.split())
    return f"The text contains {count} words."


search_tool = Tool(
    name="web_search",
    description=(
        "Search the web for information on a topic. "
        "Pass a single argument: query (str)."
    ),
    func=web_search,
)
search_papers_tool = Tool(
    name="search_papers",
    description=(
        "Search arXiv for papers on a topic. "
        "Pass a single argument: topic (str)."
    ),
    func=search_papers,
)

counter_tool = Tool(
    name="word_count",
    description=(
        "Count the words in a piece of text. "
        "Pass a single argument: text (str)."
    ),
    func=word_count,
)


# ---------------------------------------------------------------------------
# 2.  Define Agents
# ---------------------------------------------------------------------------
# 💡 Provider is auto-detected from the model name — no extra config needed.
#
#   Gemini models  → start with "gemini-"
#     llm="gemini-2.5-flash"         (requires GEMINI_API_KEY in .env)
#     llm="gemini-2.0-flash"
#
#   Groq models    → start with "llama", "mixtral", "deepseek", etc.
#     llm="llama-3.3-70b-versatile"  (requires GROQ_API_KEY in .env)
#     llm="llama-3.1-8b-instant"
#     llm="mixtral-8x7b-32768"
#     llm="deepseek-r1-distil-llama-70b"
#
#   You can also override explicitly:
#     Agent(..., llm="my-model", provider="groq")
# ---------------------------------------------------------------------------

research_agent = Agent(
    role="Research Analyst",
    goal="Find accurate, concise information on the given topic.",
    backstory=(
        "You are an experienced research analyst who excels at distilling "
        "complex information into clear, factual summaries."
    ),
    # ── Switch provider by changing the model name ──────────────────────────
    llm="llama-3.3-70b-versatile",      # Groq  (auto-detected)
    # llm="gemini-2.5-flash",           # Gemini (auto-detected)
    # ────────────────────────────────────────────────────────────────────────
    provider='groq',
    tools=[search_papers_tool],
    max_iterations=4,
    verbose=True,
)

writer_agent = Agent(
    role="Technical Writer",
    goal="Transform research notes into a well-structured short report.",
    backstory=(
        "You are a skilled technical writer who can turn bullet-point research "
        "into engaging, professional prose that is easy to understand."
    ),
    # ── Switch provider by changing the model name ──────────────────────────
    llm="llama-3.1-8b-instant",         # Groq  (auto-detected, fast + cheap)
    # llm="gemini-2.5-flash",           # Gemini (auto-detected)
    # ────────────────────────────────────────────────────────────────────────
    tools=[counter_tool],
    max_iterations=3,
    verbose=True,
)


# ---------------------------------------------------------------------------
# 3.  Define Tasks
# ---------------------------------------------------------------------------

research_task = Task(
    description=(
        "Research the topic 'Python' and summarise: "
        "(1) what they are, (2) what are the use cases, "
        "(3) what are the key features."
    ),
    expected_output=(
        "A 3-5 sentence factual summary covering the three points above."
    ),
    agent=research_agent,
)

writing_task = Task(
    description=(
        "Using the research notes provided, write a short report (100-150 words) "
        "about Python.  The report should have a title, two short paragraphs, "
        "and a one-line conclusion."
    ),
    expected_output=(
        "A polished short report with a title, two paragraphs, and a conclusion."
    ),
    agent=writer_agent,
    context_tasks=[research_task],   # ← writer sees the researcher's output
)


# ---------------------------------------------------------------------------
# 4.  Run the crew
# ---------------------------------------------------------------------------

def main() -> None:
    crew = RawAgent(
        tasks=[research_task, writing_task],
        agents=[research_agent, writer_agent],
        process="sequential",
        verbose=True,
    )

    outputs = crew.run()

    print("\n" + "=" * 60)
    print("FINAL OUTPUTS")
    print("=" * 60)
    for i, (task, output) in enumerate(zip(crew.tasks, outputs), 1):
        print(f"\n--- Task {i}: {task.description[:60]}... ---")
        print(output)


if __name__ == "__main__":
    main()
