# RawCrew 🚀

A minimal, lightweight multi-agent orchestration framework inspired by CrewAI. Built for simplicity, speed, and modularity.

## ✨ Features

- **Multi-Provider Support**: Switch between **Google Gemini** and **Groq** (LLaMA 3, Mixtral, etc.) by just changing the model name.
- **ReAct Agent Loop**: Structured reasoning with thought-action-observation cycles.
- **Sequential & Parallel Processing**: Run tasks one-by-one with context chaining, or all at once via multi-threading.
- **Context Chaining**: Automatically feed the output of one task as context into the next.
- **Tool Support**: Easily define and attach Python functions as tools for agents.
- **Zero-Config Detection**: Automatically detects the LLM provider (Gemini vs Groq) from the model string.

## 🛠️ Installation

```bash
git clone https://github.com/your-username/rawcrew.git
cd rawcrew
pip install -r requirements.txt
```

## ⚙️ Configuration

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
```

## 🚀 Quick Start (Sequential)

```python
from agents.raw_agent import Agent, Task, RawAgent, Tool

# 1. Define Tools
def web_search(query: str):
    return f"Results for {query}..."

search_tool = Tool(name="search", description="Search the web", func=web_search)

# 2. Define Agents (Provider is auto-detected)
researcher = Agent(
    role="Researcher",
    goal="Gather info",
    llm="llama-3.3-70b-versatile"  # Groq
)

writer = Agent(
    role="Writer",
    goal="Write report",
    llm="gemini-2.5-flash"         # Gemini
)

# 3. Define Tasks
task1 = Task(description="Research AI", expected_output="Summary", agent=researcher)
task2 = Task(description="Write blog", expected_output="Post", agent=writer, context_tasks=[task1])

# 4. Run!
crew = RawAgent(tasks=[task1, task2], process="sequential")
crew.run()
```

## ⚡ Parallel Execution

Run independent tasks simultaneously:

```python
crew = RawAgent(tasks=[task_a, task_b], process="parallel")
crew.run()
```

## 📂 Project Structure

- `agents/`: Core framework logic.
- `main.py`: Entry point and demo examples.
- `requirements.txt`: Dependencies (google-genai, groq, arxiv, pydantic, etc.).
- `.env`: API keys and environment variables.

---
Built with ❤️ for rapid AI agent development.
