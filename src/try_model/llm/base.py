from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain.messages import HumanMessage
from rich import print as rprint
from rich.text import Text
from prompt_toolkit import prompt
from prompt_toolkit.styles import Style

# forbiden the warning of langchain experimental outputing
import warnings
from langchain_core._api import LangChainBetaWarning
warnings.filterwarnings("ignore", category=LangChainBetaWarning)

llm = init_chat_model(
    model="mistralai/ministral-3-3b",
    model_provider="openai",
    base_url="http://localhost:1234/v1",
    api_key="NONE"
)

def chat_node(state: MessagesState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
graph = builder.compile()

if __name__ == "__main__":
    state = {"messages": []}

    while True:
        input_style = Style.from_dict({
                "": "bold green",
                "prompt": "bold green"
            })
        human_msg = prompt("You: ", style=input_style)
        # skip the null input
        if not human_msg.strip():
            continue

        state["messages"].append(HumanMessage(content=human_msg))

        stream = graph.stream_events(state, version="v3")

        rprint("[yellow]AI: [/yellow]", end="", flush=True)
        for m in stream.messages:
            for t in m.text:
                rprint(Text(t, style="yellow"), end="", flush=True)
        rprint()

        # update the state, that's to say 'update the messages'
        state = stream.output
