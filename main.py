# SMediaBot - Agente que ayuda a construir diversos tipos de contenido para redes sociales
import os
from dotenv import load_dotenv
from langgraph.graph import START, END, StateGraph
from langchain_ollama import ChatOllama

from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.checkpoint.postgres import PostgresSaver
import psycopg
from psycopg.rows import dict_row # Necesario para row_factory

from langchain_tavily import TavilySearch


load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL")
DB_URI = os.getenv("DB_URI")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Define la herramienta de búsqueda
search_tool = TavilySearch(tavily_api_key=TAVILY_API_KEY, max_results=5) # Puedes ajustar max_results

llm = ChatOllama(
    base_url=OLLAMA_URL, 
    model="deepseek-r1:8b", 
    temperature=0.7,
    tools=[search_tool]
    )
# Definir el estado del grafo
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# Inicializar el StateGraph
workflow = StateGraph(AgentState)

# Definir el nodo 'chat'
def chat_node(state: AgentState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# Describir el grafo
workflow.add_node("chat", chat_node)
workflow.add_edge(START, "chat")
workflow.add_edge("chat", END)

# Configurar el checkpointer con PostgreSQL
# _checkpointer_instance = PostgresSaver.from_conn_string(DB_URI)
# checkpointer = _checkpointer_instance # Asignamos la instancia a 'checkpointer'
try:
    # Abre la conexión. Autocommit es crucial para .setup() y row_factory para el funcionamiento interno.
    conn = psycopg.connect(DB_URI, autocommit=True, row_factory=dict_row)
    checkpointer = PostgresSaver(conn=conn) # Pasa el objeto de conexión

    # IMPORTANTE: Llama a .setup() la primera vez que uses el checkpointer
    checkpointer.setup()
    print("Tablas de checkpoint creadas/verificadas en la base de datos.")
except Exception as e:
    print(f"Error al intentar configurar el checkpointer: {e}")
    # Si las tablas ya existen, o el error es de conexión, manejarlo
    print("Asegúrate de que la DB_URI es correcta y que la base de datos está accesible.")
    print("Si el error persiste y las tablas ya existen, puedes comentar checkpointer.setup().")
    # Aquí puedes decidir si quieres que el programa termine o intente continuar
    # Por ahora, para depurar, es mejor que se muestre el error claro.
    exit() # Salir para depurar si hay un problema grave con el checkpointer

app = workflow.compile(checkpointer=checkpointer)

# Prueba del grafo
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            print("\nSaliendo...")
            break
        # Define un thread_id fijo para probar la persistencia en una misma conversación
        # ESTO ES NECESARIO para que el checkpointer sepa dónde guardar y cargar
        config = {"configurable": {"thread_id": "smb_convo_1"}}

        inputs = {"messages": [HumanMessage(content=user_input)]}
        response = app.invoke(inputs, config=config)
        print(response["messages"][-1].content)
    except KeyboardInterrupt:
        print("\nSaliendo...")
        break