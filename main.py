# SMediaBot - Agente que ayuda a construir diversos tipos de contenido para redes sociales
import os
from dotenv import load_dotenv
from langgraph.graph import START, END, StateGraph
from langchain_ollama import ChatOllama

from typing import TypedDict, Annotated, List
import operator

from langgraph.checkpoint.postgres import PostgresSaver
import psycopg
from psycopg.rows import dict_row # Necesario para row_factory

from langchain_tavily import TavilySearch
# from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import Tool


load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL")
DB_URI = os.getenv("DB_URI")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Define la herramienta de búsqueda
search_tool = TavilySearch(tavily_api_key=TAVILY_API_KEY, max_results=5) # Puedes ajustar max_results
# search_tool = TavilySearchResults(api_key=TAVILY_API_KEY, maax_results=5)
search_tool_for_llm = Tool(
    name="tavily_search", # Nombre que el LLM usará para llamar a la herramienta
    func=search_tool.run, # La función real a ejecutar (el método .run de tu TavilySearch)
    description="Útil para buscar información en la web sobre cualquier tema.",
)


# Definir el prompt que le dega al LLM como comportarse y cuando usar herramientas.
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""Eres un asistente útil y creativo para redes sociales. Tu tarea principal es ayudar al usuario a generar contenido y responder preguntas.
    Tienes acceso a una herramienta de búsqueda web (`tavily_search`).
    **ES ABSOLUTAMENTE CRÍTICO que uses la herramienta `tavily_search` SIEMPRE que una pregunta requiera información actualizada, datos específicos, o algo que no se pueda responder con conocimiento general.**
    Por ejemplo, para preguntas como '¿Cuál es la capital de [país]?', '¿Noticias sobre [tema] hoy?', '¿Beneficios de [algo] según estudios recientes?', debes usar la herramienta.
    Responde al usuario solo después de haber realizado la búsqueda si es necesario.
    """),
    MessagesPlaceholder(variable_name="messages"),
])

llm = ChatOllama(
    base_url=OLLAMA_URL, 
    model="qwen2.5:latest", 
    temperature=0.7,
    tools=[search_tool]
    ).bind_tools([search_tool_for_llm])

# Definir el estado del grafo
class AgentState(TypedDict):
    messages: Annotated[List[AIMessage], operator.add]

# Inicializar el StateGraph
workflow = StateGraph(AgentState)

# Definir el nodo 'chat'
def chat_node(state: AgentState):
    messages = state["messages"]
    response = llm.invoke(messages)

    # --- AÑADIR ESTAS LÍNEAS PARA DEPURACIÓN ---
    print(f"\n--- RESPUESTA DEL LLM EN chat_node ---")
    print(f"Content: {response.content}")
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"Tool Calls detectados: {response.tool_calls}")
    else:
        print("No se detectaron tool calls en la respuesta del LLM.")
    print(f"-----------------------------------\n")
    # ------------------------------------------

    return {"messages": [response]}

tools_map = {"tavily_search": search_tool.invoke}

# Definir el nodo 'tool'
def tool_node(state: AgentState):
    tool_calls = state["messages"][-1].tool_calls
    tool_results = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # Esto es un workaround específico para cómo algunos LLMs generan los argumentos.
        # Idealmente, el LLM debería usar el nombre de argumento definido en la descripción de la herramienta.
        if tool_name == "tavily_search":
            if "__arg1" in tool_args:
                tool_args = {"query": tool_args["__arg1"]} # Mapear __arg1 a query
            # Asegurarse de que 'query' esté presente si la herramienta lo requiere
            elif "query" not in tool_args:
                # Esto es un fallback, si no hay __arg1 ni 'query'
                print(f"Advertencia: No se encontró el argumento 'query' esperado para tavily_search. Argumentos recibidos: {tool_args}")
                tool_results.append(ToolMessage(tool_call_id=tool_call["id"], content="Error: Argumento 'query' faltante para la búsqueda en Tavily."))
                continue # Saltar esta llamada a herramienta y pasar a la siguiente
        # -------------------------------------------------------------

        if tool_name in tools_map:
            # Ejecuta la herramienta
            result = tools_map[tool_name](tool_args)
            tool_results.append(ToolMessage(tool_call_id=tool_call["id"], content=str(result)))
        else:
            tool_results.append(ToolMessage(tool_call_id=tool_call["id"], content=f"Lo siento, no puedo ejecutar la herramienta {tool_name}."))
    return {"messages": tool_results}

# Definir un router para dirigir el flujo
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    # Si el ultimo mensaje es del LLM y tiene llamadas a herramientas, ir al nodo de herramientas.
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "call_tool"
    # S no, significa que el LLM ha terminado de responder
    return END

# Describir el grafo
workflow.add_node("chat", chat_node)
workflow.add_node("call_tool", tool_node)
workflow.add_edge(START, "chat")
# despues del nodo chat, la decision es condicional
workflow.add_conditional_edges(
    "chat",
    should_continue,
    {
        "call_tool": "call_tool",
        END: END
    }
)

# Despues de ejecutar una herramienta, volver al chat para que el LLM procese el resultado
workflow.add_edge("call_tool", "chat")

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