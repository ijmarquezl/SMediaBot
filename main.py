# SMediaBot - Agente que ayuda a construir diversos tipos de contenido para redes sociales
import os
from dotenv import load_dotenv
from langgraph.graph import START, END, StateGraph
# from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from typing import TypedDict, Annotated, List, Literal
import operator

from langgraph.checkpoint.postgres import PostgresSaver
import psycopg
from psycopg.rows import dict_row # Necesario para row_factory

from langchain_tavily import TavilySearch

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import Tool

# Importaciones necesarias para Google Imagen
from google.cloud import aiplatform
from google.oauth2 import service_account
import google.auth
from pydantic import Field, BaseModel
import vertexai 
from vertexai.preview.generative_models import GenerativeModel


load_dotenv()
# OLLAMA_URL = os.getenv("OLLAMA_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_URI = os.getenv("DB_URI")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Configuración de Google Vertex AI
VERTEX_AI_LOCATION = "us-central1"

class ImageGenerationInput(BaseModel):
    """Input para la herramienta de generación de imágenes."""
    prompt: str = Field(description="La descripción detallada del tema para la imagen.")

class GoogleImageGenerator:
    """Clase para interactuar con la API de Google Imagen."""
    def __init__(self, project_id: str, location: str):
        self.project_id = project_id
        self.location = location
        
        credentials, _ = google.auth.default()
        vertexai.init(project=self.project_id, location=self.location, credentials=credentials)
        
        self.model = GenerativeModel("imagegeneration@latest")

    def run(self, prompt: str) -> str:
        """Genera una imagen basada en el prompt y devuelve su URL."""
        try:
            response = self.model.generate_content(
                [prompt]
            )
            
            image_urls = []
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if hasattr(part, 'image') and hasattr(part.image, 'uri'):
                        image_urls.append(part.image.uri)
                    elif hasattr(part, 'blob_uri'):
                        image_urls.append(part.blob_uri)
            
            if image_urls:
                print(f"Imagen generada: {image_urls[0]}")
                return f"Imagen generada con éxito. URL: {image_urls[0]}"
            else:
                print(f"Respuesta cruda del modelo: {response}")
                return "No se pudo generar la imagen o la respuesta no contenía una URL."
        except Exception as e:
            if "ResourceExhausted" in str(e):
                return f"Error de cuota excedida con Google Imagen: {e}. Por favor, verifica tu cuota en Google Cloud."
            else:
                return f"Error general al generar imagen con Google Imagen: {e}"

GOOGLE_CLOUD_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
if not GOOGLE_CLOUD_PROJECT_ID:
    print("Advertencia: GOOGLE_CLOUD_PROJECT_ID no está configurado en el .env. La generación de imágenes no funcionará.")

google_image_generator = GoogleImageGenerator(
    project_id=GOOGLE_CLOUD_PROJECT_ID,
    location=VERTEX_AI_LOCATION
)

image_generation_tool_for_llm = Tool(
    name="google_image_generator",
    func=google_image_generator.run,
    description="Útil para generar imágenes de alta calidad a partir de una descripción detallada. Proporciona un prompt descriptivo.",
    args_schema=ImageGenerationInput
)

search_tool = TavilySearch(tavily_api_key=TAVILY_API_KEY, max_results=5)
search_tool_for_llm = Tool(
    name="tavily_search",
    func=search_tool.run,
    description="Útil para buscar información en la web sobre cualquier tema.",
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""Eres un asistente útil y creativo para redes sociales. Tu tarea principal es ayudar al usuario a generar contenido y responder preguntas.
    Tienes acceso a una herramienta de búsqueda web (`tavily_search`).
    **ES ABSOLUTAMENTE CRÍTICO que uses la herramienta `tavily_search` SIEMPRE que una pregunta requiera información actualizada, datos específicos, o algo que no se pueda responder con conocimiento general.**
    Por ejemplo, para preguntas como '¿Cuál es la capital de [país]?', '¿Noticias sobre [tema] hoy?', '¿Beneficios de [algo] según estudios recientes?', debes usar la herramienta.
    Responde al usuario solo después de haber realizado la búsqueda si es necesario.
    """),
    MessagesPlaceholder(variable_name="messages"),
])

# llm = ChatOllama(
#     base_url=OLLAMA_URL, 
#     model="hf.co/unsloth/Qwen3-8B-GGUF:UD-Q4_K_XL", 
#     temperature=0.7,
#     tools=[search_tool, image_generation_tool_for_llm]
#     ).bind_tools([search_tool_for_llm, image_generation_tool_for_llm])

llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.7,
    tools=[search_tool, image_generation_tool_for_llm]
    ).bind_tools([search_tool_for_llm, image_generation_tool_for_llm])

class AgentState(TypedDict):
    messages: Annotated[List[AIMessage], operator.add]

workflow = StateGraph(AgentState)

def chat_node(state: AgentState):
    messages = state["messages"]
    response = llm.invoke(messages)

    print(f"\n--- RESPUESTA DEL LLM EN chat_node ---")
    print(f"Content: {response.content}")
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"Tool Calls detectados: {response.tool_calls}")
    else:
        print("No se detectaron tool calls en la respuesta del LLM.")
    print(f"-----------------------------------\n")

    return {"messages": [response]}

tools_map = {
    "tavily_search": search_tool.invoke,
    "google_image_generator": image_generation_tool_for_llm.invoke,
}

def tool_node(state: AgentState):
    tool_calls = state["messages"][-1].tool_calls
    tool_results = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "tavily_search":
            if "__arg1" in tool_args:
                tool_args = {"query": tool_args["__arg1"]}
            elif "query" not in tool_args:
                print(f"Advertencia: No se encontró el argumento 'query' esperado para tavily_search. Argumentos recibidos: {tool_args}")
                tool_results.append(ToolMessage(tool_call_id=tool_call["id"], content="Error: Argumento 'query' faltante para la búsqueda en Tavily."))
                continue

        if tool_name in tools_map:
            result = tools_map[tool_name](tool_args)
            tool_results.append(ToolMessage(tool_call_id=tool_call["id"], content=str(result)))
        else:
            tool_results.append(ToolMessage(tool_call_id=tool_call["id"], content=f"Lo siento, no puedo ejecutar la herramienta {tool_name}."))
    return {"messages": tool_results}

def generar_blog_post(state: AgentState):
    """Genera un borrador de una entrada de blog basado en el tema."""
    print("\n--- ENTRANDO A generar_blog_post ---")
    tema = ""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            tema = message.content
            break
    else:
        return {"messages": [AIMessage(content="No se detectó un tema claro para la entrada del blog.")]}

    prompt_blog = ChatPromptTemplate.from_messages([
        SystemMessage(content="""Eres un escritor experto en la creación de entradas de blog atractivas e informativas. 
        Basándote en el siguiente tema, genera un borrador completo de una entrada de blog. 
        Incluye un título atractivo, una introducción, varios puntos principales desarrollados y una conclusión.
        El borrador debe estar listo para ser revisado y publicado en un blog de WordPress.
        Intenta mantener un tono alegre, energético y divertido, pero no excesivo y que priorice la información y calidad de la comunicación."""),
        HumanMessage(content=f"El tema para la entrada del blog es: {tema}"),
    ])

    chain = prompt_blog | llm
    response = chain.invoke({"tema": tema})

    print("\n--- SALIENDO DE generar_blog_post ---")
    return {"messages": [response]}

def generar_imagenes_para_blog(state: AgentState):
    """
    Guía al LLM para que genere la llamada a la herramienta de imagen directamente.
    """
    print("\n--- ENTRANDO A generar_imagenes_para_blog ---")
    messages = state["messages"]
    blog_content = ""

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and "Título:" in msg.content:
            blog_content = msg.content
            break

    if not blog_content:
        print("Advertencia: No se encontró contenido de blog para generar imágenes.")
        return {"messages": [AIMessage(content="No se encontró contenido de blog para generar imágenes.")]}

    print(f"Contenido del blog extraído:\n{blog_content[:200]}...")

    image_request_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""Eres un experto en contenido visual. Tu ÚNICA tarea ahora es generar una llamada a la herramienta `google_image_generator` para crear una imagen impactante y relevante para el siguiente contenido del blog.

            Analiza el blog y crea un prompt único, conciso, MUY descriptivo y visual (máximo 30 palabras) que capture la esencia del tema para una imagen de alta calidad. Piensa en elementos visuales concretos y vibrantes.

            DEBES generar una llamada a la herramienta `google_image_generator` con este prompt. No respondas con texto adicional, solo la llamada a la herramienta en el formato adecuado.

            Ejemplo de formato de llamada a la herramienta:
            ```json
            {{
                "tool_calls": [
                    {{
                        "id": "tool_call_id_ejemplo",
                        "type": "function",
                        "function": {{
                            "name": "google_image_generator",
                            "arguments": "{{\\"prompt\\": \\"Una persona meditando en un jardín floreciente, con pensamientos positivos como nubes en el cielo.\\"}}"
                        }}
                    }}
                ]
            }}
            Contenido del blog para inspirar la imagen:
            {blog_content}
            """),

            HumanMessage(content="Genera la llamada a la herramienta ahora."),
            ])

    llm_response_with_tool_calls = llm.invoke(image_request_prompt)

    print(f"\n--- RESPUESTA DEL LLM PARA LA IMAGEN (posibles ToolCalls) ---")
    print(f"Content: {llm_response_with_tool_calls.content}")
    if hasattr(llm_response_with_tool_calls, 'tool_calls') and llm_response_with_tool_calls.tool_calls:
        print(f"Tool Calls detectados: {llm_response_with_tool_calls.tool_calls}")
    else:
        print("No se detectaron tool calls en la respuesta del LLM para la imagen.")
    print(f"----------------------------------------------------------\n")

    print("\n--- SALIENDO DE generar_imagenes_para_blog ---")
    return {"messages": [llm_response_with_tool_calls]}

def final_response_node(state: AgentState):
    print("\n--- ENTRANDO A final_response_node ---")
    messages = state["messages"]
    blog_content = ""
    image_urls = []

    # Buscar el contenido del blog (el AIMessage con "Título:")
    # y las URLs de las imágenes (de los ToolMessages)
    # Recorremos los mensajes desde el final para encontrar los más recientes.
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and "Título:" in msg.content:
            blog_content = msg.content
            # Una vez que encontramos el blog, podemos detener la búsqueda del blog
            # pero seguimos buscando ToolMessages en caso de múltiples imágenes.
        elif isinstance(msg, ToolMessage):
            # Asumimos que los ToolMessages relevantes son los de la generación de imagen
            if "Imagen generada con éxito. URL:" in msg.content:
                url = msg.content.split("URL:")[1].strip()
                image_urls.append(url)
            elif "Error" in msg.content and "imagen" in msg.content:
                image_urls.append(f"Error al generar imagen: {msg.content}")
        
        # Si ya tenemos el blog y al menos una imagen (o un error de imagen),
        # podemos asumir que hemos procesado los mensajes relevantes.
        if blog_content and (image_urls or "No se pudo generar" in blog_content):
            break


    if not blog_content:
        return {"messages": [AIMessage(content="Error: No se pudo recuperar el contenido del blog para la respuesta final.")]}

    final_response_content = blog_content + "\n\n--- Imágenes Sugeridas para el Blog ---\n"
    if image_urls:
        for i, url in enumerate(image_urls):
            final_response_content += f"Imagen {i+1}: {url}\n"
    else:
        final_response_content += "No se pudieron generar imágenes para el blog."

    print(f"\n--- CONTENIDO FINAL RETORNADO POR final_response_node ---")
    print(final_response_content)
    print(f"----------------------------------------------------------\n")

    return {"messages": [AIMessage(content=final_response_content)]}


def route_from_chat(state: AgentState) -> Literal["call_tool", "generar_blog", END, None]:
    last_message = state["messages"][-1]

    # 1. ¿El LLM hizo una llamada a herramientas? (Prioridad ALTA)
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "call_tool"

    # 2. ¿El usuario pidió generar un blog? (Solo si no hay tool calls pendientes)
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage) and "generar blog" in message.content.lower():
            return "generar_blog"

    # 3. Si no hay llamadas a herramientas ni una petición clara de blog, finalizar.
    return END

# Nueva función para enrutar después de una llamada a herramienta
def route_from_tool(state: AgentState) -> str:
    last_human_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human_message = msg.content
            break

    last_tool_call = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            last_tool_call = msg.tool_calls[-1]["name"]
            break

    if last_human_message and "generar blog" in last_human_message.lower():
        if last_tool_call == "tavily_search":
            return "chat"  # Volver al chat para generar el blog con los resultados
        elif last_tool_call == "google_image_generator":
            return "final_response_node" # Si la última tool call fue para la imagen
        else:
            return "chat" # En otros casos, volver al chat
    else:
        return "chat" # Si no fue una petición de blog inicial


# Describir el grafo
workflow.add_node("chat", chat_node)
workflow.add_node("call_tool", tool_node)
workflow.add_node("generar_blog", generar_blog_post)
workflow.add_node("generar_imagenes_para_blog", generar_imagenes_para_blog)
workflow.add_node("final_response_node", final_response_node) # Añadir el nuevo nodo

workflow.add_edge(START, "chat")

workflow.add_conditional_edges(
    "chat",
    route_from_chat,
    {
        "call_tool": "call_tool",
        "generar_blog": "generar_blog",
        END: END
    }
)

# Flujo específico para la generación de blog e imagen
workflow.add_edge("generar_blog", "generar_imagenes_para_blog")
# Ahora, generar_imagenes_para_blog devuelve un tool_call, que debe ser ejecutado por call_tool
workflow.add_edge("generar_imagenes_para_blog", "call_tool") 
workflow.add_edge("call_tool", "chat")

# Después de ejecutar una herramienta, la decisión es condicional
workflow.add_conditional_edges(
    "call_tool",
    route_from_tool, # Usamos la nueva función para enrutar desde tool_node
    {
        "final_response_node": "final_response_node", # Si es la imagen del blog, ir al nodo final
        "chat": "chat" # Si es una búsqueda normal, volver al chat
    }
)

# El nodo final de respuesta marca el fin de este flujo
workflow.add_edge("final_response_node", END)

try:
    conn = psycopg.connect(DB_URI, autocommit=True, row_factory=dict_row)
    checkpointer = PostgresSaver(conn=conn)
    checkpointer.setup()
    print("Tablas de checkpoint creadas/verificadas en la base de datos.")
except Exception as e:
    print(f"Error al intentar configurar el checkpointer: {e}")
    print("Asegúrate de que la DB_URI es correcta y que la base de datos está accesible.")
    exit()

app = workflow.compile(checkpointer=checkpointer)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            print("\nSaliendo...")
            break
        config = {"configurable": {"thread_id": "smb_convo_1"}}

        inputs = {"messages": [HumanMessage(content=user_input)]}
        response = app.invoke(inputs, config=config)
        print(response["messages"][-1].content)
    except KeyboardInterrupt:
        print("\nSaliendo...")
        break