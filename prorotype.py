import os
import re
import uuid
from typing import Annotated, Literal, List, Dict

import pydantic
from dotenv import load_dotenv
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.store.postgres import PostgresStore
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

os.environ["GOOGLE_CSE_ID"] = "YOUR_CSE_ID"
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"

DB_URI = "postgresql://postgres:YOUR_DB_URI"

# MODELS AND SCHEMAS
class MessageClassifier(pydantic.BaseModel):
    message_types: Literal["emotional", "logical"] = pydantic.Field(
        ...,
        description="Classify if the messages requires an emotional or logical response."
    )


class TherapistSearchDetector(pydantic.BaseModel):
    needs_therapist_search: bool = pydantic.Field(
        ...,
        description="Determine if the user is asking for therapist recommendations, psychological help, or mental health professionals"
    )
    location_mentioned: bool = pydantic.Field(
        ...,
        description="Whether the user mentioned a specific location in their message. If not ask the user about the location"
    )
    location: str = pydantic.Field(
        default="",
        description="Location mentioned by the user (empty if not mentioned)"
    )

class LocationExtractor(pydantic.BaseModel):
    location: str = pydantic.Field(
        ...,
        description="The location where the user wants to find therapists"
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_types: str | None
    needs_therapist_search: bool | None
    location_mentioned: bool | None
    location: str | None
    waiting_for_location: bool | None
    next: str | None


class TherapistInfo(pydantic.BaseModel):
    name: str = pydantic.Field(default="Not specified", description="Name of the therapist/psychologist")
    address: str = pydantic.Field(default="Not specified", description="Address or location")
    phone: str = pydantic.Field(default="Not specified", description="Phone number")
    email: str = pydantic.Field(default="Not specified", description="Email address")
    website: str = pydantic.Field(default="Not specified", description="Website URL")
    specialization: str = pydantic.Field(default="Not specified", description="Area of specialization")

class TherapistList(pydantic.BaseModel):
    therapists: List[TherapistInfo] = pydantic.Field(description="List of therapist information")


# LLM INITIALIZATION
def initialize_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key="API_KEY")
    print("LLM model initialized")
    return llm


# DOCUMENT PROCESSING
def setup_document_retrieval():
    pdf_loader = PyPDFLoader("YOUR_LINK")
    pages = pdf_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=50,
        length_function=len,
    )
    split_pages = text_splitter.split_documents(pages)
    print("Documents loaded and split")

    split_pages = split_pages[0:1]

    # embeddings
    embeddings = OllamaEmbeddings(
        model="llama3.2",
        base_url="YOUR_URL",
    )

    # vector store
    vectorstore = Chroma.from_documents(split_pages, embeddings)
    retriever = vectorstore.as_retriever()
    print("Vector store and retriever created")
    return retriever


def create_tools(retriever):
    tools = []
    if retriever:
        retriever_tool = create_retriever_tool(
            retriever,
            "retrieve_mental_support",
            "Search and return information about psychological support, mental health, and emotional wellness.",
        )
        tools.append(retriever_tool)
    try:
        search = GoogleSearchAPIWrapper()

        def search_therapists(query: str) -> str:
            search_queries = [
                f"psicologo {query} contatti telefono",
                f"psicoterapeuta {query} studio",
                f"centro psicologico {query}",
                f"therapist psychologist {query} contact"
            ]

            all_results = []
            for search_query in search_queries:
                try:
                    print(f"Searching for: {search_query}")
                    results = search.run(search_query)
                    if results:
                        all_results.append(f"Results for '{search_query}':\n{results}\n")
                        print(f"Got {len(results)} characters of results")
                except Exception as e:
                    print(f"Search failed for '{search_query}': {e}")
                    continue

            combined_results = "\n".join(all_results)
            print(f"Total combined results length: {len(combined_results)}")

            return combined_results if combined_results else "No search results found"

        therapist_search_tool = Tool(
            name="search_therapists",
            description="Search for psychological therapists, psychologists, and mental health professionals with their contact information.",
            func=search_therapists,
        )
        tools.append(therapist_search_tool)
        print("Google search tool created successfully")
    except Exception as e:
        print(f"Could not initialize Google search tool: {e}")

    print(f"Created {len(tools)} tools")
    return tools


# MEMORY
def save_conversation_memory(store: PostgresStore, user_id: str, user_message: str, assistant_response: str,
                             message_type: str):
    namespace = ("conversations", user_id)
    memory_id = str(uuid.uuid4())
    print(f"Saving conversation memory for user_id: {user_id}")

    memory_data = {
        "timestamp": str(uuid.uuid1()),
        "user_message": user_message,
        "assistant_response": assistant_response,
        "message_type": message_type,
        "interaction_type": "conversation"
    }
    try:
        store.put(namespace, memory_id, memory_data)
        print(f"Successfully saved conversation memory with ID: {memory_id}")
    except Exception as e:
        print(f"Error saving conversation memory: {e}")


def save_user_context(store: PostgresStore, user_id: str, context_type: str, context_data: str):
    namespace = ("user_context", user_id)
    context_id = str(uuid.uuid4())

    context_memory = {
        "timestamp": str(uuid.uuid1()),
        "context_type": context_type,
        "data": context_data,
        "interaction_type": "context"
    }
    try:
        store.put(namespace, context_id, context_memory)
        print(f"Successfully saved user context for user: {user_id[:8]}")
    except Exception as e:
        print(f"Error saving user context: {e}")


def retrieve_relevant_memories(store: PostgresStore, user_id: str, current_message: str, limit: int = 5):
    try:
        # conversation history
        conv_namespace = ("conversations", user_id)
        conv_memories = store.search(conv_namespace, query=current_message, limit=limit)

        # user context
        context_namespace = ("user_context", user_id)
        context_memories = store.search(context_namespace, query=current_message, limit=10)

        print(f"Found {len(conv_memories)} conversation memories and {len(context_memories)} context memories")
        return conv_memories, context_memories
    except Exception as e:
        print(f"Error retrieving relevant memories: {e}")
        return None, None


def get_all_user_memories(store: PostgresStore, user_id: str):
    try:
        conv_namespace = ("conversations", user_id)
        conv_memories = store.search(conv_namespace, query="", limit=100)

        context_namespace = ("user_context", user_id)
        context_memories = store.search(context_namespace, query="", limit=50)

        print(
            f"Retrieved {len(conv_memories)} conversation memories and {len(context_memories)} context memories for summary")
        return conv_memories, context_memories
    except Exception as e:
        print(f"Error retrieving all user memories: {e}")
        return None, None


def generate_user_summary(store: PostgresStore, user_id: str, llm):
    try:
        conv_memories, context_memories = get_all_user_memories(store, user_id)

        if not conv_memories and not context_memories:
            return "No previous conversation history found."

        personal_info = []
        if context_memories:
            for context in context_memories:
                try:
                    if hasattr(context, 'value') and context.value:
                        if isinstance(context.value, dict):
                            context_data = context.value.get('data', '')
                            if context_data and context_data != "None":
                                personal_info.append(context_data)
                except Exception as e:
                    print(f"Error processing context memory: {e}")

        recent_conversations = []
        if conv_memories:
            for conv in conv_memories[-10:]:
                try:
                    if hasattr(conv, 'value') and conv.value:
                        if isinstance(conv.value, dict):
                            user_msg = conv.value.get('user_message', '')
                            msg_type = conv.value.get('message_type', '')
                            if user_msg:
                                recent_conversations.append(f"[{msg_type}] {user_msg[:100]}...")
                except Exception as e:
                    print(f"Error processing conversation memory: {e}")

        summary_prompt = f"""
        Based on the conversation history and personal information below, create a brief, friendly summary of what you know about this user.

        Personal Information:
        {chr(10).join(personal_info) if personal_info else "No specific personal information stored."}

        Recent Conversation Topics:
        {chr(10).join(recent_conversations) if recent_conversations else "No recent conversations found."}

        Create personalized summary that shows you remember the user. Keep it under 150 words.
        Focus on:
        1. Personal details you know about them
        2. Main topics they've discussed
        3. Their communication style (emotional vs logical)
        4. Any ongoing concerns or interests

        Format it as a friendly greeting that shows continuity from previous conversations.
        """

        response = llm.invoke([{"role": "user", "content": summary_prompt}])
        return response.content.strip()
    except Exception as e:
        print(f"Error generating user summary: {e}")
        return "Welcome back! I'm having trouble accessing your conversation history, but I'm here to help."


def extract_important_info(message: str, llm):
    extraction_prompt = f"""
     Extract specific personal information from this message that should be remembered about the user.
    Focus on concrete facts and details:
     EXTRACT IF MENTIONED:
    - Personal details (name, age, location)
    - Mental health concerns or conditions mentioned
    - Important life events or situations
    - Preferences or interests
    - Specific requests to remember something

    Message: "{message}"

    If there's important information to remember, return it as a brief summary. Format your response as clear, factual statements. For example:
    "Name is Sarah, 28 years old, lives in Rome, has anxiety issues, enjoys reading.."

    If there's nothing particularly important to remember, return "None".
    """
    try:
        response = llm.invoke([{"role": "user", "content": extraction_prompt}])
        extracted = response.content.strip()
        return extracted if extracted.lower() != "none" else "None"
    except Exception as e:
        print(f"Error extracting important info: {e}")
        return "None"


def build_memory_context(conv_memories, context_memories, show_extracted_only=True):
    memory_context = ""
    if context_memories:
        personal_info = []
        for context in context_memories:
            try:
                if hasattr(context, 'value') and context.value:
                    if isinstance(context.value, dict):
                        context_data = context.value.get('data', '')
                        if context_data and context_data != "None":
                            personal_info.append(context_data)
            except Exception as e:
                print(f"Error processing context memory: {e}")

        if personal_info:
            memory_context += "What I know about you:\n"
            for info in personal_info[-10:]:
                memory_context += f"- {info}\n"
            memory_context += "\n"

    if conv_memories and not show_extracted_only:
        memory_context += "Based on our previous conversations, I understand you've discussed topics related to mental health and personal support.\n\n"

    return memory_context


# AGENT FUNCTIONS
def classify_message(state: State, llm):
    last_message = state["messages"][-1]

    print(f"User message: '{last_message.content}'")
    print(f"Waiting for location: {state.get('waiting_for_location', False)}")

    if state.get("waiting_for_location", False):
        print("Processing location response...")
        location_extractor = llm.with_structured_output(LocationExtractor, method='json_schema')
        location_result = location_extractor.invoke([
            {
                "role": "system",
                "content": """Extract the location/city from the user's message. 
                    Look for city names, regions, or any geographical indicators.
                    If the user is providing a location, extract it. If they're asking something else, 
                    try to extract any location mentioned anyway."""
            },
            {"role": "user", "content": last_message.content}
        ])

        if location_result.location and location_result.location.strip():
            print(f"Location extracted: '{location_result.location}'")
            return {
                "message_types": "therapist_search",
                "needs_therapist_search": True,
                "location_mentioned": True,
                "location": location_result.location.strip(),
                "waiting_for_location": False
            }
        else:
            print("No location found in response, need to ask again")
            return {
                "message_types": "therapist_search",
                "needs_therapist_search": True,
                "location_mentioned": False,
                "location": None,
                "waiting_for_location": False
            }

    print("Processing regular classification...")

    classifier_llm = llm.with_structured_output(MessageClassifier, method='json_schema')
    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as either:
            - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
            - 'logical': if it asks for facts, information, logical analysis, or practical solutions
            """
        },
        {"role": "user", "content": last_message.content}
    ])

    print(f"Message type: {result.message_types}")

    therapist_detector = llm.with_structured_output(TherapistSearchDetector, method='json_schema')
    therapist_result = therapist_detector.invoke([
        {
            "role": "system",
            "content": """Determine if the user is asking for:
                - Therapist recommendations
                - Psychologist contact information
                - Mental health professionals
                - Therapy services or counseling
                - Professional psychological help

                Look for location mentions (city, region, country).
                For location: Only set location_mentioned=True if a specific city/location is clearly mentioned.
                                
                IMPORTANT: Only extract locations that are actually mentioned in the user's message.
                If no location is mentioned, return empty string for location."""
        },
        {"role": "user", "content": last_message.content}
    ])

    print(f"Needs therapist search: {therapist_result.needs_therapist_search}")
    print(f"Location mentioned: {therapist_result.location_mentioned}")
    print(f"Extracted location: '{therapist_result.location}'")

    return {
        "message_types": result.message_types,
        "needs_therapist_search": therapist_result.needs_therapist_search,
        "location_mentioned": therapist_result.location_mentioned,
        "location": therapist_result.location if therapist_result.location_mentioned else None,
        "waiting_for_location": False
    }

def router(state: State):
    needs_therapist_search = state.get("needs_therapist_search", False)
    location_mentioned = state.get("location_mentioned", False)
    location = state.get("location")
    message_types = state.get("message_types")

    print(f"needs_therapist_search: {needs_therapist_search}")
    print(f"location_mentioned: {location_mentioned}")
    print(f"location: '{location}'")
    print(f"message_types: {message_types}")
    print(f"Full state: {state}")

    if needs_therapist_search:
        if not location_mentioned or not location or location.strip() == "":
            print("Router decision: ask_location")
            return {"next": "ask_location"}
        else:
            print("Router decision: therapist_search")
            return {"next": "therapist_search"}
    elif state.get("message_types") == "emotional":
        print("Router decision: friend")
        return {"next": "friend"}
    print("Router decision: logical")
    return {"next": "logical"}


def ask_location_agent(state: State, config: RunnableConfig, *, store: PostgresStore, llm):
    last_message = state["messages"][-1]
    user_id = config["configurable"]["user_id"]

    location_prompt = """The user is asking for therapist recommendations so ask them politely where they are located and only then help them to find local mental health professionals.
        """

    response = llm.invoke([{"role": "user", "content": location_prompt}])

    save_conversation_memory(store, user_id, last_message.content, response.content, "location_request")

    return {
        "messages": [AIMessage(content=response.content)],
        "waiting_for_location": True,
        "needs_therapist_search": True,
        "message_types": "therapist_search",
        "location_mentioned": False,
        "location": None
    }

def friend_agent(state: State, config: RunnableConfig, *, store: PostgresStore, llm):
    all_messages = state["messages"]
    last_message = all_messages[-1]
    user_id = config["configurable"]["user_id"]

    print(f"Friend agent using user_id: {user_id}")

    conv_memories, context_memories = retrieve_relevant_memories(store, user_id, last_message.content)
    memory_context = build_memory_context(conv_memories, context_memories, show_extracted_only=True)

    conversation_history = ""
    if len(all_messages) > 1:
        recent_messages = all_messages[-6:]
        conversation_history = "Current conversation context:\n"
        for msg in recent_messages[:-1]:
            role = "You" if isinstance(msg, HumanMessage) else "Me"
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            conversation_history += f"{role}: {content}\n"
        conversation_history += "\n"

    system_prompt = f"""You are a compassionate and empathetic friend. Your role is to provide emotional support and understanding.

            {conversation_history}
            {memory_context}

           Guidelines:
            - Show genuine empathy and validate their feelings
            - Ask thoughtful questions to help them explore emotions
            - Use any personal information you know about them naturally in conversation
            - Avoid giving direct advice unless asked
            - Focus on emotional support rather than logical solutions
            - Be warm, caring, and supportive in your tone
            - IMPORTANT: Respond ONLY to the current message. Do NOT repeat or reference previous messages in your response."""

    llm_messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": last_message.content}
                    ]

    reply = llm.invoke(llm_messages)

    save_conversation_memory(store, user_id, last_message.content, reply.content, "emotional")

    important_info = extract_important_info(last_message.content, llm)
    if important_info != "None":
        save_user_context(store, user_id, "personal_info", important_info)

    return {"messages": [AIMessage(content=reply.content)]}


def logical_agent(state: State, config: RunnableConfig, *, store: PostgresStore, llm):
    all_messages = state["messages"]
    last_message = all_messages[-1]
    user_id = config["configurable"]["user_id"]

    print(f"Logical agent using user_id: {user_id}")

    conv_memories, context_memories = retrieve_relevant_memories(store, user_id, last_message.content)
    memory_context = build_memory_context(conv_memories, context_memories, show_extracted_only=True)

    conversation_history = ""
    if len(all_messages) > 1:
        recent_messages = all_messages[-6:]
        conversation_history = "Current conversation:\n"
        for msg in recent_messages[:-1]:
            role = "You" if isinstance(msg, HumanMessage) else "Me"
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            conversation_history += f"{role}: {content}...\n"
        conversation_history += "\n"

    system_prompt = f"""You are a logical and a knowledgeable assistant. Provide clear, factual, and well-reasoned responses.

            {conversation_history}
            {memory_context}

            Guidelines:
            - Focus on facts, logic, and practical information
            - Be direct and clear in your explanations
            - Use any personal information you know about them naturally
            - Provide evidence-based responses when possible
            - Be helpful and informative
            - IMPORTANT: Respond ONLY to the current message. Do NOT repeat or reference previous messages in your response."""

    llm_messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": last_message.content}
                    ]

    reply = llm.invoke(llm_messages)

    save_conversation_memory(store, user_id, last_message.content, reply.content, "logical")

    important_info = extract_important_info(last_message.content, llm)
    if important_info != "None":
        save_user_context(store, user_id, "personal_info", important_info)

    return {"messages": [AIMessage(content=reply.content)]}


def extract_therapist_data(search_results: str, llm) -> List[Dict[str, str]]:
    extraction_prompt = f"""
    Extract information about mental health professionals, psychologists, or therapists from these search results.

    Look for:
    - Any names of professionals (Dr., Dott., Psicologo, Psicoterapeuta, etc.)
    - Practice names or clinic names
    - Any contact information available
    - Locations or addresses
    - Specializations mentioned

    IMPORTANT: 
    - Include a professional even if you only have their name
    - If contact info is missing, use "Not specified" 
    - Focus on finding at least the NAME of professionals
    - Look for any mental health related professionals, not just perfect matches

    Search Results:
    {search_results}

    Extract as many mental health professionals as you can find, even with incomplete information.
    """

    try:
        structured_llm = llm.with_structured_output(TherapistList, method='json_schema')
        result = structured_llm.invoke([{"role": "user", "content": extraction_prompt}])

        therapist_data = []
        for therapist in result.therapists:
            # Only require that we have a name - everything else can be "Not specified"
            if therapist.name and therapist.name.lower() != "not specified":
                therapist_data.append({
                    "Name": therapist.name,
                    "Address": therapist.address,
                    "Phone": therapist.phone,
                    "Email": therapist.email,
                    "Website": therapist.website,
                    "Specialization": therapist.specialization
                })

        print(f"DEBUG: Extracted {len(therapist_data)} therapists")

        if not therapist_data:
            print("No therapists found, trying basic extraction...")
            therapist_data = extract_names_only(search_results)

        return therapist_data

    except Exception as e:
        print(f"Error in structured extraction: {e}")
        return extract_names_only(search_results)


def extract_names_only(search_results: str) -> List[Dict[str, str]]:
    therapist_data = []

    patterns = [
        r'(?:Dott\.?|Dr\.?|Psicologo|Psicoterapeuta)\s+([A-Za-z\s]{3,30})',
        r'Studio\s+([A-Za-z\s]{3,30})',
        r'Centro\s+([A-Za-z\s]{3,30})',
    ]

    found_names = set()
    for pattern in patterns:
        matches = re.findall(pattern, search_results, re.IGNORECASE)
        for match in matches:
            name = match.strip()
            if len(name) > 2 and name not in found_names:
                found_names.add(name)

    for name in list(found_names)[:5]:  # Limit to 5
        therapist_data.append({
            "Name": name,
            "Address": "Not specified",
            "Phone": "Not specified",
            "Email": "Not specified",
            "Website": "Not specified",
            "Specialization": "Not specified"
        })

    print(f"Basic extraction found {len(therapist_data)} names")
    return therapist_data

def extract_with_patterns(search_results: str) -> List[Dict[str, str]]:
    therapist_data = []

    name_patterns = [
        r'Dott\.?\s*([A-Za-z\s]+)',
        r'Dr\.?\s*([A-Za-z\s]+)',
        r'Psicologo\s+([A-Za-z\s]+)',
        r'Psicoterapeuta\s+([A-Za-z\s]+)',
    ]

    phone_patterns = [
        r'(\+39\s*)?(\d{2,3}[-.\s]?\d{6,8})',
        r'(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',
    ]

    email_patterns = [
        r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
    ]

    website_patterns = [
        r'(https?://[^\s]+)',
        r'(www\.[^\s]+)',
    ]

    found_names = set()
    for pattern in name_patterns:
        matches = re.findall(pattern, search_results, re.IGNORECASE)
        for match in matches:
            name = match.strip()
            if len(name) > 2 and name not in found_names:
                found_names.add(name)

    phones = re.findall('|'.join(phone_patterns), search_results)
    emails = re.findall('|'.join(email_patterns), search_results)
    websites = re.findall('|'.join(website_patterns), search_results)

    for i, name in enumerate(list(found_names)[:5]):  # Limit to 5 results
        therapist_data.append({
            "Name": name,
            "Address": "Not specified",
            "Phone": phones[i] if i < len(phones) else "Not specified",
            "Email": emails[i] if i < len(emails) else "Not specified",
            "Website": websites[i] if i < len(websites) else "Not specified",
            "Specialization": "Not specified"
        })

    return therapist_data

def debug_search_results(search_results: str):
    print("Search Results")
    print(f"Length: {len(search_results)}")
    print(f"First 500 chars: {search_results[:500]}")

def therapist_search_agent(state: State, config: RunnableConfig, *, store: PostgresStore, llm, tools):
    last_message = state["messages"][-1]
    user_id = config["configurable"]["user_id"]
    location = state.get("location")

    print(f"User message: '{last_message.content}'")
    print(f"Location type: {type(location)}")
    print(f"Location is None: {location is None}")
    print(f"Location stripped: '{location.strip() if location else 'None'}'")
    print(f"Full state: {state}")

    if not location or location.strip() == "" or location.lower() == "italy":
        error_message = "I need to know your location to search for therapists. Please tell me which city you're in."
        save_conversation_memory(store, user_id, last_message.content, error_message, "therapist_search")
        return {"messages": [AIMessage(content=error_message)]}

    print(f"Searching for therapists in: '{location}'")

    therapist_tool = None
    for tool in tools:
        if tool.name == "search_therapists":
            therapist_tool = tool
            break

    if not therapist_tool:
        error_message = f"I don't have access to search tools right now. Please try searching online for 'psicologo {location}' or contact your local health authority for mental health professional referrals."
        save_conversation_memory(store, user_id, last_message.content, error_message, "therapist_search")
        return {"messages": [AIMessage(content=error_message)]}

    if "italia" not in location.lower() and "italy" not in location.lower():
        search_query = f"{location} Italia"
    else:
        search_query = location

    print(f"Final search query: '{search_query}'")

    try:
        print(f"Searching with query: {search_query}")
        search_results = therapist_tool.func(search_query)

        if not search_results or search_results == "No search results found":
            fallback_message = f"""I wasn't able to find specific therapist information for {location} right now. Here are some ways to find mental health professionals in your area:

    1. **Online search**: Try searching for "psicologo {location}" or "psicoterapeuta {location}"
    2. **Professional directories**: Check the Ordine degli Psicologi (regional psychology boards)
    3. **Local healthcare**: Contact your family doctor or local ASL for referrals
    4. **Emergency support**: If you need immediate help, contact emergency services or crisis hotlines

    Would you like me to help you with anything else regarding mental health support?"""

            save_conversation_memory(store, user_id, last_message.content, fallback_message, "therapist_search")
            return {"messages": [AIMessage(content=fallback_message)]}

        therapist_data = extract_therapist_data(search_results, llm)

        if therapist_data:
                therapist_table = format_therapist_table(therapist_data)
                response_content = f"""I found some mental health professionals in {location}:
    {therapist_table}

    *Note: Please verify contact information before reaching out, as details may have changed.*
    Would you like me to help you with anything else regarding mental health support?"""

        else:
            response_content = f"""I searched for mental health professionals in {location} but couldn't extract specific contact information. Here are alternative ways to find help:

        1. **Search online for:**
           - "psicologo {location}"
           - "psicoterapeuta {location}"
           - "centro psicologico {location}"

        2. **Professional directories:**
           - Ordine degli Psicologi (regional psychology boards)
           - Local healthcare directories

        3. **Contact your:**
           - Family doctor for referrals
           - Local health authority (ASL)
           - Community mental health centers

        Would you like me to help you with anything else?"""

        save_conversation_memory(store, user_id, last_message.content, response_content, "therapist_search")
        return {"messages": [AIMessage(content=response_content)]}
    except Exception as e:
        print(f"Error in therapist search: {e}")
        error_message = f"""I encountered an error while searching for therapists in {location}. Please try:

        1. Searching online for "psicologo {location}"
        2. Contacting your local health authority for referrals
        3. Asking your family doctor for recommendations

        Would you like me to help you with anything else?"""

        save_conversation_memory(store, user_id, last_message.content, error_message, "therapist_search")
        return {"messages": [AIMessage(content=error_message)]}

def format_therapist_table(therapist_data: List[Dict[str, str]]) -> str:
    if not therapist_data:
        return "No therapist information could be extracted from the search results."

    from tabulate import tabulate

    headers = ["Name", "Address", "Phone", "Email", "Website", "Specialization"]
    table_data = []

    for therapist in therapist_data:
        row = [
            therapist.get("Name", "Not specified"),
            therapist.get("Address", "Not specified"),
            therapist.get("Phone", "Not specified"),
            therapist.get("Email", "Not specified"),
            therapist.get("Website", "Not specified"),
            therapist.get("Specialization", "Not specified")
        ]
        table_data.append(row)

    table = tabulate(table_data, headers=headers, tablefmt="grid", maxcolwidths=[20, 30, 15, 25, 30, 25])

    return f"\nMENTAL HEALTH PROFESSIONALS - {len(therapist_data)} found:\n\n{table}"


def has_tool_calls(state: State):
    if not state.get('messages'):
        return False
    last_message = state['messages'][-1]
    return hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0


# GRAPH SETUP
def create_graph(llm, tools, store, checkpointer):
    retrieve = ToolNode(tools)
    graph_builder = StateGraph(State)

    graph_builder.add_node("classifier", lambda state: classify_message(state, llm))
    graph_builder.add_node("router", router)
    graph_builder.add_node("ask_location",
                           lambda state, config: ask_location_agent(state, config, store=store, llm=llm))
    graph_builder.add_node("friend", lambda state, config: friend_agent(state, config, store=store, llm=llm))
    graph_builder.add_node("logical", lambda state, config: logical_agent(state, config, store=store, llm=llm))
    graph_builder.add_node("therapist_search",
                           lambda state, config: therapist_search_agent(state, config, store=store, llm=llm,
                                                                        tools=tools))
    graph_builder.add_node("retrieve", retrieve)

    # Edges
    graph_builder.add_edge(START, "classifier")
    graph_builder.add_edge("classifier", "router")

    # Conditional edges
    graph_builder.add_conditional_edges(
        "router",
        lambda state: state.get("next"),
        {
            "ask_location": "ask_location",
            "friend": "friend",
            "logical": "logical",
            "therapist_search": "therapist_search"
        }
    )

    # Conditional edges for tool usage
    graph_builder.add_edge("ask_location", END )

    graph_builder.add_conditional_edges(
        "therapist_search",
        has_tool_calls,
        {True: "retrieve", False: END}
    )

    graph_builder.add_edge("friend", END)
    graph_builder.add_edge("logical", END)
    graph_builder.add_edge("retrieve", END)

    # Compile graph
    graph = graph_builder.compile(
        checkpointer=checkpointer,
        store=store,
    )
    print("Graph compiled successfully")
    return graph


# MAIN CHATBOT FUNCTION
def run_chatbot():
    llm = initialize_llm()
    retriever = setup_document_retrieval()
    tools = create_tools(retriever)

    with (
        PostgresStore.from_conn_string(DB_URI) as store,
        PostgresSaver.from_conn_string(DB_URI) as checkpointer,
    ):
        store.setup()
        checkpointer.setup()
        print("Database connections established")

        # Graph
        graph = create_graph(llm, tools, store, checkpointer)

        print("CHATBOT READY")
        user_id = input("Enter your user ID (or press Enter for new session): ").strip()
        if not user_id:
            user_id = str(uuid.uuid4())
            print(f" Created new user session: {user_id[:8]}")
        else:
            print(f"Welcome back! User ID: {user_id[:8]}")
            print("\nCONVERSATION SUMMARY\n")
            summary = generate_user_summary(store, user_id, llm)
            print(f"\n{summary}\n")
            print("=" * 50)

        thread_id = f"conversation_{user_id}"
        config = RunnableConfig(
            configurable={
                "user_id": user_id,
                "thread_id": thread_id,
                "checkpoint_ns": "chat_session",
            }
        )
        print("Chatbot is ready! Type 'quit', 'exit', or 'goodbye' to end the conversation.")

        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "goodbye", "bye"]:
                print("\nGoodbye!")
                break

            state = {
                "messages": [HumanMessage(content=user_input)],
                "message_types": None,
                "needs_therapist_search": None,
                "location": None,
                "next": None
            }

            try:
                result = graph.invoke(state, config=config,)
                if result.get("messages"):
                    messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
                    if messages:
                        last_message = messages[-1]
                        if hasattr(last_message, 'content') and last_message.content:
                            print(f"Assistant: {last_message.content}")
                print("\n" + "-" * 50)
            except Exception as e:
                print(f"Error: {e}")
                print("\n" + "-" * 50)


# MAIN EXECUTION
if __name__ == "__main__":
    run_chatbot()
    print("Chatbot session ended")
