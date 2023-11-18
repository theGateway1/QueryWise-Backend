import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from openai import OpenAI
import google.generativeai as genai
import requests
import pandas as pd
from dotenv import load_dotenv
from typing import List, Literal
import random

load_dotenv()
app = FastAPI()

# LLM configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:1234")  # Default Ollama port

openai_client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)


# Pydantic models
class DBCredentials(BaseModel):
    db_user: str
    db_password: str
    db_host: str
    db_port: str
    db_name: str


class QueryRequest(BaseModel):
    question: str
    db_credentials: DBCredentials
    llm_choice: Literal["openai", "gemini", "local"] = "openai"


class DBStructureRequest(BaseModel):
    db_credentials: DBCredentials


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    db_credentials: DBCredentials
    llm_choice: Literal["openai", "gemini", "local"] = "openai"


# LLM choice function
def choose_llm(llm_choice: str):
    if llm_choice == "openai":
        return nl_to_sql_openai
    elif llm_choice == "gemini":
        return nl_to_sql_gemini
    elif llm_choice == "local":
        return nl_to_sql_local
    else:
        raise ValueError("Invalid LLM choice")


# OpenAI function
def nl_to_sql_openai(question: str, table_info: str) -> str:
    prompt = f"""
    The sql code should not have ``` in beginning or end and sql word in output
    Given the following tables in a PostgreSQL database:

    {table_info}

    Convert the following natural language question to a SQL query:

    {question}

    Return only the SQL query, without any additional explanation.

    """

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a SQL expert. Convert natural language questions to SQL queries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


def nl_to_sql_gemini(question: str, table_info: str) -> str:
    prompt = f"""
    Given the following tables in a PostgreSQL database, Also the sql code should not have ``` in beginning or end and sql word in output:

    {table_info}

    Convert the following natural language question to a SQL query:

    {question}

    Return only the SQL query, without any additional explanation.
    """

    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)

    return response.text.strip()


# Local LLM function (using Ollama)
def nl_to_sql_local(question: str, table_info: str) -> str:
    prompt = f"""
    Given the following tables in a PostgreSQL database:

    {table_info}

    Convert the following natural language question to a SQL query:

    {question}

    Return only the SQL query, without any additional explanation.
    Also the sql code should not have ``` in beginning or end and sql word in output
    """

    response = requests.post(
        f"{LOCAL_LLM_URL}/v1/chat/completions",
        json={
            "model": "defog/sqlcoder-7b-2/sqlcoder-7b-q5_k_m.gguf",  # or any other model you have in Ollama
            "messages": [
                {
                    "role": "system",
                    "content": prompt
                }
            ]
            ,
            "stream": False
        }
    )
    response = response.json()
    print(response['choices'][0]['message']['content'])

    if response:
        return response['choices'][0]['message']['content'].strip()
    else:
        raise HTTPException(status_code=500, detail="Error in local LLM request")


def get_db_structure(db_credentials: DBCredentials):
    db_url = f"postgresql://{db_credentials.db_user}:{db_credentials.db_password}@{db_credentials.db_host}:{db_credentials.db_port}/{db_credentials.db_name}"
    # db_url = "sqlite:///users.db"
    engine = create_engine(db_url)
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = 'public';"))
        table_info = pd.DataFrame(result.fetchall(), columns=result.keys())
    # with engine.connect() as connection:
    #     result = connection.execute(
    #         text("SELECT m.name as table_name, p.name as column_name "
    #              "FROM sqlite_master m "
    #              "JOIN pragma_table_info(m.name) p "
    #              "ON m.type = 'table' "
    #              "ORDER BY table_name, column_name;")
    #     )
    #     table_info = pd.DataFrame(result.fetchall(), columns=result.keys())
    return table_info.groupby('table_name')['column_name'].apply(list).to_dict()


def execute_sql_query(query: str, db_credentials: DBCredentials):
    db_url = f"postgresql://{db_credentials.db_user}:{db_credentials.db_password}@{db_credentials.db_host}:{db_credentials.db_port}/{db_credentials.db_name}"
    # db_url = "sqlite:///users.db"
    engine = create_engine(db_url)
    with engine.connect() as connection:
        result = connection.execute(text(query))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df.to_dict(orient="records")


def format_response_with_llm(sql_query: str, query_results: str, llm_choice: str) -> str:
    prompt = f"""
    Analyze the following query results and provide insights:

    Results: {query_results}

    Please provide a clear and concise analysis of the data. Focus on key trends, patterns, or notable information in the results. Use markdown formatting to structure your response, including:

    - Headers for main sections
    - Bullet points or numbered lists for key points
    - Bold or italic text for emphasis
    - Code blocks for any numerical data or examples

    Your analysis should be informative and easy to understand for someone looking at this data.
    """

    if llm_choice == "openai":
        formatted_response = format_response_openai(prompt)
    elif llm_choice == "gemini":
        formatted_response = format_response_gemini(prompt)
    elif llm_choice == "local":
        formatted_response = format_response_local(prompt)
    else:
        raise ValueError("Invalid LLM choice")

    # Add the SQL query at the end without displaying it in the chat
    formatted_response += f"\n\n[SQL_QUERY]{sql_query}[/SQL_QUERY]"

    return formatted_response


def format_response_openai(prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You are a data analyst providing insights on query results. Use markdown formatting in your responses."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def format_response_gemini(prompt: str) -> str:
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text.strip()


def format_response_local(prompt: str) -> str:
    response = requests.post(
        f"{LOCAL_LLM_URL}/v1/chat/completions",
        json={
            "model": "defog/sqlcoder-7b-2/sqlcoder-7b-q5_k_m.gguf",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a data analyst providing insights on query results. Use markdown formatting in your responses."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        }
    )
    response_json = response.json()
    return response_json['choices'][0]['message']['content'].strip()


# def format_response_with_llm(sql_query: str, query_results: str) -> str:
#     prompt = f"""
#     Analyze the following query results and provide insights:
#
#     Results: {query_results}
#
#     Please provide a clear and concise analysis of the data. Focus on key trends, patterns, or notable information in the results. Use markdown formatting to structure your response, including:
#
#     - Headers for main sections
#     - Bullet points or numbered lists for key points
#     - Bold or italic text for emphasis
#     - Code blocks for any numerical data or examples
#
#     Your analysis should be informative and easy to understand for someone looking at this data.
#     """
#
#     response = openai_client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system",
#              "content": "You are a data analyst providing insights on query results. Use markdown formatting in your responses."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.7
#     )
#
#     formatted_response = response.choices[0].message.content.strip()
#
#     # Add the SQL query at the end without displaying it in the chat
#     formatted_response += f"\n\n[SQL_QUERY]{sql_query}[/SQL_QUERY]"
#
#     return formatted_response
# Gemini function


# Existing functions (get_db_structure, execute_sql_query, format_response_with_llm) remain unchanged

@app.post("/query")
async def query(request: QueryRequest):
    try:
        db_structure = get_db_structure(request.db_credentials)
        table_info_str = "\n".join(
            [f"Table: {table}, Columns: {', '.join(columns)}" for table, columns in db_structure.items()])

        # Choose LLM based on request
        nl_to_sql_func = choose_llm(request.llm_choice)

        # Convert natural language to SQL
        sql_query = nl_to_sql_func(request.question, table_info_str)
        print(sql_query)

        # Execute the SQL query
        results = execute_sql_query(sql_query, request.db_credentials)

        return {
            "question": request.question,
            "sql_query": sql_query,
            "results": results
        }
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        db_structure = get_db_structure(request.db_credentials)

        system_message = f"""You are a helpful AI assistant that can query a PostgreSQL database. 
        When generating SQL queries, do not include ``` or 'sql' tags. Only return the raw SQL query.
        Here's the database schema: {db_structure}
        """

        messages = [{"role": "system", "content": system_message}] + [m.dict() for m in request.messages]

        # Choose LLM based on request
        if request.llm_choice == "openai":
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0,
            )
            ai_message = response.choices[0].message.content.strip()
        elif request.llm_choice == "gemini":
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(
                [messages[i]['content'] for i in range(len(messages))])  # Only using the last message for simplicity
            ai_message = response.text.strip()
        elif request.llm_choice == "local":
            response = requests.post(
                f"{LOCAL_LLM_URL}/v1/chat/completions",
                json={
                    "model": "defog/sqlcoder-7b-2/sqlcoder-7b-q5_k_m.gguf",  # or any other model you have in Ollama
                    "messages": messages,
                    "stream": False
                }
            )
            response = response.json()

            if response:
                ai_message = response['choices'][0]['message']['content'].strip()

            else:
                raise HTTPException(status_code=500, detail="Error in local LLM request")
        else:
            raise ValueError("Invalid LLM choice")

        # Check if the AI's response contains a SQL query
        if "SELECT" in ai_message.upper():
            try:
                results = execute_sql_query(ai_message, request.db_credentials)
                formatted_response = format_response_with_llm(ai_message, str(results), request.llm_choice)

                return {
                    "role": "assistant",
                    "content": formatted_response,
                    "tabular_data": results
                }
            except Exception as e:
                error_message = f"Error executing query: {str(e)}"
                formatted_response = format_response_with_llm(ai_message, error_message, request.llm_choice)
                return {"role": "assistant", "content": formatted_response}
        else:
            return {"role": "assistant", "content": ai_message}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/db-structure")
async def get_db_structure_endpoint(request: DBStructureRequest):
    try:
        structure = get_db_structure(request.db_credentials)
        return {"structure": structure}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

class HealthData(BaseModel):
    average_heart_rate: float
    average_temperature: float
    average_ecg: float
    average_spo2: float


def get_average_sensor_data(user_id: int, days: int = 800):


    return HealthData(
        average_heart_rate=random.randint(60, 100),
        average_temperature=random.randint(96, 100),
        average_ecg=random.randint(60, 100),
        average_spo2=random.randint(90, 100)
    )


@app.get("/api/health_data/{user_id}")
async def health_data_api(user_id: int):
    data = get_average_sensor_data(user_id)
    if data is None:
        print(f"No data found for user {user_id}")
        raise HTTPException(status_code=404, detail="No data found for this user")
    return data