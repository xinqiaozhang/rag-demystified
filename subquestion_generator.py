import json
from typing import List
from enum import Enum

from instructor import OpenAISchema
from pydantic import Field, create_model
from openai_utils import llm_call
import pdb

# DEFAULT_SUBQUESTION_GENERATOR_PROMPT = """
#                  You are an AI agent that takes a complex user question and returns a list of simple subquestions to answer the user's question.
#                  You are provided a set of functions and data sources that you can use to answer each subquestion.
#                  If the user question is simple, just return the user question, the function, and the data source to use.
#                  You can only use the provided functions and data sources.
#                  The subquestions should be complete questions that can be answered by a single function and a single data source.
#                  """

# DEFAULT_SUBQUESTION_GENERATOR_PROMPT = """
#     You are an AI assistant that specializes in breaking down complex questions into simpler, manageable sub-questions.
#     When presented with a complex user question, your role is to generate a list of sub-questions that, when answered, will comprehensively address the original query.
#     You have at your disposal a pre-defined set of functions and data sources to utilize in answering each sub-question.
#     If a user question is straightforward, your task is to return the original question, identifying the appropriate function and data source to use for its solution.
#     Please remember that you are limited to the provided functions and data sources, and that each sub-question should be a full question that can be answered using a single function and a single data source.
# """

DEFAULT_SUBQUESTION_GENERATOR_PROMPT = """
    You are an AI assistant that specializes in breaking down complex questions into simpler, manageable sub-questions.
    You have at your disposal a pre-defined set of functions and files to utilize in answering each sub-question.
    Please remember that your output should only contain the provided function names and file names, and that each sub-question should be a full question that can be answered using a single function and a single file.
"""

DEFAULT_USER_TASK = ""


class FunctionEnum(str, Enum):
    """The function to use to answer the questions.
    Use vector_retrieval for fact-based questions such as demographics, sports, arts and culture, etc.
    Use llm_retrieval for summarization questions, such as positive aspects, history, etc.
    """

    VECTOR_RETRIEVAL = "vector_retrieval"
    LLM_RETRIEVAL = "llm_retrieval"
    # top_k_retrieval = "top_k_retrieval"
    # keyword_search = "keyword_search"
    # ranking = 'ranking'
    # most_special_place = 'most_special_place'
    


def generate_subquestions(
    question,
    file_names: List[str] = None,
    system_prompt=DEFAULT_SUBQUESTION_GENERATOR_PROMPT,
    user_task=DEFAULT_USER_TASK,
    llm_model="gpt-3.5-0613",
):
    """Generates a list of subquestions from a user question along with the
    file name and the function to use to answer the question using OpenAI LLM.
    """
    FilenameEnum = Enum("FilenameEnum", {x.upper(): x for x in file_names})
    FilenameEnum.__doc__ = f"The names of the file to use to answer the corresponding subquestion - e.g. {file_names[0]}"

    # Create pydantic class dynamically
    QuestionBundle = create_model(
        "QuestionBundle",
        question=(
            str,
            Field(
                None, description="The subquestion extracted from the user's question"
            ),
        ),
        function=(FunctionEnum, Field(None)),
        file_name=(FilenameEnum, Field(None)),
    )

    SubQuestionBundleList = create_model(
        "SubQuestionBundleList",
        subquestion_bundle_list=(
            List[QuestionBundle],
            Field(
                None,
                description="A list of subquestions - each item in the list contains a question, a function, and a file name",
            ),
        ),
        __base__=OpenAISchema,
    )

    user_prompt = f"{user_task}\n Here is the user question: {question}"

    few_shot_examples = [
        {
            "role": "user",
            "content": "Compare the population of Atlanta and Toronto?",
        },
        {
            "role": "function",
            "name": "SubQuestionBundleList",
            "content": """
            {
                "subquestion_bundle_list": [
                    {
                        "question": "What is the population of Atlanta?",
                        "function": "vector_retrieval",
                        "file_name": "Atlanta"
                    },
                    {
                        "question": "What is the population of Toronto?"
                        "function": "vector_retrieval",
                        "file_name": "Toronto"
                    }
                ]
            }""",
        },
        {
            "role": "user",
            "content": "Summarize the history of Chicago and Houston.",
        },
        {
            "role": "function",
            "name": "SubQuestionBundleList",
            "content": """
            {
                "subquestion_bundle_list": [
                    {
                        "question": "What is the history of Chicago?",
                        "function": "llm_retrieval",
                        "file_name": "Chicago"
                    },
                    {
                        "question": "What is the history of Houston?",
                        "function": "llm_retrieval",
                        "file_name": "Houston"
                    }
                ]
            }""",
        },
    ]

    response, cost = llm_call(
        model=llm_model,
        function_schema=[SubQuestionBundleList.openai_schema],
        output_schema={"name": SubQuestionBundleList.openai_schema["name"]},
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        few_shot_examples=few_shot_examples,
    )

    subquestions_list = json.loads(response.choices[0].message.function_call.arguments)
    # subquestions_list
    # pdb.set_trace()
    subquestions_pydantic_obj = SubQuestionBundleList(**subquestions_list)
    subquestions_list = subquestions_pydantic_obj.subquestion_bundle_list
    
    return subquestions_list, cost
