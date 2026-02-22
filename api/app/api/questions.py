import urllib.parse

from asyncpg import UniqueViolationError
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.data.connection import Database
from app.data.db import get_db
from app.data.questions import (
    create_question_query,
    delete_question_query,
    get_nth_question_query,
    get_question_by_name_query,
    get_question_names_query,
    get_questions_query,
    get_questions_without_embeddings_query,
    update_question_query,
)
from app.data.questions import (
    get_closest_questions as query_closest_questions,
)
from app.llms.embeddings import generate_embeddings, stream_fill_embeddings
from app.llms.models import MODEL_EMBEDDINGS_COLUMNS, Model
from app.schemas.questions import (
    ClosestQuestionsSchema,
    CreateQuestionSchema,
    FillEmbeddingsSchema,
    QuestionSchema,
    UpdateQuestionSchema,
)
from app.utils.auth import verify_api_key

db_dep = Depends(get_db)
api_key_dep = Depends(verify_api_key)

router = APIRouter(
    prefix="/questions",
    tags=["Questions"],
    dependencies=[db_dep],
)


@router.get(
    "/list",
    summary="List all questions",
    description="Returns a list of all stored questions.",
    response_model=list[QuestionSchema],
    status_code=status.HTTP_200_OK,
    operation_id="listQuestions",
)
async def list_questions(db: Database = db_dep) -> list[QuestionSchema]:
    return await get_questions_query(db)


@router.get(
    "/names",
    summary="List question names",
    description="Returns only the names of all stored questions.",
    response_model=list[str],
    status_code=status.HTTP_200_OK,
    operation_id="listQuestionNames",
)
async def list_question_names(db: Database = db_dep) -> list[str]:
    return await get_question_names_query(db)


@router.get(
    "/closest",
    summary="Find closest questions",
    description="Given a query and an embedding model, return the top N closest question names.",
    response_model=list[QuestionSchema],
    status_code=status.HTTP_200_OK,
    responses={status.HTTP_404_NOT_FOUND: {"description": "No questions found"}},
    operation_id="getClosestQuestions",
)
async def closest_questions(
    params: ClosestQuestionsSchema = Depends(),  # noqa: B008
    db: Database = db_dep,
) -> list[QuestionSchema]:
    prompt_embedding = await generate_embeddings(params.prompt, params.embeddings_model)
    results = await query_closest_questions(
        db,
        prompt_embedding,
        params.embeddings_model,
        limit=params.limit,
        threshold=params.threshold,
    )
    return results


@router.get(
    "/name/{name:path}",
    summary="Get question by name",
    description="Return the matching question, 404 if not found.",
    response_model=QuestionSchema,
    status_code=status.HTTP_200_OK,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Question not found"}},
    operation_id="getQuestionByName",
)
async def get_question_by_name(
    name: str,
    db: Database = db_dep,
) -> QuestionSchema:
    decoded = urllib.parse.unquote(name)
    question = await get_question_by_name_query(db, decoded)
    if not question:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Question '{decoded}' not found",
        )
    return question


@router.post(
    "/",
    summary="Create a new question",
    description="Insert a new question. 400 if one with the same name exists.",
    response_model=QuestionSchema,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Question already exists or creation failed",
        },
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Invalid or missing API Key",
        },
    },
    dependencies=[api_key_dep],
    operation_id="createQuestion",
)
async def create_question(
    payload: CreateQuestionSchema,
    db: Database = db_dep,
) -> QuestionSchema:
    try:
        created = await create_question_query(db, payload)
    except UniqueViolationError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Question '{payload.name}' already exists",
        ) from None
    if not created:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create question",
        )
    return created


@router.put(
    "/{name:path}",
    summary="Update an existing question",
    description="Apply partial updates, 404 if not found.",
    response_model=QuestionSchema,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "No fields to update or update failed",
        },
        status.HTTP_404_NOT_FOUND: {"description": "Question not found"},
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Invalid or missing API Key",
        },
    },
    dependencies=[api_key_dep],
    operation_id="updateQuestion",
)
async def update_question(
    name: str,
    payload: UpdateQuestionSchema,
    db: Database = db_dep,
) -> QuestionSchema:
    decoded = urllib.parse.unquote(name)
    existing = await get_question_by_name_query(db, decoded)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Question '{decoded}' not found",
        )
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No updates provided",
        )
    updated = await update_question_query(db, decoded, payload)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to update question",
        )
    return updated


@router.delete(
    "/{name:path}",
    summary="Delete a question",
    description="Delete the question, and return the deleted record.",
    response_model=QuestionSchema,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Question not found"},
        status.HTTP_401_UNAUTHORIZED: {"description": "Invalid or missing API Key"},
    },
    dependencies=[api_key_dep],
    operation_id="deleteQuestion",
)
async def delete_question(
    name: str,
    db: Database = db_dep,
) -> QuestionSchema:
    decoded = urllib.parse.unquote(name)
    existing = await get_question_by_name_query(db, decoded)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Question '{decoded}' not found",
        )
    await delete_question_query(db, decoded)
    return existing


@router.get(
    "/nth/{n}",
    summary="Get the Nth question",
    description="Return the Nth question in alphabetical order (0-based), 404 if out of range.",
    response_model=QuestionSchema,
    status_code=status.HTTP_200_OK,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Index out of range"}},
    operation_id="getNthQuestion",
)
async def get_nth_question(
    n: int,
    db: Database = db_dep,
) -> QuestionSchema:
    question = await get_nth_question_query(db, n)
    if not question:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No question at index {n}",
        )
    return question


@router.post(
    "/fill",
    summary="Fill embeddings with progress",
    description="Streams back per-row progress as Server-Sent Events (SSE).",
    response_class=StreamingResponse,
    status_code=status.HTTP_200_OK,
    operation_id="fillEmbeddings",
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Unsupported model"},
        status.HTTP_401_UNAUTHORIZED: {"description": "Invalid or missing API Key"},
    },
    dependencies=[api_key_dep],
)
async def fill_embeddings(
    payload: FillEmbeddingsSchema,
    db: Database = db_dep,
) -> StreamingResponse:
    return await stream_fill_embeddings(
        db,
        payload.embeddings_model,
        questions=payload.questions,
        all_questions=payload.all_questions,
        all_models=payload.all_models,
    )


@router.get(
    "/unfilled",
    summary="List questions with unfilled embeddings",
    description="Returns a list of questions that have unfilled embeddings for the specified model.",
    response_model=list[QuestionSchema],
    status_code=status.HTTP_200_OK,
    operation_id="listUnfilledQuestions",
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Unsupported model specified",
        },
    },
)
async def list_unfilled_questions(
    model: Model = Query(description="The model to check for unfilled embeddings"),  # noqa: B008
    db: Database = db_dep,
) -> list[QuestionSchema]:
    if model not in MODEL_EMBEDDINGS_COLUMNS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model: {model}",
        )

    results = await get_questions_without_embeddings_query(db, model)

    return results
