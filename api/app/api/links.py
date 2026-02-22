import urllib.parse

from asyncpg import UniqueViolationError
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.data.connection import Database
from app.data.db import get_db
from app.data.links import (
    create_link_query,
    delete_link_query,
    get_link_by_name_query,
    get_link_names_query,
    get_links_query,
    get_nth_link_query,
    update_link_query,
)
from app.schemas.links import CreateLinkSchema, LinkSchema, UpdateLinkSchema
from app.utils.auth import verify_api_key

db_dep = Depends(get_db)
api_key_dep = Depends(verify_api_key)

router = APIRouter(
    prefix="/links",
    tags=["Links"],
    dependencies=[db_dep],
)


@router.get(
    "/list",
    summary="List all links",
    description="Return a list of all stored links.",
    response_model=list[LinkSchema],
    status_code=status.HTTP_200_OK,
    operation_id="listLinks",
)
async def list_links(db: Database = db_dep) -> list[LinkSchema]:
    return await get_links_query(db)


@router.get(
    "/names",
    summary="List link names",
    description="Return only the names (keys) of all stored links.",
    response_model=list[str],
    status_code=status.HTTP_200_OK,
    operation_id="listLinkNames",
)
async def list_link_names(db: Database = db_dep) -> list[str]:
    return await get_link_names_query(db)


@router.get(
    "/name/{name:path}",
    summary="Fetch a link by name",
    description="Return the matching link, or 404 if not found.",
    response_model=LinkSchema,
    status_code=status.HTTP_200_OK,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Link not found"}},
    operation_id="getLinkByName",
)
async def get_link_by_name(name: str, db: Database = db_dep) -> LinkSchema:
    decoded = urllib.parse.unquote(name)
    link = await get_link_by_name_query(db, decoded)
    if not link:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Link '{decoded}' not found",
        )
    return link


@router.post(
    "/",
    summary="Create a new link",
    description="Create a link with a unique name. Returns 400 if name already exists.",
    response_model=LinkSchema,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Link already exists or creation failed",
        },
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Invalid or missing API Key",
        },
    },
    dependencies=[api_key_dep],
    operation_id="createLink",
)
async def create_link(
    payload: CreateLinkSchema,
    db: Database = db_dep,
) -> LinkSchema:
    try:
        created = await create_link_query(db, payload)
    except UniqueViolationError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Link '{payload.name}' already exists",
        ) from None
    if not created:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create link",
        )
    return created


@router.put(
    "/{name:path}",
    summary="Update an existing link",
    description="Apply partial updates, or 404 if missing.",
    response_model=LinkSchema,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "No updates provided or update failed",
        },
        status.HTTP_404_NOT_FOUND: {"description": "Link not found"},
        status.HTTP_401_UNAUTHORIZED: {"description": "Invalid or missing API Key"},
    },
    dependencies=[api_key_dep],
    operation_id="updateLink",
)
async def update_link(
    name: str,
    payload: UpdateLinkSchema,
    db: Database = db_dep,
) -> LinkSchema:
    decoded = urllib.parse.unquote(name)
    existing = await get_link_by_name_query(db, decoded)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Link '{decoded}' not found",
        )
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No updates provided",
        )
    updated = await update_link_query(db, decoded, updates)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to update link",
        )
    return updated


@router.delete(
    "/{name:path}",
    summary="Delete a link",
    description="Delete the link, and return the deleted record.",
    response_model=LinkSchema,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Link not found"},
        status.HTTP_401_UNAUTHORIZED: {"description": "Invalid or missing API Key"},
    },
    dependencies=[api_key_dep],
    operation_id="deleteLink",
)
async def delete_link(
    name: str,
    db: Database = db_dep,
) -> LinkSchema:
    decoded = urllib.parse.unquote(name)
    existing = await get_link_by_name_query(db, decoded)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Link '{decoded}' not found",
        )
    await delete_link_query(db, decoded)
    return existing


@router.get(
    "/nth/{n}",
    summary="Get the Nth link",
    description="Return the Nth link in alphabetical order (0-based), or 404 if out of range.",
    response_model=LinkSchema,
    status_code=status.HTTP_200_OK,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Index out of range"}},
    operation_id="getNthLink",
)
async def get_nth_link(
    n: int,
    db: Database = db_dep,
) -> LinkSchema:
    link = await get_nth_link_query(db, n)
    if not link:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No link at index {n}",
        )
    return link
