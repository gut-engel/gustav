"""API route definitions for authentication and chat handling.
This file wires up the missing handlers referenced in the plan.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from typing import List, Any

from gustav.server.auth import Accounts, Identity, COOKIE
from gustav.server.chats import Chats, Chat

router = APIRouter()

# ----- Schemas -----
class LoginRequest(BaseModel):
    name: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)

class TokenRequest(BaseModel):
    name: str = Field(..., min_length=1)
    label: str = "cli"
    days: int | None = None

class ChatCreateRequest(BaseModel):
    title: str = ""
    source: str = "web"

class ChatRenameRequest(BaseModel):
    title: str = Field(..., min_length=1)

# ----- Dependency helpers -----
def get_accounts(request: Request) -> Accounts:
    return request.app.state.accounts

def get_chats(request: Request) -> Chats:
    # The daemon instance is stored in app.state.daemon
    return request.app.state.daemon.chats

def get_identity(request: Request) -> Identity:
    # Re‑use the same logic as app.identify() – calling the helper directly would be a circular import.
    # First, check for Bearer token
    auth_header = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if auth_header:
        identity = request.app.state.accounts.identify_token(auth_header)
        if identity:
            return identity
    # Then, check cookie
    cookie = request.cookies.get(COOKIE)
    if cookie:
        identity = request.app.state.accounts.identify_session(cookie)
        if identity:
            return identity
    raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Bearer"})

# ----- Auth endpoints -----
@router.post("/auth/login")
def login(data: LoginRequest, request: Request, response: Response, accounts: Accounts = Depends(get_accounts)):
    user = accounts.authenticate(data.name, data.password)
    token = accounts.start_session(user, user_agent=request.headers.get("user-agent", ""))
    response.set_cookie(
        key=COOKIE,
        value=token,
        httponly=True,
        samesite="strict",
        secure=False,  # Set True when serving over HTTPS
    )
    return {"message": "logged in", "user": user.to_dict()}

@router.post("/auth/token")
def issue_token(data: TokenRequest, request: Request, accounts: Accounts = Depends(get_accounts), identity: Identity = Depends(get_identity)):
    # Owner can issue for anyone; regular users only for themselves
    if not identity.is_owner and identity.user.name != data.name:
        raise HTTPException(status_code=403, detail="Only owner can issue tokens for other accounts")
    token = accounts.issue_token(data.name, label=data.label, days=data.days)
    return {"token": token}

# ----- Chat endpoints -----
@router.get("/chats", response_model=List[Chat])
def list_chats(identity: Identity = Depends(get_identity), chats: Chats = Depends(get_chats)):
    return chats.all(user_id=identity.user_id)

@router.post("/chats", response_model=Chat)
def create_chat(data: ChatCreateRequest, identity: Identity = Depends(get_identity), chats: Chats = Depends(get_chats)):
    return chats.create(user_id=identity.user_id, title=data.title, source=data.source)

@router.get("/chats/{chat_id}", response_model=Chat)
def get_chat(chat_id: int, identity: Identity = Depends(get_identity), chats: Chats = Depends(get_chats)):
    return chats.get(chat_id, user_id=identity.user_id)

@router.patch("/chats/{chat_id}", response_model=Chat)
def rename_chat(chat_id: int, data: ChatRenameRequest, identity: Identity = Depends(get_identity), chats: Chats = Depends(get_chats)):
    return chats.rename(chat_id, title=data.title, user_id=identity.user_id)

@router.delete("/chats/{chat_id}")
def archive_chat(chat_id: int, identity: Identity = Depends(get_identity), chats: Chats = Depends(get_chats)):
    chats.archive(chat_id, user_id=identity.user_id)
    return {"message": "archived"}
