from pydantic import BaseModel, Field
from typing import Any, Optional, List

class RegisterRagRequest(BaseModel):
    rag_id: str
    label: str
    owner: Optional[str] = None
    domain: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    config: dict[str, Any]
    route_examples: Optional[List[str]] = None

    data_uri: Optional[str] = None
    index_uri: Optional[str] = None
    data_hash: Optional[str] = None
    description: Optional[str] = None

class RegisterRagResponse(BaseModel):
    rag_id: str
    version: int
    status: str

class ActivateRagResponse(BaseModel):
    rag_id: str
    active_version: int
    status: str
    description: Optional[str] = None

class ActiveRagOut(BaseModel):
    rag_id: str
    label: str
    domain: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    active_version: int
    config: dict[str, Any]
    route_examples: List[str]
    description: Optional[str] = None
    data_uri: Optional[str] = None
    index_uri: Optional[str] = None
