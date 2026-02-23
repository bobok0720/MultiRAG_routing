import os

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from .db import Base, engine, get_db
from .models import Rag, RagVersion
from .schemas import RegisterRagRequest, RegisterRagResponse, ActivateRagResponse, ActiveRagOut
from multirag.metadata_generate import generate_rag_metadata

app = FastAPI(title="RAG Registry API")

# MVP: auto-create tables (later you can migrate with Alembic)
Base.metadata.create_all(bind=engine)

def next_version(db: Session, rag_id: str) -> int:
    q = select(func.max(RagVersion.version)).where(RagVersion.rag_id == rag_id)
    mx = db.execute(q).scalar_one_or_none()
    return 1 if mx is None else int(mx) + 1

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/rags/register", response_model=RegisterRagResponse)
def register_rag(payload: RegisterRagRequest, db: Session = Depends(get_db)):
    rag = db.get(Rag, payload.rag_id)
    if rag is None:
        rag = Rag(
            rag_id=payload.rag_id,
            label=payload.label,
            owner=payload.owner,
            domain=payload.domain,
            tags=payload.tags,
            status="draft",
        )
        db.add(rag)
    else:
        rag.label = payload.label
        rag.owner = payload.owner
        rag.domain = payload.domain
        rag.tags = payload.tags

    route_examples = [str(x).strip() for x in (payload.route_examples or []) if str(x).strip()]
    description = (payload.description or "").strip()
    needs_generation = (not description) or (len(route_examples) == 0)

    if needs_generation:
        try:
            gen_description, gen_examples = generate_rag_metadata(
                rag_id=payload.rag_id,
                label=payload.label,
                domain=payload.domain,
                tags=payload.tags,
                data_uri=payload.data_uri,
                index_uri=payload.index_uri,
                config=payload.config,
                repo_root=os.getenv("REPO_ROOT", os.getcwd()),
                max_examples=6,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{e} Provide readable data via index_uri/chunks.jsonl or data_uri, "
                    "or submit description/route_examples explicitly."
                ),
            ) from e

        if not description:
            description = gen_description
        if len(route_examples) == 0:
            route_examples = gen_examples

    if len(route_examples) == 0:
        raise HTTPException(
            status_code=422,
            detail="route_examples is empty after validation/generation.",
        )

    v = next_version(db, payload.rag_id)
    rv = RagVersion(
        rag_id=payload.rag_id,
        version=v,
        config=payload.config,
        route_examples=route_examples,
        data_uri=payload.data_uri,
        index_uri=payload.index_uri,
        data_hash=payload.data_hash,
        description=description,
    )
    db.add(rv)
    db.commit()
    return RegisterRagResponse(rag_id=payload.rag_id, version=v, status=rag.status)

@app.post("/rags/{rag_id}/activate", response_model=ActivateRagResponse)
def activate_rag(rag_id: str, version: int, db: Session = Depends(get_db)):
    rag = db.get(Rag, rag_id)
    if rag is None:
        raise HTTPException(status_code=404, detail="Unknown rag_id")

    q = select(RagVersion).where(RagVersion.rag_id == rag_id, RagVersion.version == version)
    rv = db.execute(q).scalar_one_or_none()
    if rv is None:
        raise HTTPException(status_code=404, detail="Unknown version")

    rag.active_version = version
    rag.status = "active"
    db.commit()
    return ActivateRagResponse(rag_id=rag_id, active_version=version, description=rv.description,status=rag.status)

@app.get("/rags/active", response_model=list[ActiveRagOut])
def list_active(db: Session = Depends(get_db)):
    q = select(Rag).where(Rag.status == "active", Rag.active_version.is_not(None))
    rags = db.execute(q).scalars().all()

    out = []
    for r in rags:
        qv = select(RagVersion).where(RagVersion.rag_id == r.rag_id, RagVersion.version == r.active_version)
        rv = db.execute(qv).scalar_one_or_none()
        if rv is None:
            # active_version points to a version row that doesn't exist
            # Skip this rag or mark it as misconfigured instead of crashing.
            continue
        out.append(ActiveRagOut(
            rag_id=r.rag_id,
            label=r.label,
            domain=r.domain,
            tags=r.tags or [],
            active_version=r.active_version,
            config=rv.config,
            route_examples=rv.route_examples,
            description=rv.description,
            data_uri=rv.data_uri,
            index_uri=rv.index_uri,
        ))
    return out
