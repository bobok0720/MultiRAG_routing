from sqlalchemy import String, Integer, DateTime, ForeignKey, JSON, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .db import Base

class Rag(Base):
    __tablename__ = "rags"

    rag_id: Mapped[str] = mapped_column(String, primary_key=True)
    label: Mapped[str] = mapped_column(String, nullable=False)
    owner: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="draft")  # draft|active|disabled|deprecated
    domain: Mapped[str | None] = mapped_column(String, nullable=True)
    tags: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    active_version: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    versions = relationship("RagVersion", back_populates="rag", cascade="all, delete-orphan")

class RagVersion(Base):
    __tablename__ = "rag_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rag_id: Mapped[str] = mapped_column(ForeignKey("rags.rag_id"), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)

    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    route_examples: Mapped[list] = mapped_column(JSON, nullable=False)
    data_uri: Mapped[str | None] = mapped_column(String, nullable=True)   # where PDFs live
    index_uri: Mapped[str | None] = mapped_column(String, nullable=True)  # later (artifacts)
    data_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    description: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    rag = relationship("Rag", back_populates="versions")