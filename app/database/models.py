"""Metadata tables (schema 'app'). The uploaded data itself lives in schema 'user_data'."""
from datetime import datetime, timezone

from sqlalchemy import JSON, BigInteger, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = {"schema": "app"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255))
    table_name: Mapped[str] = mapped_column(String(100), unique=True)
    row_count: Mapped[int] = mapped_column(BigInteger)
    column_count: Mapped[int] = mapped_column(Integer)
    profile: Mapped[dict] = mapped_column(JSON)            # full profile shown in the UI
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    columns: Mapped[list["DatasetColumn"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan", order_by="DatasetColumn.position"
    )


class DatasetColumn(Base):
    __tablename__ = "dataset_columns"
    __table_args__ = {"schema": "app"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("app.datasets.id", ondelete="CASCADE"))
    position: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(100))
    original_name: Mapped[str] = mapped_column(String(255))
    kind: Mapped[str] = mapped_column(String(20))           # numeric/categorical/text/datetime/boolean
    sql_type: Mapped[str] = mapped_column(String(30))
    missing_count: Mapped[int] = mapped_column(BigInteger, default=0)
    unique_count: Mapped[int] = mapped_column(BigInteger, default=0)
    stats: Mapped[dict] = mapped_column(JSON, default=dict)
    top_values: Mapped[list] = mapped_column(JSON, default=list)

    dataset: Mapped[Dataset] = relationship(back_populates="columns")


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = {"schema": "app"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("app.datasets.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = {"schema": "app"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("app.conversations.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(20))           # "user" or "assistant"
    content: Mapped[str] = mapped_column(Text)
    resolved_question: Mapped[str | None] = mapped_column(Text, nullable=True)
    sql: Mapped[str | None] = mapped_column(Text, nullable=True)
    chart_spec: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str | None] = mapped_column(String(30), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class QueryLog(Base):
    __tablename__ = "query_logs"
    __table_args__ = {"schema": "app"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    question: Mapped[str] = mapped_column(Text)
    sql: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(30))
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
