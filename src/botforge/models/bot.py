from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, declarative_base
import uuid
from datetime import datetime

from sqlalchemy import (
	Column, String, Text, Integer, Boolean, DateTime, ForeignKey, DECIMAL, JSON
)

Base = declarative_base()

class User(Base):
	__tablename__ = "users"
	id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	email = Column(String(255), unique=True, nullable=False)
	password_hash = Column(Text, nullable=False)
	name = Column(String(255))
	created_at = Column(DateTime, default=datetime.utcnow)

	bots = relationship("Bot", back_populates="user")
	subscriptions = relationship("UserSubscription", back_populates="user")


class Subscription(Base):
	__tablename__ = "subscriptions"
	id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	name = Column(String(100), nullable=False)
	max_bots = Column(Integer, nullable=False)
	price_per_month = Column(DECIMAL(10, 2))
	created_at = Column(DateTime, default=datetime.utcnow)

	user_subscriptions = relationship("UserSubscription", back_populates="subscription")


class UserSubscription(Base):
	__tablename__ = "user_subscriptions"
	id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
	subscription_id = Column(UUID(as_uuid=True), ForeignKey("subscriptions.id"))
	start_date = Column(DateTime, default=datetime.utcnow)
	end_date = Column(DateTime)
	is_active = Column(Boolean, default=True)

	user = relationship("User", back_populates="subscriptions")
	subscription = relationship("Subscription", back_populates="user_subscriptions")


class Bot(Base):
	__tablename__ = "bots"
	id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
	name = Column(String(255), nullable=False)
	description = Column(Text)
	is_active = Column(Boolean, default=True)
	created_at = Column(DateTime, default=datetime.utcnow)

	user = relationship("User", back_populates="bots")
	sources = relationship("Source", back_populates="bot")
	conversations = relationship("Conversation", back_populates="bot")
	settings = relationship("Setting", back_populates="bot", uselist=False)
	progress = relationship("BotProgress", back_populates="bot", uselist=False)


class Source(Base):
	__tablename__ = "sources"
	id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id", ondelete="CASCADE"))
	type = Column(String(50), nullable=False)
	url = Column(Text)
	file_path = Column(Text)
	text_content = Column(Text)
	created_at = Column(DateTime, default=datetime.utcnow)

	bot = relationship("Bot", back_populates="sources")


class Conversation(Base):
	__tablename__ = "conversations"
	id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id", ondelete="CASCADE"))
	user_question = Column(Text, nullable=False)
	bot_response = Column(Text)
	timestamp = Column(DateTime, default=datetime.utcnow)

	bot = relationship("Bot", back_populates="conversations")


class Setting(Base):
	__tablename__ = "settings"
	id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id", ondelete="CASCADE"))
	theme = Column(JSON, default=dict)
	welcome_message = Column(Text)
	enable_smart_actions = Column(Boolean, default=True)
	created_at = Column(DateTime, default=datetime.utcnow)

	bot = relationship("Bot", back_populates="settings")


class BotProgress(Base):
	__tablename__ = "bot_progress"
	bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id", ondelete="CASCADE"), primary_key=True)
	step_1_created = Column(Boolean, default=True)
	step_2_sources_added = Column(Boolean, default=False)
	step_3_config_done = Column(Boolean, default=False)
	step_4_tested = Column(Boolean, default=False)
	step_5_embedded = Column(Boolean, default=False)

	bot = relationship("Bot", back_populates="progress")