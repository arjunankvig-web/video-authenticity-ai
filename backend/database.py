from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Database configuration
DATABASE_URL = "sqlite:///./videos.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class VideoRecord(Base):
    __tablename__ = "videos"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    filepath = Column(String)
    upload_date = Column(DateTime, default=datetime.utcnow)
    file_size = Column(Integer)  # in bytes
    duration = Column(Float)  # in seconds
    num_frames = Column(Integer)
    motion_score = Column(Float)
    pose_score = Column(Float)
    physics_score = Column(Float)
    authenticity_score = Column(Float)
    verdict = Column(String)  # "Real" or "AI-Generated"
    detected_entities = Column(Text)  # JSON string
    analysis_status = Column(String, default="pending")  # pending, completed, error


# Create tables
Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
