"""
Database Models
===============
SQLAlchemy models for persistent data storage.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

# Database connection
engine = create_engine("sqlite:///amis.db", echo=False)

class DBPortfolio(Base):
    __tablename__ = 'portfolios'

    id = Column(String, primary_key=True)
    name = Column(String)
    manager = Column(String)
    nav = Column(Float)
    cash_weight = Column(Float)
    benchmark = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    holdings = relationship("DBHolding", back_populates="portfolio")

class DBHolding(Base):
    __tablename__ = 'holdings'

    id = Column(Integer, primary_key=True)
    portfolio_id = Column(String, ForeignKey('portfolios.id'))
    ticker = Column(String)
    name = Column(String)
    sector = Column(String)
    weight = Column(Float)
    quantity = Column(Float)
    price = Column(Float)
    market_value = Column(Float)
    cost_basis = Column(Float)
    unrealized_pnl = Column(Float)
    beta = Column(Float)
    liquidity_score = Column(Float)

    portfolio = relationship("DBPortfolio", back_populates="holdings")

class PriceData(Base):
    __tablename__ = 'price_data'

    id = Column(Integer, primary_key=True)
    ticker = Column(String)
    date = Column(DateTime)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Float)
    returns_1d = Column(Float)
    returns_1w = Column(Float)
    returns_1m = Column(Float)
    returns_3m = Column(Float)
    returns_ytd = Column(Float)

class Report(Base):
    __tablename__ = 'reports'

    id = Column(String, primary_key=True)
    portfolio_id = Column(String, ForeignKey('portfolios.id'))
    generated_at = Column(DateTime, default=datetime.utcnow)
    period = Column(String)
    portfolio_return = Column(Float)
    benchmark_return = Column(Float)
    alpha = Column(Float)
    overview = Column(Text)
    attribution = Column(Text)
    recommendations = Column(Text)

# Create tables
Base.metadata.create_all(engine)