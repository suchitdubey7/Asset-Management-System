"""
FastAPI Backend
===============
REST API for the Asset Management System.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from models.database import DBPortfolio, DBHolding, engine
from sqlalchemy.orm import sessionmaker

app = FastAPI(title="AMIS API", version="1.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Session = sessionmaker(bind=engine)

# Pydantic models for API
class HoldingResponse(BaseModel):
    ticker: str
    name: str
    sector: str
    weight: float
    price: float
    market_value: float
    beta: float

class PortfolioResponse(BaseModel):
    id: str
    name: str
    manager: str
    nav: float
    benchmark: str
    holdings: List[HoldingResponse]

@app.get("/")
def read_root():
    return {"message": "AMIS API", "version": "1.0.0"}

@app.get("/portfolio", response_model=PortfolioResponse)
def get_portfolio():
    """Get the current portfolio with holdings."""
    session = Session()
    try:
        portfolio = session.query(DBPortfolio).first()
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        holdings = session.query(DBHolding).filter_by(portfolio_id=portfolio.id).all()
        holdings_data = [
            HoldingResponse(
                ticker=h.ticker,
                name=h.name,
                sector=h.sector,
                weight=h.weight,
                price=h.price,
                market_value=h.market_value,
                beta=h.beta
            ) for h in holdings
        ]

        return PortfolioResponse(
            id=portfolio.id,
            name=portfolio.name,
            manager=portfolio.manager,
            nav=portfolio.nav,
            benchmark=portfolio.benchmark,
            holdings=holdings_data
        )
    finally:
        session.close()

@app.get("/holdings", response_model=List[HoldingResponse])
def get_holdings():
    """Get all holdings."""
    session = Session()
    try:
        holdings = session.query(DBHolding).all()
        return [
            HoldingResponse(
                ticker=h.ticker,
                name=h.name,
                sector=h.sector,
                weight=h.weight,
                price=h.price,
                market_value=h.market_value,
                beta=h.beta
            ) for h in holdings
        ]
    finally:
        session.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)