""
Monitoring API for the Aegis Dashboard
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import logging
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from contextlib import asynccontextmanager
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "aegis"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432")
}

# Models
class MetricPoint(BaseModel):
    timestamp: datetime
    value: float

class MetricSeries(BaseModel):
    name: str
    data: List[MetricPoint]

class Alert(BaseModel):
    id: int
    timestamp: datetime
    severity: str
    message: str
    metric: str
    value: float
    acknowledged: bool

class SystemStatus(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    timestamp: datetime

# Database connection pool
class Database:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance._init_pool()
        return cls._instance
    
    def _init_pool(self):
        self.conn_pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            **DB_CONFIG
        )
        logger.info("Database connection pool initialized")
    
    def get_connection(self):
        return self.conn_pool.getconn()
    
    def put_connection(self, conn):
        self.conn_pool.putconn(conn)
    
    def close_all(self):
        self.conn_pool.closeall()
        logger.info("Database connection pool closed")

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up...")
    app.state.db = Database()
    
    # Initialize database schema
    await init_db(app.state.db)
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    app.state.db.close_all()

# Create FastAPI app
app = FastAPI(
    title="Aegis Monitoring API",
    description="API for monitoring the Aegis trading system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependencies
def get_db():
    db = Database()
    conn = db.get_connection()
    try:
        yield conn
    finally:
        db.put_connection(conn)

# Routes
@app.get("/api/health", response_model=SystemStatus)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "uptime_seconds": 0,  # Would be calculated in a real implementation
        "timestamp": datetime.utcnow()
    }

@app.get("/api/metrics", response_model=Dict[str, MetricSeries])
async def get_metrics(
    metric_names: str,
    start_time: datetime = None,
    end_time: datetime = None,
    interval: str = "1m",
    db: psycopg2.extensions.connection = Depends(get_db)
):
    """Get time series metrics"""
    if not start_time:
        start_time = datetime.utcnow() - timedelta(hours=1)
    if not end_time:
        end_time = datetime.utcnow()
    
    metrics = {}
    names = metric_names.split(",")
    
    with db.cursor(cursor_factory=RealDictCursor) as cur:
        for name in names:
            # In a real implementation, this would use proper SQL with time bucketing
            # based on the interval parameter
            cur.execute("""
                SELECT time_bucket(%s, timestamp) as time, 
                       avg(value) as value
                FROM metrics
                WHERE name = %s 
                  AND timestamp BETWEEN %s AND %s
                GROUP BY time
                ORDER BY time
            ""
