# src/api/middleware.py
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
import structlog

logger = structlog.get_logger()

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request logging and tracing"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        # Log request start
        start_time = time.time()
        logger.info(
            "Request started",
            correlation_id=correlation_id,
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else None
        )
        
        # Process request
        response = await call_next(request)
        
        # Log request completion
        process_time = time.time() - start_time
        logger.info(
            "Request completed",
            correlation_id=correlation_id,
            status_code=response.status_code,
            process_time=process_time
        )
        
        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Process-Time"] = str(process_time)
        
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware"""
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old entries
        self.clients = {
            ip: timestamps for ip, timestamps in self.clients.items()
            if any(current_time - t < self.period for t in timestamps)
        }
        
        # Check rate limit
        if client_ip in self.clients:
            self.clients[client_ip] = [
                t for t in self.clients[client_ip] 
                if current_time - t < self.period
            ]
            
            if len(self.clients[client_ip]) >= self.calls:
                return Response(
                    content="Rate limit exceeded",
                    status_code=429,
                    headers={"Retry-After": str(self.period)}
                )
        else:
            self.clients[client_ip] = []
        
        # Record request
        self.clients[client_ip].append(current_time)
        
        return await call_next(request)